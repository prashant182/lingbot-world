import gc
import hashlib
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
# import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward_causal, sp_dit_forward_causal
from .distributed.util import get_world_size
from .modules.model_fast import WanModelFast
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE

from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.cam_utils import (
    compute_relative_poses,
    interpolate_camera_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)
from einops import rearrange


class WanI2VFast:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        pipe_dtype=torch.bfloat16,
        local_attn_size=-1,
        sink_size=0,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype
        self.pipe_dtype = pipe_dtype
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        if 'cam' in checkpoint_dir:
            self.control_type = 'cam'
        elif 'act' in checkpoint_dir:
            self.control_type = 'act'

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModelFast from {checkpoint_dir}")
        self.model = WanModelFast.from_pretrained(
            checkpoint_dir,
            subfolder=config.fast_noise_checkpoint,
            torch_dtype=torch.bfloat16,
            control_type=self.control_type,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size)

        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype).to(self.device)

        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

        # T5 prompt-embedding cache. Same-prompt re-encodes hit this dict
        # instead of re-running the umt5-xxl encoder (~360 ms/call).
        # Keyed by sha256(prompt.utf8); value is the list returned by
        # T5EncoderModel.__call__ (already device-resident). Unbounded;
        # callers can clear via `pipe.clear_text_cache()` if needed.
        self._t5_cache: dict[str, list] = {}

        # Reset per generate() and flipped True after the first DiT forward.
        # Passed into model.forward as `cross_attn_first_call` to skip the
        # crossattn_cache["is_init"].item() sync inside WanCrossAttention.
        self._cross_attn_initialized: bool = False

    def clear_text_cache(self):
        """Drop all cached T5 prompt embeddings. Frees ~4 MB per entry."""
        self._t5_cache.clear()

    def prewarm(
        self,
        img,
        max_area: int = 480 * 832,
        frame_num: int = 81,
        chunk_size: int = 3,
        text_seq_len: int = 512,
    ):
        """Opt-in pre-warm. Run one dummy DiT forward at the same shape a
        subsequent generate() call will use, so CUDA kernels are autotuned,
        FSDP all-gathers happen, and Ulysses all-to-alls handshake — all
        outside the timed generate() window.

        Without this call, the first generate() pays a ~7s warmup tax in
        chunk 0 (CUDA lazy init, kernel autotuning, NCCL handshake). On
        8xH100 at 480*832/81 frames, calling prewarm() before the first
        generate() reduces generate()'s wall-clock by ~6.5s (~30%) with
        bit-identical output.

        Idempotent: subsequent calls on the same pipe are no-ops.
        Shape-keyed: if generate() is later invoked with a different shape,
        the autotuner will warm those kernels on demand in chunk 0 (no
        incorrect output, just the tax re-paid once).

        Args:
            img: PIL image or torch tensor — used only for its h/w to match
                generate()'s lat_h/lat_w derivation.
            max_area, frame_num, chunk_size: shape parameters; must match
                the subsequent generate() call to be effective.
            text_seq_len: T5 sequence length (defaults to config.text_len).

        Caller pattern:
            pipe = WanI2VFast(...)
            pipe.prewarm(img, max_area=..., frame_num=...)
            # start your timer here
            video = pipe.generate(prompt, img, ...)
        """
        if getattr(self, "_warmed", False):
            return

        cfg = self.config

        # Match generate()'s shape derivation exactly.
        F = frame_num
        h, w = (img.shape[1], img.shape[2]) if hasattr(img, 'shape') else (img.size[1], img.size[0])
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // cfg.vae_stride[1] //
            cfg.patch_size[1] * cfg.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // cfg.vae_stride[2] //
            cfg.patch_size[2] * cfg.patch_size[2])
        lat_f = (F - 1) // cfg.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))

        frame_seqlen = (lat_h * lat_w) // (cfg.patch_size[1] * cfg.patch_size[2])
        max_seq_len = chunk_size * frame_seqlen
        head_dim = cfg.dim // cfg.num_heads
        local_num_heads = cfg.num_heads // self.sp_size

        if self.local_attn_size > -1:
            kv_size = frame_seqlen * self.local_attn_size
        else:
            kv_size = frame_seqlen * lat_f

        transformer_dtype = self.pipe_dtype
        # generate() folds the VAE spatial stride into the Plücker channel
        # dim via rearrange 'f (h s1) (w s2) c -> (f h w) (c s1 s2)' with
        # s1=s2=vae_stride[1]=8, so control_dim=6 → 6 * 8 * 8 = 384.
        plucker_channels = 6 * cfg.vae_stride[1] * cfg.vae_stride[2]
        # T5 (umt5-xxl) hidden size; cross-attn projects t5_hidden → cfg.dim.
        t5_hidden = 4096

        warmup_self_kv = self._initialize_self_kv_cache(
            num_layers=cfg.num_layers,
            shape=[1, kv_size, local_num_heads, head_dim],
            dtype=transformer_dtype,
            device=self.device)
        warmup_cross_kv = self._initialize_crossattn_cache(
            num_layers=cfg.num_layers,
            shape=[1, text_seq_len, cfg.num_heads, head_dim],
            dtype=transformer_dtype,
            device=self.device)

        # `y` is concat([msk_4ch, vae_latent_16ch]) → 20 channels; combined
        # with latent's 16 ch at patch-embed concat, the DiT sees 36 ch in.
        dummy_latent = torch.zeros(
            16, chunk_size, lat_h, lat_w,
            device=self.device, dtype=torch.float32)
        dummy_y = torch.zeros(
            20, chunk_size, lat_h, lat_w,
            device=self.device, dtype=transformer_dtype)
        dummy_c2ws = torch.zeros(
            1, plucker_channels, chunk_size, lat_h, lat_w,
            device=self.device, dtype=self.param_dtype)
        dummy_context = torch.zeros(
            text_seq_len, t5_hidden,
            device=self.device, dtype=self.param_dtype)
        dummy_t = torch.tensor(
            [500.0], device=self.device, dtype=torch.float32)

        @contextmanager
        def _noop_no_sync():
            yield
        no_sync_model = getattr(self.model, 'no_sync', _noop_no_sync)

        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        t0 = time.perf_counter()

        with torch.amp.autocast('cuda', dtype=self.param_dtype), \
             torch.no_grad(), \
             no_sync_model():
            _ = self.model(
                x=[dummy_latent],
                t=dummy_t,
                context=[dummy_context],
                seq_len=max_seq_len,
                y=[dummy_y],
                dit_cond_dict={"c2ws_plucker_emb": (dummy_c2ws,)},
                kv_cache=warmup_self_kv,
                crossattn_cache=warmup_cross_kv,
                current_start=0,
                max_attention_size=kv_size,
                frame_seqlen=frame_seqlen,
                kv_write_index=torch.arange(
                    0, chunk_size * frame_seqlen,
                    device=self.device, dtype=torch.long),
            )

        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logging.info(f"WanI2VFast.prewarm: {dt_ms:.0f} ms")

        del (warmup_self_kv, warmup_cross_kv, dummy_latent, dummy_y,
             dummy_c2ws, dummy_context, dummy_t)
        torch.cuda.empty_cache()
        self._warmed = True

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward_causal, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward_causal, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, F, H, W]
        xt: the input noisy data with shape [B, C, F, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred

        return x0_pred.to(original_dtype)


    def generate(self,
                 input_prompt,
                 img,
                 action_path=None,
                 chunk_size=3,
                 max_area=480 * 832,
                 frame_num=81,
                 timesteps_index=[0, 179, 358, 679],
                 shift=5.0,
                 seed=-1,
                 offload_model=True,
                 max_sequence_length=512,
                 max_attention_size=None,):
        r"""
        Generates video frames from one OR more user inputs in a single
        batched chunk loop.

        Single-user (backward-compatible) — scalar inputs, scalar output:
            video = pipe.generate("a prompt", pil_image, action_path="examples/03",
                                  seed=42)

        Multi-user (B>1) — list inputs, list output:
            videos = pipe.generate(["prompt A", "prompt B"],
                                   [img_A, img_B],
                                   action_path=["examples/03", "examples/04"],
                                   seed=[42, 123])
            # videos[0] is user A's output, videos[1] is user B's.

        Each per-user input may be passed as a scalar OR a list; scalars
        broadcast across the batch. The shape derivation (lat_h, lat_w,
        lat_f) is shared across the batch — homogeneous batching only.
        Each user keeps its own seeded RNG so noise streams don't cross.

        At batch_size=1, the RNG draw order matches the pre-refactor path
        token-for-token → bit-identical output (gate-tested).

        Returns
        -------
        torch.Tensor (rank-0, single-user input)
            Video frames tensor, shape [C, N, H, W].
        list[torch.Tensor] (rank-0, list inputs)
            One video tensor per user, in input order.
        None (other ranks)
        """
        # ── 0. Input normalization. Track whether the caller passed
        # scalars so we can unpack the output the same way on return.
        was_scalar_input = isinstance(input_prompt, str)
        if was_scalar_input:
            prompts = [input_prompt]
            imgs = [img]
            action_paths = [action_path]
            seeds = [seed]
        else:
            prompts = list(input_prompt)
            imgs = list(img) if isinstance(img, list) else [img] * len(prompts)
            action_paths = (list(action_path) if isinstance(action_path, list)
                            else [action_path] * len(prompts))
            seeds = list(seed) if isinstance(seed, list) else [seed] * len(prompts)
        batch_size = len(prompts)
        assert len(imgs) == batch_size, \
            f"img count {len(imgs)} != batch {batch_size}"
        assert len(action_paths) == batch_size
        assert len(seeds) == batch_size

        # ── 1. Shape derivation from user 0 (homogeneous batching: all
        # users share H, W, F, chunk_size). Frame count is clamped by the
        # shortest action path's poses.
        F = frame_num
        if action_paths[0] is not None:
            c2ws_first = np.load(os.path.join(action_paths[0], "poses.npy"))
            len_c2ws_first = ((len(c2ws_first) - 1) // 4) * 4 + 1
            F = min(((F - 1) // 4) * 4 + 1, len_c2ws_first)

        # All imgs are assumed same shape; preprocess user 0 to derive
        # spatial dims, then re-use the derivation for all users.
        img0_t = TF.to_tensor(imgs[0]).sub_(0.5).div_(0.5).to(self.device)
        h0, w0 = img0_t.shape[1:]
        aspect_ratio = h0 / w0
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        max_seq_len = chunk_size * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        # Reset per-generate state: cross-attn K/V cache will be freshly
        # initialized below; the first DiT forward must compute and store.
        self._cross_attn_initialized = False

        # ── 2. Timesteps (shared across users; scheduler is stateless
        # at inference). Mask is also shared (same F/lat_h/lat_w).
        self.scheduler.set_timesteps(self.num_train_timesteps, shift=shift)
        timesteps = self.scheduler.timesteps[timesteps_index]

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # ── 3. Per-user preprocess. Each iteration replicates the single-
        # user path exactly: seed → noise, prompt → T5 context, image →
        # VAE-encoded y, action_path → c2ws_plucker_emb. At B=1, exactly
        # one iteration runs and produces the same tensors as the
        # pre-refactor code, in the same RNG order.
        per_user_noise = []
        per_user_context = []
        per_user_y = []
        per_user_c2ws = []
        seed_gens = []  # kept alive across chunk loop for per-step add_noise.

        for ui in range(batch_size):
            # Seeded RNG per user.
            user_seed = seeds[ui] if seeds[ui] >= 0 else random.randint(0, sys.maxsize)
            sg = torch.Generator(device=self.device)
            sg.manual_seed(user_seed)
            seed_gens.append(sg)
            per_user_noise.append(torch.randn(
                16, lat_f, lat_h, lat_w,
                dtype=torch.float32, generator=sg, device=self.device))

            # T5 prompt encoding (uses the existing cache).
            cache_key = hashlib.sha256(prompts[ui].encode('utf-8')).hexdigest()
            if cache_key in self._t5_cache:
                ctx_list = self._t5_cache[cache_key]
            else:
                if not self.t5_cpu:
                    self.text_encoder.model.to(self.device)
                    ctx_list = self.text_encoder([prompts[ui]], self.device)
                    if offload_model:
                        self.text_encoder.model.cpu()
                else:
                    ctx_list = self.text_encoder([prompts[ui]], torch.device('cpu'))
                    ctx_list = [t.to(self.device) for t in ctx_list]
                self._t5_cache[cache_key] = ctx_list
            per_user_context.append(ctx_list[0])

            # Image VAE encode.
            img_t = (img0_t if ui == 0
                     else TF.to_tensor(imgs[ui]).sub_(0.5).div_(0.5).to(self.device))
            y_ui = self.vae.encode([
                torch.concat([
                    torch.nn.functional.interpolate(
                        img_t[None].cpu(), size=(h, w),
                        mode='bicubic').transpose(0, 1),
                    torch.zeros(3, F - 1, h, w)
                ], dim=1).to(self.device)
            ])[0]
            y_ui = torch.concat([msk, y_ui])
            per_user_y.append(y_ui)

            # Camera conditioning (only if this user has an action path).
            ap = action_paths[ui]
            if ap is not None:
                c2ws_ui = np.load(os.path.join(ap, "poses.npy"))[:F]
                Ks_ui = torch.from_numpy(np.load(
                    os.path.join(ap, "intrinsics.npy"))).float()
                Ks_ui = get_Ks_transformed(Ks_ui,
                                            height_org=480, width_org=832,
                                            height_resize=h, width_resize=w,
                                            height_final=h, width_final=w)[0]
                len_c2ws_ = int((len(c2ws_ui) - 1) // 4) + 1
                len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
                c2ws_infer = interpolate_camera_poses(
                    src_indices=np.linspace(0, len(c2ws_ui) - 1, len(c2ws_ui)),
                    src_rot_mat=c2ws_ui[:, :3, :3],
                    src_trans_vec=c2ws_ui[:, :3, 3],
                    tgt_indices=np.linspace(0, len(c2ws_ui) - 1, len_c2ws_),
                )
                c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
                Ks_ui = Ks_ui.repeat(len(c2ws_infer), 1)
                c2ws_infer = c2ws_infer.to(self.device)
                Ks_ui = Ks_ui.to(self.device)
                if self.control_type == 'act':
                    wasd_ui = np.load(os.path.join(ap, "action.npy"))[:F]
                    wasd_ui = torch.from_numpy(wasd_ui[::4]).float().to(self.device)
                else:
                    wasd_ui = None
                only_rays_d = wasd_ui is not None
                cam_emb_ui = get_plucker_embeddings(
                    c2ws_infer, Ks_ui, h, w, only_rays_d=only_rays_d)
                cam_emb_ui = rearrange(
                    cam_emb_ui,
                    'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                    c1=int(h // lat_h), c2=int(w // lat_w))[None]  # [1, fhw, C]
                cam_emb_ui = rearrange(
                    cam_emb_ui, 'b (f h w) c -> b c f h w',
                    f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
                if wasd_ui is not None:
                    wasd_t = wasd_ui[:, None, None, :].repeat(1, h, w, 1)
                    wasd_t = rearrange(
                        wasd_t,
                        'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                        c1=int(h // lat_h), c2=int(w // lat_w))[None]
                    wasd_t = rearrange(
                        wasd_t, 'b (f h w) c -> b c f h w',
                        f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
                    cam_emb_ui = torch.cat([cam_emb_ui, wasd_t], dim=1)
                per_user_c2ws.append(cam_emb_ui)
            else:
                per_user_c2ws.append(None)

        # ── 4. Stack per-user tensors into B=N batched form for the chunk
        # loop. At B=1 these reduce to one-element stacks (functionally
        # the same as the pre-refactor scalar code).
        noise_batched = torch.stack(per_user_noise, dim=0)        # [B, 16, lat_f, lat_h, lat_w]
        y_batched = torch.stack(per_user_y, dim=0)                 # [B, 20, F, lat_h, lat_w]
        has_cam = per_user_c2ws[0] is not None
        if has_cam:
            assert all(c is not None for c in per_user_c2ws), \
                "Mixed action_path=None and not-None across batch is not supported"
            c2ws_batched = torch.cat(per_user_c2ws, dim=0)          # [B, C, F, lat_h, lat_w]

        @contextmanager
        def noop_no_sync():
            yield
        no_sync_model = getattr(self.model, 'no_sync', noop_no_sync)

        # KV cache shapes at batch_size = B.
        model_args = self.model.config
        transformer_dtype = self.pipe_dtype
        frame_seqlen = int(noise_batched.shape[-2] * noise_batched.shape[-1] // 4)
        if self.local_attn_size > -1:
            kv_size = frame_seqlen * self.local_attn_size
        else:
            kv_size = frame_seqlen * lat_f
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.sp_size
        self_kv_cache = self._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=[batch_size, kv_size, local_num_heads, head_dim],
            dtype=transformer_dtype, device=self.device)
        cross_kv_cache = self._initialize_crossattn_cache(
            num_layers=model_args.num_layers,
            shape=[batch_size, max_sequence_length, model_args.num_heads, head_dim],
            dtype=transformer_dtype, device=self.device)

        # ── 5. Chunk loop with batched inputs. Same structure as before
        # but every per-step tensor carries the batch dim.
        with (torch.amp.autocast('cuda', dtype=self.param_dtype),
              torch.no_grad(),
              no_sync_model()):
            # Split along the FRAME dim (dim=2 now that batch is dim=0).
            latents_chunk = noise_batched.split(chunk_size, dim=2)
            condition_chunk = y_batched.split(chunk_size, dim=2)
            if has_cam:
                c2ws_chunk = c2ws_batched.split(chunk_size, dim=2)
            num_inference_chunk = len(latents_chunk)
            pred_latent_chunks = []
            for chunk_id in tqdm(range(num_inference_chunk)):
                current_latent = latents_chunk[chunk_id]        # [B, 16, csz, h, w]
                current_condition = condition_chunk[chunk_id]    # [B, 20, csz, h, w]

                if has_cam:
                    c_chunk = c2ws_chunk[chunk_id]               # [B, C, csz, h, w]
                    # The block expects a tuple of per-batch tensors.
                    dit_cond_dict = {
                        "c2ws_plucker_emb":
                            tuple(c_chunk[bi:bi+1] for bi in range(batch_size)),
                    }
                else:
                    dit_cond_dict = None

                current_start_int = chunk_id * chunk_size * frame_seqlen
                current_end_int = current_start_int + chunk_size * frame_seqlen
                kv_write_index = torch.arange(
                    current_start_int, current_end_int,
                    device=self.device, dtype=torch.long)
                kwargs = {
                    'context': per_user_context,
                    'seq_len': max_seq_len,
                    'y': [current_condition[bi] for bi in range(batch_size)],
                    'dit_cond_dict': dit_cond_dict,
                    'kv_cache': self_kv_cache,
                    'crossattn_cache': cross_kv_cache,
                    'current_start': current_start_int,
                    'max_attention_size': kv_size if max_attention_size is None else max_attention_size,
                    'frame_seqlen': frame_seqlen,
                    'kv_write_index': kv_write_index,
                }

                if offload_model:
                    torch.cuda.empty_cache()

                for timestep_idx in range(len(timesteps)):
                    latent_input_list = [
                        current_latent[bi].to(self.device)
                        for bi in range(batch_size)
                    ]
                    # Timestep tensor must have batch dim B (one per user;
                    # all the same value for homogeneous batching).
                    timestep = timesteps[timestep_idx].repeat(batch_size).to(self.device)

                    noise_pred_list = self.model(
                        x=latent_input_list, t=timestep,
                        cross_attn_first_call=not self._cross_attn_initialized,
                        **kwargs)
                    self._cross_attn_initialized = True
                    # Stack list of [16, csz, h, w] → [B, 16, csz, h, w]
                    noise_pred = torch.stack(noise_pred_list, dim=0)

                    if offload_model:
                        torch.cuda.empty_cache()

                    # x0 per user (scheduler ops are elementwise → batch-safe).
                    x0_list = [
                        self._convert_flow_pred_to_x0(
                            flow_pred=noise_pred[bi],
                            xt=current_latent[bi],
                            timestep=timesteps[timestep_idx],
                            scheduler=self.scheduler,
                        )
                        for bi in range(batch_size)
                    ]
                    x0 = torch.stack(x0_list, dim=0)               # [B, 16, csz, h, w]

                    if timestep_idx < len(timesteps) - 1:
                        next_timestep = timesteps[timestep_idx + 1]
                        # Per-user seeded noise — preserves RNG stream
                        # ordering token-for-token vs the single-user path
                        # at B=1.
                        step_noise = torch.stack([
                            torch.randn(x0[bi].shape, generator=seed_gens[bi],
                                        device=x0.device, dtype=x0.dtype)
                            for bi in range(batch_size)
                        ], dim=0)
                        current_latent = self.scheduler.add_noise(
                            x0, step_noise, next_timestep)
                    else:
                        break

                pred_latent_chunks.append(x0)

                # KV cache update forward (t=0).
                timestep = (timesteps[-1] * 0.0).repeat(batch_size).to(self.device)
                self.model(
                    x=[x0[bi] for bi in range(batch_size)],
                    t=timestep, cross_attn_first_call=False, **kwargs)

            # Concatenate chunks along the FRAME dim (dim=2 for batched).
            pred_latent_full = torch.cat(pred_latent_chunks, dim=2)    # [B, 16, F, h, w]

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            # ── 6. Per-user VAE decode on rank-0. VAE is causal-temporal
            # internally; decode users sequentially (E4 is the parallel
            # decode follow-up).
            if self.rank == 0:
                videos = [self.vae.decode([pred_latent_full[bi]])[0]
                          for bi in range(batch_size)]

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if self.rank != 0:
            return None
        if was_scalar_input:
            return videos[0]
        return videos

    def _initialize_self_kv_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a Per-GPU KV cache for the SelfAttn.
        """
        self_kv_cache = []
        for _ in range(num_layers):
            self_kv_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'global_end_index': torch.tensor([0], dtype=torch.long, device=device),
                'local_end_index': torch.tensor([0], dtype=torch.long, device=device)
            })

        return self_kv_cache


    def _initialize_crossattn_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a per-GPU cross-attention cache.
        """
        crossattn_cache = []
        for _ in range(num_layers):
            crossattn_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'is_init': torch.tensor(0, dtype=torch.int32, device=device),
            })

        return crossattn_cache