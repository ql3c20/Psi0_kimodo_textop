#!/usr/bin/env python3
"""Psi0-style prefix-RTC adapter for GR00T N1.7 (no edits to Isaac-GR00T sources).

Enable via serve ``--prefix-rtc``.  After loading an official ``Gr00tPolicy``,
``apply_prefix_rtc(policy)`` monkeypatches:

1. Action encoder: accept ``(B, T)`` timesteps (prefix frames at t=0).
2. DiT / AlternateVLDiT AdaLN: accept per-token ``temb (B, T, D)``.
3. Inference ``get_action_with_features``: hard-rewrite overlap every denoising
   step (no ``vel_strength`` soft freeze), with a selectable prefix timestep:
   ``legacy_zero`` for existing checkpoints or ``groot_clean`` for GR00T's
   clean flow endpoint.

Native GR00T (no ``--prefix-rtc``) is unchanged.  Weights are reused as-is;
no new parameters are introduced.

Training helper ``apply_prefix_rtc_train(action_head)`` mirrors the selected
per-frame prefix timestep + token AdaLN conditioning for RTC finetuning.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional
import types

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature


_PREFIX_TIMESTEP_MODES = {"legacy_zero", "groot_clean"}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _sync_for_timing(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _normalize_prefix_timestep_mode(mode: str) -> str:
    aliases = {
        "legacy": "legacy_zero",
        "zero": "legacy_zero",
        "groot": "groot_clean",
        "groot_style": "groot_clean",
        "clean": "groot_clean",
    }
    normalized = aliases.get(str(mode).strip().lower(), str(mode).strip().lower())
    if normalized not in _PREFIX_TIMESTEP_MODES:
        raise ValueError(
            "prefix_rtc_timestep_mode must be one of "
            f"{sorted(_PREFIX_TIMESTEP_MODES)}, got {mode!r}"
        )
    return normalized


def _resolve_prefix_timestep_mode(action_head, explicit_mode: str | None) -> str:
    if explicit_mode is None:
        explicit_mode = getattr(
            action_head.config,
            "prefix_rtc_timestep_mode",
            "legacy_zero",
        )
    return _normalize_prefix_timestep_mode(explicit_mode)


def _prefix_timestep_bucket(action_head) -> int:
    mode = getattr(action_head, "_prefix_rtc_timestep_mode", "legacy_zero")
    if mode == "legacy_zero":
        return 0
    if mode == "groot_clean":
        # GR00T uses x_t=(1-t)*noise+t*action, so clean data is t=1.
        # Training discretizes t<1 into [0, num_timestep_buckets-1]; use the
        # last in-distribution bucket instead of the unseen exact endpoint.
        return max(int(action_head.num_timestep_buckets) - 1, 0)
    raise RuntimeError(f"Unsupported resolved prefix timestep mode: {mode!r}")


# ---------------------------------------------------------------------------
# Low-level patches (shape-compatible with official weights)
# ---------------------------------------------------------------------------


def _patched_action_encoder_forward(self, actions, timesteps, cat_ids):
    """Like MultiEmbodimentActionEncoder.forward, but also accepts (B, T)."""
    from gr00t.model.modules.embodiment_conditioned_mlp import swish

    B, T, _ = actions.shape
    if timesteps.dim() == 1 and timesteps.shape[0] == B:
        timesteps = timesteps.unsqueeze(1).expand(-1, T)
    elif timesteps.dim() == 2 and timesteps.shape == (B, T):
        pass
    else:
        raise ValueError(
            f"Expected timesteps shape (B,) or (B, T)=({B}, {T}), got {tuple(timesteps.shape)}"
        )

    a_emb = self.W1(actions, cat_ids)
    tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
    x = torch.cat([a_emb, tau_emb], dim=-1)
    x = swish(self.W2(x, cat_ids))
    x = self.W3(x, cat_ids)
    return x


def _patched_timestep_encoder_forward(self, timesteps):
    """Encode (B,) or (B, T) timesteps -> (B, D) or (B, T, D)."""
    dtype = next(self.parameters()).dtype
    if timesteps.dim() == 1:
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(timesteps_proj)
    if timesteps.dim() == 2:
        B, T = timesteps.shape
        flat = timesteps.reshape(B * T)
        timesteps_proj = self.time_proj(flat).to(dtype)
        emb = self.timestep_embedder(timesteps_proj)  # (B*T, D)
        return emb.reshape(B, T, -1)
    raise ValueError(f"Expected timesteps dim 1 or 2, got shape {tuple(timesteps.shape)}")


def _patched_ada_layer_norm_forward(
    self,
    x: torch.Tensor,
    temb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """AdaLN with batch-global (B, D) or per-token (B, T, D) temb."""
    if temb is None:
        raise ValueError("AdaLayerNorm requires temb")
    if temb.dim() == 2:
        # Official path: (B, D) -> broadcast over sequence.
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]
    if temb.dim() == 3:
        # Per-token: (B, T, D) modulators aligned with x.
        if temb.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"per-token temb shape {tuple(temb.shape)} incompatible with x {tuple(x.shape)}"
            )
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift
    raise ValueError(f"Expected temb dim 2 or 3, got shape {tuple(temb.shape)}")


def _apply_output_adaln(module, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
    """Final DiT output AdaLN supporting (B, D) or (B, T, D) temb."""
    if temb.dim() == 2:
        shift, scale = module.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        return module.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
    if temb.dim() == 3:
        if temb.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                f"per-token temb {tuple(temb.shape)} vs hidden {tuple(hidden_states.shape)}"
            )
        shift, scale = module.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        return module.norm_out(hidden_states) * (1 + scale) + shift
    raise ValueError(f"Expected temb dim 2 or 3, got {tuple(temb.shape)}")


def _patched_dit_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: Optional[torch.LongTensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_all_hidden_states: bool = False,
    encoder_kv_cache: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]] = None,
):
    temb = self.timestep_encoder(timestep)
    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()
    all_hidden_states = [hidden_states]

    for idx, block in enumerate(self.transformer_blocks):
        if idx % 2 == 1 and self.config.interleave_self_attention:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                temb=temb,
            )
        else:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                temb=temb,
                encoder_kv_cache=encoder_kv_cache,
                encoder_kv_cache_key=idx,
            )
        all_hidden_states.append(hidden_states)

    hidden_states = _apply_output_adaln(self, hidden_states, temb)
    if return_all_hidden_states:
        return self.proj_out_2(hidden_states), all_hidden_states
    return self.proj_out_2(hidden_states)


def _patched_alternate_vl_dit_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: Optional[torch.LongTensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_all_hidden_states: bool = False,
    image_mask: Optional[torch.Tensor] = None,
    backbone_attention_mask: Optional[torch.Tensor] = None,
    encoder_kv_cache: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]] = None,
):
    assert image_mask is not None, "Image mask is required"
    temb = self.timestep_encoder(timestep)
    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()

    image_attention_mask = image_mask & backbone_attention_mask
    non_image_attention_mask = (~image_mask) & backbone_attention_mask
    all_hidden_states = [hidden_states]
    assert self.config.interleave_self_attention, "Interleave self attention must be enabled"

    for idx, block in enumerate(self.transformer_blocks):
        if idx % 2 == 1:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                temb=temb,
            )
        else:
            if idx % (2 * self.attend_text_every_n_blocks) == 0:
                curr_encoder_attention_mask = non_image_attention_mask
            else:
                curr_encoder_attention_mask = image_attention_mask
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=curr_encoder_attention_mask,
                temb=temb,
                encoder_kv_cache=encoder_kv_cache,
                encoder_kv_cache_key=idx,
            )
        all_hidden_states.append(hidden_states)

    hidden_states = _apply_output_adaln(self, hidden_states, temb)
    if return_all_hidden_states:
        return self.proj_out_2(hidden_states), all_hidden_states
    return self.proj_out_2(hidden_states)


# ---------------------------------------------------------------------------
# Inference: hard rewrite + selectable prefix timestep convention
# ---------------------------------------------------------------------------


@torch.no_grad()
def _prefix_rtc_get_action_with_features(
    self,
    backbone_features: torch.Tensor,
    state_features: torch.Tensor,
    embodiment_id: torch.Tensor,
    backbone_output: BatchFeature,
    action_input: BatchFeature,
    options: dict[str, Any] | None = None,
    timing: dict[str, float] | None = None,
    timing_sync_cuda: bool = False,
) -> BatchFeature:
    """Flow-matching decode with optional Psi0-style prefix RTC."""
    vl_embeds = backbone_features
    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    action_horizon = self.config.action_horizon

    actions = torch.randn(
        size=(batch_size, action_horizon, self.action_dim),
        dtype=vl_embeds.dtype,
        device=device,
    )
    dt = 1.0 / self.num_inference_timesteps
    use_dit_kv_cache = (
        bool(options.get("dit_cross_attn_kv_cache"))
        if options is not None and "dit_cross_attn_kv_cache" in options
        else _env_flag("GR00T_DIT_CROSS_ATTN_KV_CACHE", False)
    )
    dit_encoder_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = (
        {} if use_dit_kv_cache else None
    )
    if timing is not None:
        timing["dit_kv_cache_enabled"] = float(use_dit_kv_cache)
        _sync_for_timing(device, timing_sync_cuda)
        sampler_start = time.perf_counter()

    rtc_prefix = None
    rtc_overlap_steps = 0
    if "action" in action_input:
        assert options is not None, "options is required for RTC"
        assert "action_horizon" in options, "action_horizon is not in options"
        assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
        action_horizon_before_padding = int(options["action_horizon"])
        rtc_overlap_steps = int(options["rtc_overlap_steps"])
        if not (0 < rtc_overlap_steps < action_horizon):
            raise ValueError(
                f"rtc_overlap_steps must be in (0, {action_horizon}), got {rtc_overlap_steps}"
            )
        rtc_prefix = action_input["action"][
            :,
            action_horizon_before_padding - rtc_overlap_steps : action_horizon_before_padding,
            :,
        ].to(dtype=actions.dtype, device=device)
        actions[:, :rtc_overlap_steps, :] = rtc_prefix

    for step in range(self.num_inference_timesteps):
        if rtc_prefix is not None:
            actions[:, :rtc_overlap_steps, :] = rtc_prefix

        t_cont = step / float(self.num_inference_timesteps)
        t_discretized = int(t_cont * self.num_timestep_buckets)
        t_global = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=device, dtype=torch.long
        )

        if timing is not None:
            _sync_for_timing(device, timing_sync_cuda)
            start = time.perf_counter()
        if rtc_prefix is not None:
            # Action tokens: prefix uses the selected endpoint convention;
            # suffix uses the current GR00T denoising timestep.
            t_action = t_global[:, None].expand(-1, action_horizon).clone()
            t_action[:, :rtc_overlap_steps] = _prefix_timestep_bucket(self)
            action_features = self.action_encoder(actions, t_action, embodiment_id)
            # sa_embs = [state | actions]; state uses global t, actions use t_action.
            t_sa = torch.cat([t_global[:, None], t_action], dim=1)  # (B, 1+Tp)
        else:
            action_features = self.action_encoder(actions, t_global, embodiment_id)
            t_sa = t_global

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        if timing is not None:
            _sync_for_timing(device, timing_sync_cuda)
            timing["action_encoder_ms"] = timing.get("action_encoder_ms", 0.0) + (
                time.perf_counter() - start
            ) * 1000.0

        sa_embs = torch.cat((state_features, action_features), dim=1)

        if timing is not None:
            _sync_for_timing(device, timing_sync_cuda)
            start = time.perf_counter()
        if self.config.use_alternate_vl_dit:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=t_sa,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
                encoder_kv_cache=dit_encoder_kv_cache,
            )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=t_sa,
                encoder_kv_cache=dit_encoder_kv_cache,
            )
        if timing is not None:
            _sync_for_timing(device, timing_sync_cuda)
            timing["dit_ms"] = timing.get("dit_ms", 0.0) + (
                time.perf_counter() - start
            ) * 1000.0
            start = time.perf_counter()

        pred = self.action_decoder(model_output, embodiment_id)
        if timing is not None:
            _sync_for_timing(device, timing_sync_cuda)
            timing["action_decoder_ms"] = timing.get("action_decoder_ms", 0.0) + (
                time.perf_counter() - start
            ) * 1000.0
        pred_velocity = pred[:, -action_horizon :]
        actions = actions + dt * pred_velocity

        if rtc_prefix is not None:
            actions[:, :rtc_overlap_steps, :] = rtc_prefix

    if timing is not None:
        _sync_for_timing(device, timing_sync_cuda)
        timing["sampler_ms"] = (time.perf_counter() - sampler_start) * 1000.0
        timing["dit_kv_cache_entries"] = float(len(dit_encoder_kv_cache or {}))

    return BatchFeature(
        data={
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
    )


# ---------------------------------------------------------------------------
# Training: per-frame prefix timestep + token AdaLN on clean-prefix RTC
# ---------------------------------------------------------------------------


def _prefix_rtc_compute_loss(self, backbone_output: BatchFeature, action_input: BatchFeature):
    """Action-head training forward with prefix-rtc time conditioning.

    Mirrors official train_rtc (clean prefix + loss mask) and additionally:
    - action encoder gets a per-frame prefix timestep selected by mode
    - DiT gets (B, 1+Tp) timesteps for token AdaLN
    """
    self.set_frozen_modules_to_eval_mode()

    backbone_output = self.process_backbone_output(backbone_output)
    vl_embeds = backbone_output.backbone_features
    device = vl_embeds.device
    embodiment_id = action_input.embodiment_id

    state = action_input.state
    assert state.shape[1] == self.config.state_history_length, "current_T != state_history_length"
    state = state.view(state.shape[0], 1, -1)
    state_features = self.state_encoder(state, embodiment_id)

    if self.training and self.state_dropout_prob > 0:
        do_dropout = (
            torch.rand(state_features.shape[0], device=state_features.device)
            < self.state_dropout_prob
        )
        do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
        state_features = state_features * (1 - do_dropout)

    actions = action_input.action
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
    t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
    t = t[:, None, None]
    noisy_trajectory = (1 - t) * noise + t * actions
    velocity = actions - noise

    rtc_prefix_mask = None
    if self.training and getattr(self.config, "train_rtc", False):
        min_delay = int(getattr(self.config, "train_rtc_min_delay", 0))
        max_delay = int(getattr(self.config, "train_rtc_max_delay", 8))
        min_delay = max(0, min(min_delay, actions.shape[1]))
        max_delay = max(0, min(max_delay, actions.shape[1]))
        if max_delay > min_delay:
            delays = torch.randint(
                low=min_delay,
                high=max_delay,
                size=(actions.shape[0],),
                device=actions.device,
                dtype=torch.long,
            )
            rtc_prefix_mask = (
                torch.arange(actions.shape[1], device=actions.device)[None, :] < delays[:, None]
            )
            noisy_trajectory = torch.where(
                rtc_prefix_mask[:, :, None],
                actions,
                noisy_trajectory,
            )

    t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
    if rtc_prefix_mask is not None:
        t_action = t_discretized[:, None].expand(-1, actions.shape[1]).clone()
        prefix_t = torch.full_like(t_action, _prefix_timestep_bucket(self))
        t_action = torch.where(rtc_prefix_mask, prefix_t, t_action)
        action_features = self.action_encoder(noisy_trajectory, t_action, embodiment_id)
        t_sa = torch.cat([t_discretized[:, None], t_action], dim=1)
    else:
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
        t_sa = t_discretized

    if self.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs

    sa_embs = torch.cat((state_features, action_features), dim=1)
    vl_attn_mask = backbone_output.backbone_attention_mask

    if self.config.use_alternate_vl_dit:
        model_output, _ = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_sa,
            return_all_hidden_states=True,
            image_mask=backbone_output.image_mask,
            backbone_attention_mask=backbone_output.backbone_attention_mask,
        )
    else:
        model_output, _ = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_sa,
            return_all_hidden_states=True,
        )

    pred = self.action_decoder(model_output, embodiment_id)
    pred_actions = pred[:, -actions.shape[1] :]

    action_mask = action_input.action_mask
    if rtc_prefix_mask is not None:
        action_mask = action_mask * (~rtc_prefix_mask)[:, :, None].to(dtype=action_mask.dtype)
    action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
    loss = action_loss.sum() / (action_mask.sum() + 1e-6)

    return {
        "loss": loss,
        "action_loss": action_loss,
        "action_mask": action_mask,
        "backbone_features": vl_embeds,
        "state_features": state_features,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _patch_dit_stack(dit_module) -> None:
    """Patch TimestepEncoder / AdaLayerNorm / DiT forwards on a live DiT."""
    from gr00t.model.modules import dit as dit_mod

    # Class-level AdaLN so every block.norm1 picks it up.
    dit_mod.AdaLayerNorm.forward = _patched_ada_layer_norm_forward

    dit_module.timestep_encoder.forward = types.MethodType(
        _patched_timestep_encoder_forward, dit_module.timestep_encoder
    )

    # Bind the correct forward for DiT vs AlternateVLDiT.
    from gr00t.model.modules.dit import AlternateVLDiT

    if isinstance(dit_module, AlternateVLDiT):
        dit_module.forward = types.MethodType(_patched_alternate_vl_dit_forward, dit_module)
    else:
        dit_module.forward = types.MethodType(_patched_dit_forward, dit_module)


def apply_prefix_rtc(
    policy,
    prefix_timestep_mode: str | None = None,
) -> None:
    """Enable prefix-rtc inference on a loaded ``Gr00tPolicy`` (in-place)."""
    model = policy.model
    action_head = model.action_head
    resolved_mode = _resolve_prefix_timestep_mode(action_head, prefix_timestep_mode)
    action_head._prefix_rtc_timestep_mode = resolved_mode

    action_head.action_encoder.forward = types.MethodType(
        _patched_action_encoder_forward, action_head.action_encoder
    )
    _patch_dit_stack(action_head.model)
    action_head.get_action_with_features = types.MethodType(
        _prefix_rtc_get_action_with_features, action_head
    )

    # Official Gr00tPolicy.get_action still requires rtc_frozen_steps / rtc_ramp_rate
    # when building model_options.  Prefix-rtc does not use them; fill harmless
    # defaults so callers only need rtc_prev_action / rtc_overlap_steps / action_horizon.
    _orig_get_action = policy.get_action

    def _get_action_prefix_rtc(observation, options=None):
        if options is not None and options.get("rtc_prev_action") is not None:
            options = dict(options)
            options.setdefault("rtc_frozen_steps", 0)
            options.setdefault("rtc_ramp_rate", 1.0)
        return _orig_get_action(observation, options)

    policy.get_action = _get_action_prefix_rtc
    policy._prefix_rtc_enabled = True
    action_head._prefix_rtc_enabled = True
    print(
        "[prefix-rtc] enabled on policy: hard-rewrite overlap + "
        f"prefix_timestep_mode={resolved_mode} "
        f"(bucket={_prefix_timestep_bucket(action_head)}) + DiT per-token AdaLN"
    )


def apply_prefix_rtc_train(
    action_head,
    prefix_timestep_mode: str | None = None,
) -> None:
    """Enable prefix-rtc training conditioning on a ``Gr00tN1d7ActionHead``.

    Requires ``config.train_rtc=True`` for clean-prefix + loss-mask behavior.
    Additionally routes the selected per-frame prefix timestep through the
    action encoder and DiT.
    """
    resolved_mode = _resolve_prefix_timestep_mode(action_head, prefix_timestep_mode)
    action_head._prefix_rtc_timestep_mode = resolved_mode
    action_head.action_encoder.forward = types.MethodType(
        _patched_action_encoder_forward, action_head.action_encoder
    )
    _patch_dit_stack(action_head.model)
    # Official action head uses ``forward`` -> compute loss; bind our training path.
    if hasattr(action_head, "forward"):
        # Gr00tN1d7ActionHead.forward typically calls the loss path; replace the
        # internal method that builds features when present, else wrap forward.
        action_head._prefix_rtc_compute_loss = types.MethodType(
            _prefix_rtc_compute_loss, action_head
        )

        original_forward = action_head.forward

        def _forward_with_prefix_rtc(self, backbone_output, action_input):
            return self._prefix_rtc_compute_loss(backbone_output, action_input)

        action_head.forward = types.MethodType(_forward_with_prefix_rtc, action_head)
        action_head._prefix_rtc_original_forward = original_forward
    action_head._prefix_rtc_train_enabled = True
    print(
        "[prefix-rtc] train hook enabled: clean prefix + "
        f"prefix_timestep_mode={resolved_mode} "
        f"(bucket={_prefix_timestep_bucket(action_head)}) + "
        "DiT per-token AdaLN (set config.train_rtc=True)"
    )
