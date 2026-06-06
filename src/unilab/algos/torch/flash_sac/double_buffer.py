"""FlashSAC builder for the CPU-pinned double-buffer replay path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.flash_sac.learner import FlashSACLearner
from unilab.algos.torch.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner
from unilab.training import create_env, ensure_registries, resolve_task_checkpoint_path
from unilab.training.seed import apply_training_seed
from unilab.utils.device import get_default_device


def _validate_flashsac_double_buffer_runtime(
    cfg: DictConfig,
    *,
    device: str,
    replay_prefetch_mode: str,
) -> None:
    _ = device
    if cfg.training.num_gpus > 1:
        raise ValueError("FlashSAC-B cpu_pinned_double_buffer is single-GPU only")
    if cfg.training.no_sync_collection:
        raise ValueError("FlashSAC-B cpu_pinned_double_buffer requires synchronized collection")
    if replay_prefetch_mode != "one_tick":
        raise ValueError(
            "FlashSAC-B cpu_pinned_double_buffer requires replay_prefetch_mode='one_tick'"
        )
    if cfg.algo.algo_params.n_step != 1:
        raise ValueError("FlashSAC-B initially supports n_step=1 only")


def build_flashsac_double_buffer_runner(
    cfg: DictConfig,
    *,
    env_cfg_override: dict[str, Any] | None,
    replay_prefetch_mode: str,
    verbose_metrics: bool,
) -> DoubleBufferOffPolicyRunner:
    """Build FlashSAC with the opt-in CPU-pinned double-buffer replay pipeline."""
    from unilab.base.observations import get_obs_dims

    ensure_registries()
    apply_training_seed(cfg.algo.seed, torch_runtime=True, cuda=True)
    device = cfg.training.device or get_default_device()
    _validate_flashsac_double_buffer_runtime(
        cfg,
        device=device,
        replay_prefetch_mode=replay_prefetch_mode,
    )

    env = create_env(cfg, num_envs=1, env_cfg_override=env_cfg_override)
    try:
        obs_dim, critic_obs_dim = get_obs_dims(env.obs_groups_spec)
        action_shape = env.action_space.shape
        assert action_shape is not None
        action_dim = int(action_shape[0])
    finally:
        env.close()

    learner = FlashSACLearner(
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_obs_dim=critic_obs_dim,
        device=device,
        gamma=cfg.algo.gamma,
        tau=cfg.algo.tau,
        actor_lr=cfg.algo.actor_lr,
        critic_lr=cfg.algo.critic_lr,
        actor_hidden_dim=cfg.algo.actor_hidden_dim,
        critic_hidden_dim=cfg.algo.critic_hidden_dim,
        actor_num_blocks=cfg.algo.algo_params.actor_num_blocks,
        critic_num_blocks=cfg.algo.algo_params.critic_num_blocks,
        num_atoms=cfg.algo.num_atoms,
        critic_min_v=cfg.algo.algo_params.critic_min_v,
        critic_max_v=cfg.algo.algo_params.critic_max_v,
        temp_initial_value=cfg.algo.algo_params.temp_initial_value,
        temp_target_sigma=cfg.algo.algo_params.temp_target_sigma,
        temp_target_entropy=cfg.algo.algo_params.temp_target_entropy,
        actor_bc_alpha=cfg.algo.algo_params.actor_bc_alpha,
        actor_noise_zeta_mu=cfg.algo.algo_params.actor_noise_zeta_mu,
        actor_noise_zeta_max=cfg.algo.algo_params.actor_noise_zeta_max,
        learning_rate_init=cfg.algo.algo_params.learning_rate_init,
        learning_rate_peak=cfg.algo.algo_params.learning_rate_peak,
        learning_rate_end=cfg.algo.algo_params.learning_rate_end,
        learning_rate_warmup_steps=cfg.algo.algo_params.learning_rate_warmup_steps,
        learning_rate_decay_steps=cfg.algo.algo_params.learning_rate_decay_steps,
        normalize_reward=cfg.algo.algo_params.normalize_reward,
        normalized_g_max=cfg.algo.algo_params.normalized_g_max,
        n_step=cfg.algo.algo_params.n_step,
        obs_normalization=cfg.algo.obs_normalization,
        use_amp=cfg.training.use_amp,
        amp_dtype=cfg.algo.algo_params.amp_dtype,
        use_compile=cfg.algo.algo_params.use_compile,
    )
    _maybe_warm_start_flashsac_actor(cfg, learner)

    return DoubleBufferOffPolicyRunner(
        learner=learner,
        env_name=cfg.training.task_name,
        algo_type="flashsac",
        num_envs=cfg.algo.num_envs,
        replay_buffer_n=cfg.algo.replay_buffer_n,
        batch_size=cfg.algo.batch_size,
        learning_starts=cfg.algo.learning_starts,
        updates_per_step=cfg.algo.updates_per_step,
        policy_frequency=cfg.algo.policy_frequency,
        sync_collection=not cfg.training.no_sync_collection,
        env_steps_per_sync=cfg.training.env_steps_per_sync,
        device=device,
        actor_hidden_dim=cfg.algo.actor_hidden_dim,
        use_layer_norm=False,
        obs_normalization=cfg.algo.obs_normalization,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
        actor_kwargs={
            "actor_num_blocks": cfg.algo.algo_params.actor_num_blocks,
            "actor_noise_zeta_mu": cfg.algo.algo_params.actor_noise_zeta_mu,
            "actor_noise_zeta_max": cfg.algo.algo_params.actor_noise_zeta_max,
        },
        seed=cfg.algo.seed,
        trace_enabled=cfg.training.trace_enabled,
        trace_output_dir=cfg.training.trace_output_dir,
        trace_thread_time=cfg.training.trace_thread_time,
        trace_cuda_events=cfg.training.trace_cuda_events,
        replay_prefetch_mode=replay_prefetch_mode,
        verbose_metrics=verbose_metrics,
    )


def _maybe_warm_start_flashsac_actor(cfg: DictConfig, learner: FlashSACLearner) -> None:
    enabled = bool(OmegaConf.select(cfg, "algo.warm_start.enabled", default=False))
    if not enabled:
        return

    source_task = str(OmegaConf.select(cfg, "algo.warm_start.source_task", default="K1WalkFlat"))
    source_algo_log_name = str(
        OmegaConf.select(cfg, "algo.warm_start.source_algo_log_name", default="flash_sac")
    )
    source_load_run = str(OmegaConf.select(cfg, "algo.warm_start.source_load_run", default="-1"))
    source_checkpoint = OmegaConf.select(cfg, "algo.warm_start.source_checkpoint", default=None)
    source_checkpoint_text = (
        None
        if source_checkpoint in (None, "", -1, "-1", "null", "None")
        else str(source_checkpoint)
    )
    strict = bool(OmegaConf.select(cfg, "algo.warm_start.strict", default=False))
    root_dir = Path(__file__).resolve().parents[5]
    ckpt_path, _ = resolve_task_checkpoint_path(
        root_dir,
        task_name=source_task,
        load_run=source_load_run,
        algo_log_name=source_algo_log_name,
        checkpoint=source_checkpoint_text,
        log_root=OmegaConf.select(cfg, "training.log_root"),
    )
    if ckpt_path is None:
        print(
            "[flashsac] warm_start enabled but checkpoint not found: "
            f"task={source_task}, run={source_load_run}, algo_log_name={source_algo_log_name}"
        )
        return

    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        print(f"[flashsac] warm_start skipped: checkpoint payload is not dict ({ckpt_path})")
        return
    actor_state = payload.get("actor")
    if not isinstance(actor_state, dict):
        print(f"[flashsac] warm_start skipped: missing actor state in checkpoint ({ckpt_path})")
        return

    target_state = learner.actor.state_dict()
    compatible_state: dict[str, Any] = {}
    skipped_shape: list[str] = []
    skipped_missing: list[str] = []
    for key, tensor in actor_state.items():
        if key not in target_state:
            skipped_missing.append(key)
            continue
        target_tensor = target_state[key]
        if (
            isinstance(tensor, torch.Tensor)
            and isinstance(target_tensor, torch.Tensor)
            and tensor.shape != target_tensor.shape
        ):
            skipped_shape.append(key)
            continue
        compatible_state[key] = tensor

    if strict and (skipped_shape or skipped_missing):
        raise RuntimeError(
            "[flashsac] warm_start strict mode failed due to incompatible actor keys: "
            f"shape_mismatch={len(skipped_shape)}, missing_in_target={len(skipped_missing)}"
        )

    load_result = learner.actor.load_state_dict(compatible_state, strict=False)
    if skipped_shape:
        preview = ", ".join(skipped_shape[:5])
        suffix = "..." if len(skipped_shape) > 5 else ""
        print(
            "[flashsac] warm_start skipped shape-mismatch keys: "
            f"{len(skipped_shape)} ({preview}{suffix})"
        )
    if skipped_missing:
        preview = ", ".join(skipped_missing[:5])
        suffix = "..." if len(skipped_missing) > 5 else ""
        print(
            "[flashsac] warm_start skipped unknown checkpoint keys: "
            f"{len(skipped_missing)} ({preview}{suffix})"
        )
    print(
        "[flashsac] warm_start actor loaded from "
        f"{ckpt_path} (strict={strict}, loaded={len(compatible_state)}, "
        f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
    )
