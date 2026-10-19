# FastSAC (ManiSkill) review notes — 2026-09-24

Status legend: FIXED / OPEN / N/A (harmless at current settings)

## Critical
1. FIXED — `ReplayBuffer.add()` never stored `next_obs` (`utils/buffers.py`); every target bootstrapped from zeros and the normalizer was fed zero vectors. Any run before 2026-09-23 23:55 (e.g. wandb `428nbsyl`) is invalid.
2. FIXED (hardcoded) — distributional critic support was ±20; PushT at gamma 0.99 needs ~[0, 100]. Now `v_min=0, v_max=100` in `fastsac.py`; should become CLI args (support must track reward scale / gamma per task).

## Crashes / broken features
3. FIXED — eval called nonexistent `actor.get_eval_action`; now `actor.get_action(obs)[2]` (deterministic mean).
4. FIXED — eval fed raw obs; now `normalize_obs(eval_obs, update=False)`.
5. OPEN — `--evaluate` never loads `args.checkpoint` (no `torch.load`); evaluates a random actor.
6. OPEN — checkpoints omit the obs normalizer state (reloaded actor sees wrong input scale), save *target* critics under `qf1`/`qf2`, no optimizers; `log_alpha` undefined with `--no-autotune`.
7. OPEN — `normed_obs` undefined with `--no-obs-normalization` (NameError).

## Correctness / robustness
8. OPEN — target critics built without `use_layer_norm=args.use_layer_norm`; `--no-use-layer-norm` breaks `load_state_dict`.
9. OPEN — `use_tanh=False` default: Gaussian actions unbounded; env clips them but the buffer stores unclipped actions, so the critic learns Q for actions never executed. Clip before `rb.add` or default `use_tanh=True`.
10. N/A at n_step=1 — `ReplayBuffer` not given `gamma=args.gamma` (n-step sum uses 0.99); `batch_inds + first_done` not wrapped mod `buffer_size`.
11. OPEN — no clipped double-Q: `target_value_min/max` computed but unused; each critic trains on its own target.
12. OPEN (minor) — `critic_obs_normalizer` unused; `--bootstrap-at-done` ignored; target update keyed on `global_step` not `global_update`; `charts/samples_per_sec` mixes units; `F.softmax` without `dim`.

## Env semantics worth remembering
- `partial_reset=False` (default) → ManiSkillVectorEnv zeroes terminations: buffer never sees success terminations, critic bootstraps through success (Q ceiling ~1/(1-gamma)); train/* metrics arrive once per 100*num_envs global steps.
- Normalizer is updated on every sampled replay batch (obs and next_obs), not on rollouts.

## Re-check 2026-09-24 (fastsac.py @ 09:28): #5–#9, #11, #12 still OPEN; #10 still latent.

## Tuning deltas vs holosoma FastSACConfig (UWLab recipe that works)
| knob | this port | holosoma default |
|---|---|---|
| use_tanh | False | True (per-joint action bounds) |
| gamma | 0.8 default (0.99 used) | 0.97 |
| tau | 0.01 | 0.125 |
| policy_frequency | 1 | 4 |
| updates per env step | training_freq*utd (e.g. 128 @ 4096 envs) | num_updates=8 |
| target_entropy_ratio | 1.0 | 0.0 |
| clipped double-Q | none (min/max computed, unused) | min_q_target option (argmin critic's dist shared) |
| amp | none | bf16 |
| compile | none | option (MR.Q: compile + CudaGraphModule = 2.5x) |
| max_grad_norm | none | 0.0 (off) — same |
| v_min/v_max | hardcoded 0/100 | -20/20 (task-specific) |
Observed in PushCube smoke: alpha -> ~0 immediately (entropy = target -4); success_at_end 0.61 vs success_once 0.95; ~1.4k SPS update-bound.

## Re-review 2026-09-24 (fastsac.py @ 09:57, buffers @ 09:50, networks @ 09:41) — #11 (CDQ) deliberately WONTFIX
FIXED: #6 partly (normalizer saved; final save guards log_alpha), #7, #8 (LayerNorm now unconditional, flag removed), #9 (use_tanh=True default, scale 1/bias 0), #10 (gamma passed; first_done wrapped — verified n_step=3 sums 1/1.9/2.71 @ gamma .9), #12 partly (critic_obs_normalizer removed).
REGRESSION A: v_min/v_max back to critic defaults ±20 (fastsac.py no longer passes 0/100) — PushT @ gamma .99 Q reaches 50-90 → clamped.
NEW B: `control_mode` default "pd_joint_delta_pose" is not a ManiSkill mode → AssertionError unless --control-mode is passed.
OPEN C (#5 remainder): checkpoint load (L257-261) doesn't restore `obs_normalizer` → --evaluate/resume runs the actor on unnormalized-by-fresh-stats obs.
OPEN D: periodic save (L342-348) still unconditional `log_alpha` (NameError w/ --no-autotune) and `obs_normalizer` (NameError w/ --no-obs-normalization; final save too).
OPEN E: critic target (L414, L421) uses `log_alpha.exp()` → NameError w/ --no-autotune; use `alpha`.
Minor: target update keyed on global_step; F.softmax no dim; samples_per_sec units; --bootstrap-at-done unused; Actor init_trick references self.fc_mu.weight (Sequential) → AttributeError if ever enabled; AsymmetricReplayBuffer L572-574 still unwrapped (unused).

## Re-review 2026-09-24 (fastsac.py @ 10:07, buffers @ 10:08)
FIXED: control_mode default -> pd_ee_delta_pose; --no-autotune (entropy_coeff = alpha); conditional saves; normalizer load added; target update on global_update; softmax dim; Asymmetric buffer next_obs wrapped.
OPEN 1: --v-min/--v-max args exist (L136-139) but are NOT passed to the 4 DistributionalQNetwork(...) calls (L255-258) → always ±20. normalized_dense was already ManiSkill's default (smoke run used it; Q hit 85 at gamma .99). Normalized reward <=1/step → Q ceiling ~1/(1-gamma): 5 @0.8, 20 @0.95, 33 @0.97, 100 @0.99. ±20 OK only for gamma <= ~0.95.
OPEN 2: saves store the normalizer MODULE (`obs_normalizer`), load calls load_state_dict on it → TypeError (verified). Save `obs_normalizer.state_dict()`.
Minor: AsymmetricReplayBuffer L574 terminations index still unwrapped (unused class); init_trick left as-is (user may delete).

## 2026-09-25 — CLI split into run / env / algo
Shared `RunArgs` + `EnvArgs` (+ `gym_make_kwargs`, `restore_from_saved`, `config_dict`) live in `reinforcement_learning/utils/args.py`; fastsac.py keeps only `AlgoArgs`. Env flags now `--env.*` (`--env.id`, `--env.num-envs`, `--env.control-mode`, `--env.partial-reset`, `--env.num-eval-steps` [None = max_episode_steps], task kwargs via `--env.kwargs K V ...`); run/algo flags unchanged. Checkpoints carry `config` {run, env, algo}; loading one restores env task fields (id/obs/control/reward mode/kwargs) + algo model fields (v_min/v_max/use_tanh/obs_normalization) unless overridden on the CLI or `--no-config-from-checkpoint`. `num_steps` removed (train videos use max_episode_steps). Default env id PickCube-v4 (unregistered) -> PickCube-v1.
PushT COM kwargs are now tee_com_offset_x / tee_com_offset_y (floats) + tee_ballast_mass_ratio, e.g. --env.kwargs tee_com_offset_x 0.02 tee_ballast_mass_ratio 0.3. Latent: grad_steps_per_iteration = int(training_freq*utd) can round to 0 (e.g. training_freq 64 @ utd 0.002) -> no updates -> NameError qf1_target_values at first log.

## 2026-09-27 — replay buffer collection/export
`--collect-replay-buffer --checkpoint X`: one deterministic eval pass (num_eval_envs x num_eval_steps) into a ReplayBuffer (n_envs=num_eval_envs; errors if --buffer-size < num_eval_steps*num_eval_envs), saved to runs/<exp>/replay_buffer.pt as {'buffer': rb.export(), 'metadata': {config, checkpoint, policy, num_transitions, num_episodes, success_once, return}}. export() = raw time-major (T, n_envs, ...) filled rows + n_envs/n_steps/gamma/pos/full; ReplayBuffer.load() is the inverse (same n_envs, >= rows). Verified round trip: raw obs (next_obs[t]==obs[t+1] exact), truncations only at t=99, reload identical, n-step sampling works. Still builds --env.num-envs training envs in collect mode (line ~150 only shrinks for --evaluate).

## 2026-09-28 — expert sampling ratio
`--expert-buffer <replay_buffer.pt> --expert-ratio r`: expert data loaded into its own ReplayBuffer via `ReplayBuffer.from_export(data, device, n_steps=<run's>, gamma=<run's>)` (n-step targets computed at sample time with the run's settings); `sample_mixed(rb, expert_rb, B, r)` = exactly round(r*B) expert rows per batch (per-sample env draw, so any k works). Load prints metadata; raises on env mismatch (id/obs/control/reward mode + parsed kwargs) or obs/act dims; warns if expert eval_partial_reset != training partial_reset. Training mode only. Logs charts/expert_ratio. Expert obs go through the same normalizer update as online. Verified: exact counts (0, 3/8192, 25/100, 2/7, 2048, 4096, 8192), from_export n_step override, short PushT train with 0.25 (qf finite), mismatch ValueError, partial-reset warning.
Unrelated: utils/buffers.py:22 `from telnetlib import theNULL` is unused (DeprecationWarning on 3.11, ImportError on 3.13).

## 2026-09-28 — --reset-optimizers
With --resume, skips restoring q/actor/alpha optimizers (fresh Adam, CLI lrs); weights/alpha/normalizer/step still load. Without it, restored optimizers also restore the checkpoint's LEARNING RATES (verified: --q-lr 3e-5 on a plain resume stayed 3e-4). Plain --checkpoint without --resume never restores optimizers.
2026-09-28: expert env mismatch is now informational (prints the diff), per user — mixing a source-domain expert into a gap finetune is the point. obs/action dim mismatch still raises.
2026-09-28: --save-freq (RunArgs) decoupled from --eval-freq: None = same as eval_freq (old behavior), 0 = final ckpt only; saving moved out of the eval block into its own step-boundary check. Verified default/20480/5120/0 vs eval 10240, and --evaluate saves nothing.
2026-09-28: wandb x-axis counters — per_env_step (+1 per vector-env step, independent of num_envs) and train_episodes (finished train episodes across all envs) ride along on every wandb.log row; wandb.define_metric('*', step_metric='per_env_step') makes per_env_step the default x-axis. Both restart at 0 every run (user choice; not in ckpts) — a --wandb-run-id continuation overlaps its earlier segment on this axis. wandb's own step is still total env steps. Also in TensorBoard as charts/per_env_step, charts/train_episodes.
2026-09-29: --resume now continues per_env_step/train_episodes (saved in ckpts; derived for older ckpts from global_step / saved num_envs, episodes exact when partial_reset=False).
2026-10-01: narrow resets — `--env.init-state-path <init_states.pt> --env.init-state-index K` makes every TRAINING-env reset (incl. ManiSkillVectorEnv auto/partial resets) restore saved state K via FixedInitStateWrapper (utils/init_states.py; injects ManiSkill's reset_to_env_states option, so controller reset + obs/info come from the restored state). Eval envs keep random resets. States come from reinforcement_learning/scripts/save_init_states.py (full sim state + tee xy/yaw + robot qpos; restorable into ballast envs since mass isn't in the state). Not an ENV_TASK_FIELD, so checkpoints never carry it. Verified exact (0.0) on reset / auto-reset / partial reset, untouched non-reset envs, obs == fresh get_obs, eval still random, e2e fastsac 1-env run.
