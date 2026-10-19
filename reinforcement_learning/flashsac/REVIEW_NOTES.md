# FlashSAC (ManiSkill) review notes — 2026-10-05

Reference: ~/research/FlashSAC @ 87edc90 (`flash_rl/agents/flashSAC/{agent,update,network,layer}.py`, `configs/agent/flashSAC.yaml`).
Status legend: FIXED / OPEN / N/A. Static review; the script cannot currently run, so nothing was executed.

## Crashes (hit in order on a plain training run)
1. OPEN — flashsac.py:25-26 import `reinforcement_learning.fastsac.networks.flashsac_{actor,critic}`; the files live in `reinforcement_learning/flashsac/networks/` → ModuleNotFoundError.
2. OPEN — flashsac.py:191 `args.cuda` → AttributeError (`Args` only has run/env/algo); use `args.run.cuda`.
3. OPEN — flashsac_actor.py:26 `env.single_observation_space["policy"]` and flashsac_critic.py:25 `["critic"]`: ManiSkill state obs is a `Box` (no asymmetric obs here) → use `.shape` like fastsac_actor/critic.
4. OPEN — flashsac.py:287 `args.algo.target_entropy_ratio` not defined in AlgoArgs → AttributeError (and see #11: the reference doesn't use this formula).
5. OPEN — flashsac.py:411 rollout passes undefined `policy_obs`; should be `normed_obs` (computed at :406-408 and unused).
6. OPEN — flashsac_critic.py:76 `projection` uses an undefined global `device` → NameError on the first update (fastsac_critic uses `self.device`). Also `F` is never imported there (only reached if `next_dist is None`, which the port never does).
7. OPEN — flashsac.py:339 eval calls `actor.get_action(normed_eval_obs)` without the required `training` arg → TypeError at the first eval; use `training=False`.
8. OPEN — flashsac.py:571 logs `expert_rb`, never defined (no expert-buffer support was ported) → NameError at the first log. Either port `--expert-buffer/--expert-ratio` + `sample_mixed` from fastsac or drop the line (and the unused `sample_mixed` import).
9. OPEN — flashsac.py:478/487 critic target uses `log_alpha.exp()` → NameError with `--no-autotune` (same as fastsac's old #E); use `alpha`.
10. OPEN — flashsac.py:169 `ALGO_MODEL_FIELDS` undefined → NameError whenever `--checkpoint` has a saved config.

## Broken features
- OPEN — `--checkpoint/--resume` never load anything: no actor/critic/normalizer/optimizer/log_alpha `load_state_dict`, `start_step`/counters never restored. Worse, `--resume` skips the random warm-up (:402), so a "resumed" run trains a fresh random actor with no learning_starts. `--evaluate` likewise evaluates a random actor. Port the fastsac load block (and add `log_alpha`, `reset_optimizers`).
- OPEN — `--collect-replay-buffer` flows through but the actor is never loaded (same root cause).
- Note — ckpts save `qf1_target`/`qf2_target` under `qf1`/`qf2`; with BatchNorm the target's running stats come from target forwards (training=True), so this is loadable but not the online critic.

## Algorithm vs reference
Matches: CrossQ-style concat of (s,a) and (s',a') through the critic (shared BN stats); target critic forwarded with training=True (EMA covers params only, BN stats come from these forwards — same as reference); argmin-Q selection of the target distribution (clipped double-Q on the categorical target); entropy folded into the bin shift (r + γ(z − α logπ)); actor loss min(Q1,Q2) with the critic in eval-BN mode; weight/BN/RMSNorm re-normalization after each step (actor only after actor steps); block structure / residual / expansion 4; zeta noise-repeat exploration with a single shared counter; per-sample γ^effective_n (better than the reference's fixed γ^n).

11. OPEN — target entropy: reference = 0.5·A·log(2πe·σ²) with `temp_target_sigma=0.15` (≈ −0.48·A); port uses −A·ratio and leaves `temp_target_sigma` unused. `temp_inital_value` (sic) also unused; alpha starts at `alpha=0.001` vs reference 0.01.
12. OPEN — `use_tanh=False` default: unbounded Gaussian actions, env clips, buffer stores unclipped actions (fastsac #9). Reference is always tanh-squashed. Default True.
13. OPEN — no reward normalization. Reference scales rewards by the running discounted-return std (floor G_r_max/G_max) and fixes the support at ±normalized_G_max = ±5. Port has raw rewards on ±20 by default: PushT normalized_dense at γ 0.97 has Q up to ~33, at 0.99 ~100 → target clamped. Either port RewardNormalizer or set v_min/v_max per task (0/100 like the fastsac recipe).
14. OPEN — log-std range [-5, 0] vs reference [-10, 2] (max std 1 vs 7.4 pre-tanh).
15. OPEN — double obs normalization: external EmpiricalNormalization (updated on every replay batch, obs and next_obs) in front of the network's input BatchNorm. Reference has none — the input UnitBatchNorm is the normalizer. Recommend default `obs_normalization=False`.
16. OPEN — optimizers: port AdamW(wd 1e-3, betas 0.9/0.95) for actor/critic and AdamW with *default wd 0.01* on `log_alpha` (pulls log α → 0, i.e. α → 1). Reference: plain Adam, default betas, no weight decay, warmup+cosine LR 3e-4 → 1.5e-4 for actor/critic/temperature. Weight decay is mostly a no-op on re-normalized weights but does hit the head biases.
17. Minor — update order: reference does actor → temperature → critic each step; port does critic → actor → alpha. Reference actor forward concatenates obs+next_obs (BN stats over 2B); port uses obs only. Temperature loss is computed from a fresh training=False forward rather than the actor step's entropy.
18. Minor — default hyperparameters differ from the reference config: batch 8192 (ref 2048), tau 0.125 (ref 0.01), policy_frequency 4 (ref actor_update_period 2), hidden 128/256 and 2 blocks match, 101 bins match. No AMP / torch.compile (reference uses both).
19. Minor — `q_support` is a plain tensor hardcoded to cuda (not a buffer); `max_action` unused; target update keyed on `global_step % target_network_frequency` (fine at 1); `UnitRMSNorm` eps = finfo default instead of 1e-6.

## Re-review 2026-10-05 (flashsac.py @ 22:00, actor @ 22:07, critic @ 22:05) — goal: minimally viable from-scratch run
FIXED: #1-#7, #9 (entropy_coeff = alpha when not autotuning), #11 (target entropy now 0.5·A·log(2πe·σ²) with σ = `target_entropy_ratio`), #12 (use_tanh True), #14 (log-std [-10, 2]), #18 defaults (tau 0.01, batch 2048, policy_frequency 2, alpha 0.01).
WONTFIX (by request): checkpoint/resume/evaluate/collect loading, #10.

OPEN (crash) — flashsac.py:577 `charts/expert_ratio` references undefined `expert_rb` / `args.algo.expert_ratio` → NameError at the FIRST training log. Verified: PushCube smoke run dies at step 2048. Delete the line.
Smoke test with only that line removed (PushCube-v1, 64 envs, training_freq 512, utd 0.0625, learning_starts 2048, 60k steps): runs clean, eval success_once 0 → 0.31 (33k) → 1.00 (49k) → 0.81 (57k); qf values 0.03 → 6.7, qf_loss 4.49 → 3.69 monotone, alpha 0.010 → 0.007-0.008, ~1k SPS.

Latent (not hit by current recipes):
- actor_loss / policy_entropy / alpha_loss are only defined after the first actor update; if grad_steps_per_iteration < policy_frequency and the first log lands on the first update iteration → NameError. grad_steps_per_iteration = int(training_freq·utd) can also round to 0 → qf1_target_values NameError (same as fastsac).

Non-crash, still differs from reference:
- `target_entropy_ratio` now means σ (default 0.15) but keeps the name/docstring "ratio of target entropy"; FastSAC's flag of the same name is a multiple of −A, so the same CLI value means different things across the two scripts.
- #13 no reward normalization; ±20 support. Fine for PushCube (Q ~7), clips PushT normalized_dense at γ 0.97 (Q up to ~33).
- #15 obs_normalization still defaults True (redundant with the input BatchNorm).
- #16 AdamW wd 1e-3 + betas (0.9, 0.95) on actor/critic; `a_optimizer` AdamW with default wd 0.01 on log_alpha; no LR schedule.
- #17 update order / actor BN batch; #19 minor items unchanged. `max_action`, `sample_mixed`, `parse_env_kwargs`, `statistics` unused.
- Initial policy entropy is very negative (−22 at the first log) because the orthogonal-init log-std head spreads log σ over [−10, 2]; same init as the reference, recovers within ~10k steps.

## 2026-10-05 — optimizers → reference (#16 FIXED)
Plain Adam (default betas, no wd) for actor/critic/log_alpha; `--lr-end-ratio` (default 0.5, 1.0 = constant) cosine-decays each lr over total_updates = post-learning_starts iterations × grad_steps_per_iteration via LambdaLR stepped after its optimizer (actor/alpha step only on actor updates → end at ~0.75×, same as reference). Logs charts/q_lr, charts/policy_lr. Scheduler state not checkpointed.
Smoke (same PushCube config, 60k): clean; q_lr 3e-4 → 1.50e-4 exactly at the end, policy_lr → 2.25e-4. Single seed learned slower than the AdamW(0.9, 0.95) run: success_once 0.56 @57k (first >0 at 49k) vs 1.00 @49k; Q 2.7 vs 6.7 at the end. Not seed-controlled.

## 2026-10-05 — rollout grad leak (FIXED)
Rollout `_sample_rollout_action` ran without `torch.no_grad()` → actions carried autograd history; `rb.add` copied them in place into the preallocated buffer (`actions` got a CopySlices grad_fn), so every rollout's actor graph stayed alive (+51 MB/step at 4096 envs → CUDA OOM at ~1M steps on PickCube defaults) and each `qf_loss.backward()` also walked old actor graphs. Probe confirmed linear growth (3.1 → 11.2 GiB over 160 adds). Fix: wrap the rollout call in `torch.no_grad()`. Not visible in the 64-env smoke test (~0.8 MB/step).
