# MaxEntDP on ManiSkill: diffusion SAC and two-goal PushT

This branch ports **MaxEntDP** ("Maximum Entropy RL with Diffusion Policy", [diffusionyes/MaxEntDP](https://github.com/diffusionyes/MaxEntDP), JAX) to PyTorch on top of the FastSAC scaffolding in `reinforcement_learning/fastsac/`. It then uses it to ask:

**Can an expressive (diffusion) policy pretrained with RL represent several valid behaviours, where a Gaussian policy cannot? And does seeding it with diverse expert data help?**

The testbed is a new sparse-reward task, `PushT2Goal-v1`, with two equally valid goals in opposite directions.

All numbers below come from runs made on 2026-10-18 (local RTX 4090 plus Hyak ckpt GPUs), wandb project `profjat-university-of-washington/maniskill_pusht`.

## TL;DR

| policy (final checkpoint, 1024 eval episodes, sampled actions) | goal A | goal B |
|---|---|---|
| FastSAC | 99% | 0% |
| FastSAC + 20% expert data (2 seeds) | 1-1.5% | 94-98% |
| MaxEntDP + 20% expert data (4 runs) | 29-47% | 46-66% |
| MaxEntDP, no expert data (5 runs) | 26-60% | 37-67% |

- **Gaussian FastSAC always commits to one goal.** Sampling from it doesn't help, and neither does a 50/50 expert mix: it just collapses onto the other goal.
- **Every MaxEntDP run (9/9 at 100M steps) keeps both goals.**
- **The 50/50 expert seed** makes MaxEntDP start learning ~8M steps earlier. It doesn't consistently change the final A/B balance.
- **On dense PushT,** MaxEntDP matches FastSAC per sample (0.84 vs 0.86 success at 15M steps) but is ~3x slower in wall-clock. The fastest recipe reaches 0.92 at 30M steps in 27 minutes.

## What's in the branch

| path | what |
|---|---|
| `reinforcement_learning/maxentdp/networks/diffusion_actor.py` | `DiffusionActor`: eps-network, cosine VP schedule (as in jaxrl5), DDPM sampler, denoising-error `log_prob` estimator, `qne_target` (Q-weighted noise estimation) |
| `reinforcement_learning/maxentdp/maxentdp.py` | trainer: FastSAC env/replay/eval/checkpoint plumbing + distributional critics + diffusion actor |
| `reinforcement_learning/maxentdp/tests/toy_bimodal.py` | CPU sanity check: QNE on a fixed two-peaked Q gives a 49/51 mode split |
| `mani_skill/envs/tasks/tabletop/push_t_2goal.py` | `PushT2Goal-v1` (registered in `tabletop/__init__.py`) |
| `reinforcement_learning/scripts/collect_pusht2goal_resets.py` | builds the four reset datasets + default-reset sanity check |
| `reinforcement_learning/scripts/merge_replay_buffers.py` | concatenates `--collect-replay-buffer` exports (the 50/50 expert buffer) |
| `reinforcement_learning/scripts/viz_2goal.py` | rolls checkpoints out on default resets: T-centre trajectory plot + final-y histogram + A/B/neither fractions (`.png/.npz/.json`) |
| `reinforcement_learning/scripts/plot_2goal_grid.py` | re-plots a `viz_2goal` `.npz` as a trajectory grid (e.g. 2x2 expert / no expert x MaxEntDP / FastSAC) |
| `reinforcement_learning/scripts/video_2goal.py` | side-by-side video of N sampled episodes per checkpoint |
| `reinforcement_learning/scripts/plot_success_curves.py` | eval success vs env steps from tensorboard event dirs |
| `reinforcement_learning/fastsac/fastsac.py` | + `track_success_keys`: logs any bool `info["success_*"]` as `eval*/success_*_once` (per-goal success) |

## The algorithm, and what changed from the paper

MaxEntDP is SAC with a diffusion actor. The actor is trained by **Q-weighted noise estimation (QNE)**:
1. Noise a replay action to a random diffusion time.
2. Draw K candidate clean actions from the truncated-normal posterior.
3. Weight the candidates by softmax(Q / temperature).
4. Regress the eps-network onto the weighted average noise.

The critic target subtracts α·log π(a'|s'), with log π estimated from the denoising error.

Differences from the reference implementation:

| | paper (jaxrl5) | here | why |
|---|---|---|---|
| critic | twin scalar MSE | FastSAC's distributional C51 critics (v_min/v_max) | reuse the tuned FastSAC critic |
| temperature | fixed α in softmax(Q/α), 0.01-0.2 | `--alpha 1e-3` for the entropy backup + **`--qne-beta`**: softmax((Q − mean_K Q) / (β·std_K Q)) per state | on PushT the Q gap between candidates is ~0.01, so α=0.1 gives uniform weights (random policy); on the sparse task Q is flat in most states, so no fixed α works. β is scale-free |
| K candidates | 500 | 64 | cost; K=256 didn't help |
| log π estimate | 50 samples x 20 times | 8x10 (`--logp-samples/--logp-steps`), 4x5 in the fast recipe | cost; the estimate only enters the backup |
| actor update | every step, batch 256, UTD 1 | every critic step (`--policy-frequency 1`) on 4096-8192 states of the 8192 critic batch | actor update volume was the biggest lever (see below) |
| eval | best of 10 by Q ("select") | `--eval-mode sample` | select picks argmax-by-Q and hides diversity |
| speed | | eps-net first layer split so obs/time terms are computed once; TF32; optional single-critic candidate scoring (`--actor-q target1`) | |

## How the recipe was found (dense PushT-v1, 20-30M steps)

| change | eval success | notes |
|---|---|---|
| port defaults (α=0.1, actor every 4th step on 2048 states) | 0 at 8M | QNE weights uniform (effective sample size 63/64) |
| α = 1e-3 | 0.30 @ 20M | α 1e-4…3e-3 all 0.22-0.41: temperature no longer the bottleneck |
| actor every 2nd step, 4096 states | 0.83 @ 20M | |
| actor every step, all 8192 states | 0.95 @ 20M | |
| + `--qne-beta 0.3` (**quality recipe**) | **0.98 @ 30M, 68 min** | 7.3k env-steps/s |
| + `--actor-batch-size 4096 --diffusion-steps 10 --logp-samples 4 --logp-steps 5 --actor-q target1` (**fast recipe**) | **0.92 @ 30M, 27 min** | 18.7k env-steps/s |
| same with `--actor-batch-size 8192` (balanced) | 0.95 @ 30M, 39 min | 12.9k env-steps/s |
| FastSAC reference | 0.98 @ 19M, ~9 min | 37k env-steps/s |

Ablations that didn't help:
- K=256 candidates;
- fresh policy samples instead of replay actions for QNE (`--qne-policy-frac`);
- β=1 (too soft: 0.03 @ 20M);
- `torch.compile` of the sampler;
- bf16 candidate scoring (`--qne-bf16`, slower);
- K=32 (faster but plateaus at 0.86).

The remaining cost is inherent: 10-20 denoising passes per action, plus scoring K candidates per actor update.

## The two-goal task: `PushT2Goal-v1`

- **Default start:** the T starts near (−0.156, 0) ± 2 cm with its stem pointing away from the robot (yaw −90° ± 10°). That makes the scene mirror-symmetric about the x axis. The stick tip starts straight behind the T at (−0.30, 0), via a symmetric IK pose.
- **Goals:** A at y = +0.15 and B at y = −0.15, both at the start yaw.
- **Reward:** sparse, +1 per step while the T covers ≥ 90% of either goal. Episodes are 100 steps and don't terminate on success.
- **Observations:** no goal information, so the optimal policy is genuinely bimodal.
- **Per-goal variants:** `valid_goals a|b` rewards only one goal (observations unchanged); used to train the per-goal experts.
- **Logging:** `info["success_a"]` / `info["success_b"]` are logged as `eval*/success_a_once` / `success_b_once`.

**Training resets** (following the sparse PushTGoal FastSAC recipe) are a uniform mix of four 4096-state datasets:

| dataset | T pose | stick tip |
|---|---|---|
| `near_goal_a_succ25` | within 4 cm / 30° of goal A; 25% start solved (≥ 91% overlap) | 1-5 cm from the T |
| `near_goal_b_succ25` | same around goal B | 1-5 cm from the T |
| `between_arm_near` | x = start ± 5 cm, |y| ≤ 18 cm, yaw = start ± 45°, never ≥ 89% on either goal | 1-5 cm from the T |
| `between_anywhere` | same T distribution | anywhere in the workspace, ≥ 1 cm from the T |

Each candidate state is rejected if:
- IK misses by more than 5 mm;
- the T moves more than 2 mm / 1° while the arm holds still for 10 steps;
- the tip drifts.

All evals, plots and numbers in this README use the default start.

## Running the experiments

Everything runs from the repo root with the `maniskill` conda env (`torch 2.5.1+cu121`). Paths are relative, so `runs/` is created there.

> Only one ManiSkill GPU-PhysX process per GPU: a second process starting on the same GPU crashes the running one (CUDA 700).

```bash
cd <repo> && export PYTHONPATH=$PWD
PY="python -u"   # e.g. ~/miniconda3/envs/maniskill/bin/python -u
```

### 0. Sanity check (CPU, ~1 min)

```bash
$PY reinforcement_learning/maxentdp/tests/toy_bimodal.py
# expect: frac right mode ~0.5, mean|x| ~0.59
```

### 1. MaxEntDP on dense PushT-v1

```bash
COMMON="--env.id PushT-v1 --v-min 0 --v-max 100 --total-timesteps 30000000 --eval-freq 1000000 \
  --num-eval-envs 64 --alpha 0.001 --policy-frequency 1 --qne-beta 0.3 --track --wandb-group maxentdp_pusht"
# quality recipe (0.98 @ 30M, ~68 min on a 4090)
$PY -m reinforcement_learning.maxentdp.maxentdp --exp-name mdp_pusht_quality $COMMON --actor-batch-size 8192
# fast recipe (0.92 @ 30M, ~27 min)
$PY -m reinforcement_learning.maxentdp.maxentdp --exp-name mdp_pusht_fast $COMMON --actor-batch-size 4096 \
  --diffusion-steps 10 --logp-samples 4 --logp-steps 5 --actor-q target1
# FastSAC reference
$PY -m reinforcement_learning.fastsac.fastsac --exp-name fastsac_pusht_ref --env.id PushT-v1 --v-min 0 --v-max 100 \
  --total-timesteps 20000000 --eval-freq 1000000 --num-eval-envs 64 --num-envs 4096 --track
```

### 2. Two-goal reset datasets (~5 min)

```bash
$PY reinforcement_learning/scripts/collect_pusht2goal_resets.py
# -> runs/reset_datasets_2goal/PushT2Goal-v1_{near_goal_a_succ25,near_goal_b_succ25,between_arm_near,between_anywhere}*/reset_states.pt
#    + default_resets.png, per-dataset grid.png, and a printed default-reset check (overlap A ~0.11 / B ~0.11, success 0)
D=runs/reset_datasets_2goal
NA=$D/PushT2Goal-v1_near_goal_a_succ25_4096_seed0/reset_states.pt
NB=$D/PushT2Goal-v1_near_goal_b_succ25_4096_seed0/reset_states.pt
BN=$D/PushT2Goal-v1_between_arm_near_4096_seed0/reset_states.pt
BA=$D/PushT2Goal-v1_between_anywhere_4096_seed0/reset_states.pt
G2="--env.id PushT2Goal-v1 --env.reward-mode sparse --v-min 0 --v-max 100 --num-eval-envs 192 \
  --num-eval-default-envs 128 --eval-freq 2000000 --save-freq 5000000 --track"
```

`--num-eval-default-envs` adds the default-start eval (`eval/*`) next to the reset-mix eval (`eval_datasets/*`).

### 3. FastSAC on the two-goal task (step 4 of the study; 120M steps, ~2.5 h on a ckpt GPU)

```bash
$PY -m reinforcement_learning.fastsac.fastsac --exp-name fastsac_2goal_ab $G2 --num-envs 4096 \
  --total-timesteps 120000000 --env.kwargs reset_datasets $NA,$NB,$BN,$BA
```

### 4. Per-goal experts and the 50/50 expert buffer (step 5)

```bash
# experts: same task, but only one goal rewarded; reset mix = own near-goal set + the two between sets
$PY -m reinforcement_learning.fastsac.fastsac --exp-name fastsac_2goal_onlyA $G2 --num-envs 4096 \
  --total-timesteps 30000000 --env.kwargs reset_datasets $NA,$BN,$BA valid_goals a
$PY -m reinforcement_learning.fastsac.fastsac --exp-name fastsac_2goal_onlyB $G2 --num-envs 4096 \
  --total-timesteps 30000000 --env.kwargs reset_datasets $NB,$BN,$BA valid_goals b
# both reached 100% on their own goal by ~20M. Collect 1024 deterministic episodes each on DEFAULT resets with the
# two-goal reward (no reset_datasets / valid_goals -> --no-config-from-checkpoint).
for g in A B; do
  $PY -m reinforcement_learning.fastsac.fastsac --exp-name expert_2goal_only$g \
    --checkpoint runs/fastsac_2goal_only$g/ckpt_20000768.pt --collect-replay-buffer --no-config-from-checkpoint \
    --env.id PushT2Goal-v1 --env.reward-mode sparse --v-min 0 --v-max 100 \
    --num-envs 16 --num-eval-envs 1024 --buffer-size 102400 --env.num-eval-steps 100
done
$PY reinforcement_learning/scripts/merge_replay_buffers.py \
  runs/expert_2goal_onlyA/replay_buffer.pt runs/expert_2goal_onlyB/replay_buffer.pt --out runs/expert_2goal_ab50/replay_buffer.pt
# ours: 2048 episodes, success 0.995; A half ends at y +0.149, B half at y -0.150
EXPERT="--expert-buffer runs/expert_2goal_ab50/replay_buffer.pt --expert-ratio 0.2"
```

### 5. MaxEntDP on the two-goal task, with and without the expert seed (steps 3 and 5; 100M steps)

```bash
R3="--alpha 0.001 --eval-mode sample --policy-frequency 1 --actor-batch-size 8192 --auto-resume"
DS="--env.kwargs reset_datasets $NA,$NB,$BN,$BA"
for b in 0.3 0.5; do
  $PY -m reinforcement_learning.maxentdp.maxentdp --exp-name mdp_2goal_beta$b        $G2 $R3 --qne-beta $b --total-timesteps 100000000 $DS
  $PY -m reinforcement_learning.maxentdp.maxentdp --exp-name mdp_2goal_beta${b}_exp  $G2 $R3 --qne-beta $b --total-timesteps 100000000 $DS $EXPERT
done
# fast recipe (100M in ~2 h on a ckpt GPU instead of ~5 h):
FAST="--alpha 0.001 --eval-mode sample --policy-frequency 1 --actor-batch-size 4096 --diffusion-steps 10 \
  --logp-samples 4 --logp-steps 5 --actor-q target1 --qne-beta 0.5 --auto-resume"
$PY -m reinforcement_learning.maxentdp.maxentdp --exp-name mdp_2goal_fast        $G2 $FAST --total-timesteps 100000000 $DS
$PY -m reinforcement_learning.maxentdp.maxentdp --exp-name mdp_2goal_fast_exp    $G2 $FAST --total-timesteps 100000000 $DS $EXPERT
# control: FastSAC with the same expert mix
$PY -m reinforcement_learning.fastsac.fastsac --exp-name fastsac_2goal_ab_exp $G2 --num-envs 4096 \
  --total-timesteps 100000000 $DS $EXPERT
```

- **Resuming:** `--auto-resume` restarts from the newest readable `runs/<exp-name>/ckpt_*.pt`. The replay buffer isn't saved, so it refills. We used this to continue preemptible or time-limited jobs.
- **Seeds:** add `--seed 2` (with a different `--exp-name`) for replicates.

### 6. Plots, numbers and video

```bash
# trajectories + histogram + A/B fractions on 1024 default-reset episodes, sampled actions for every policy
$PY reinforcement_learning/scripts/viz_2goal.py \
  --ckpts runs/mdp_2goal_beta0.5_exp/final_ckpt.pt runs/mdp_2goal_beta0.5/final_ckpt.pt \
          runs/fastsac_2goal_ab_exp/final_ckpt.pt runs/fastsac_2goal_ab/final_ckpt.pt \
  --labels "MaxEntDP + expert data" "MaxEntDP" "FastSAC + expert data" "FastSAC" \
  --mode sample sample sample sample --num-episodes 1024 --out runs/viz_2goal/final
# 2x2 trajectory grid from the same rollouts (expert left / none right, MaxEntDP top / FastSAC bottom)
$PY reinforcement_learning/scripts/plot_2goal_grid.py --src runs/viz_2goal/final --cells 0 1 2 3 \
  --titles "MaxEntDP + expert data" "MaxEntDP" "FastSAC + expert data" "FastSAC" --out runs/viz_2goal/grid_2x2
# success-rate curves from tensorboard
$PY reinforcement_learning/scripts/plot_success_curves.py \
  --dirs runs/fastsac_2goal_ab runs/mdp_2goal_beta0.5 runs/mdp_2goal_beta0.5_exp \
  --labels "FastSAC" "MaxEntDP" "MaxEntDP + expert data" --out runs/viz_2goal/success_curves
# side-by-side video, 10 sampled episodes each, same start states
$PY reinforcement_learning/scripts/video_2goal.py \
  --ckpts runs/mdp_2goal_beta0.5/final_ckpt.pt runs/fastsac_2goal_ab/final_ckpt.pt \
  --labels MaxEntDP FastSAC --modes sample sample --num-episodes 10 --seed 3 --out runs/viz_2goal/side_by_side.mp4
```

`viz_2goal.py` modes:
- FastSAC: `mean` (deterministic) or `sample`;
- MaxEntDP: `sample`, `select` (best of 10 by Q), or `det` (no per-step noise).

Use `sample` for mode statistics.

## Results in detail

**Dense PushT (step 1):** see the recipe table above. Per sample, MaxEntDP now matches FastSAC early on (0.84 vs 0.86 at 15M steps).

**Two goals, final checkpoints** (1024 default-reset episodes each, sampled actions; A / B / neither, in %):

| run | steps | A | B | neither |
|---|---|---|---|---|
| FastSAC | 120M | 98.9 | 0.0 | 1.1 |
| FastSAC + expert | 100M | 1.0 | 97.8 | 1.3 |
| FastSAC + expert, seed 2 | 90M | 1.5 | 94.4 | 4.1 |
| MaxEntDP β=0.3 + expert | 100M | 41.4 | 53.8 | 4.8 |
| MaxEntDP β=0.3 + expert, seed 2 | 100M | 47.4 | 45.9 | 6.7 |
| MaxEntDP β=0.5 + expert | 100M | 43.4 | 52.6 | 4.0 |
| MaxEntDP fast β=0.5 + expert | 100M | 29.2 | 66.0 | 4.8 |
| MaxEntDP β=0.3 | 100M | 40.2 | 58.2 | 1.6 |
| MaxEntDP β=0.3, seed 2 | 100M | 59.7 | 36.8 | 3.5 |
| MaxEntDP β=0.5 | 100M | 34.6 | 60.7 | 4.7 |
| MaxEntDP β=0.5, seed 2 | 100M | 56.9 | 39.8 | 3.2 |
| MaxEntDP fast β=0.5 | 100M | 26.2 | 67.4 | 6.4 |

**Learning curves:**
- All runs learn the sparse task stably to 0.93-0.99 success, with no collapse or oscillation once learned.
- Expert-seeded MaxEntDP starts ~8M steps earlier (80% success at ~18M vs ~26M).
- The per-goal split wanders during training: one no-expert run was 88/10 at 65M and 60/37 at 100M.
- FastSAC + expert was balanced early (13/35 at 7M) and collapsed by 16M.

**Things that did not work (keep in mind):**
- **Fixed α** on the sparse task: QNE weights stay uniform where Q is flat, and the policy never left the reset baseline.
- **An earlier MaxEntDP recipe** (actor every 2nd step, β=0.3, no expert) collapsed to 0% A / 98% B. Two-goal behaviour without seeding depends on getting enough actor updates.
- **`--eval-mode select`** (the paper default) inflates success and hides the second mode. All tables above use `sample`.

**Caveats:**
- The default start is only approximately symmetric: the arm IK pose and start noise aren't perfectly mirrored.
- Splits vary run to run by ±20 points. Read the results as "both goals vs one goal", not as precise ratios.
- MaxEntDP is still ~3x slower than FastSAC in wall-clock.
