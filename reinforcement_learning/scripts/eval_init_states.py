"""success@k of a FastSAC or Stream-AC/PPO-pretrain checkpoint from each saved initial state.

Per state: num_envs parallel envs x episodes_per_env episodes, all reset to that state (all states run in one batch). success@k = fraction of the
num_envs trials whose first k episodes contain at least one success; success@1 = success rate of env 0's episodes.

    python reinforcement_learning/scripts/eval_init_states.py --checkpoint runs/fastsac_pusht_control/final_ckpt.pt \
        --init-state-path runs/init_states/PushT-v1_10_seed0/init_states.pt --env.id PushT-v1 \
        --env.kwargs tee_ballast_pos_x 0.0 tee_ballast_pos_y 0.1375 tee_ballast_mass_ratio 0.9
"""
import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch
import tyro
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from reinforcement_learning.utils.args import EnvArgs, gym_make_kwargs
from reinforcement_learning.utils.init_states import FixedInitStateWrapper, load_init_state, per_env_init_state
from reinforcement_learning.utils.policies import load_policy


@dataclass
class Args:
    checkpoint: str
    init_state_path: str
    env: EnvArgs = field(default_factory=lambda: EnvArgs(id="PushT-v1"))
    indices: Optional[List[int]] = None
    """state indices to evaluate (default: all)"""
    num_envs: int = 100
    episodes_per_env: int = 100
    ks: List[int] = field(default_factory=lambda: [1, 10, 100])
    deterministic: bool = False
    """use the policy mean (identical rollouts from a fixed state) instead of sampled actions"""
    seed: int = 0
    out: Optional[str] = None
    """csv path (default: next to the init-state file)"""


def main(args: Args):
    torch.manual_seed(args.seed)
    n_states = len(torch.load(args.init_state_path, weights_only=False)["tee_xy"])
    indices = args.indices if args.indices is not None else list(range(n_states))
    total = args.num_envs * len(indices)

    raw = gym.make(args.env.id, num_envs=total, **gym_make_kwargs(args.env))
    policy = load_policy(args.checkpoint, raw.unwrapped, deterministic=args.deterministic)
    env_states = [idx for idx in indices for _ in range(args.num_envs)]
    env = ManiSkillVectorEnv(FixedInitStateWrapper(raw, per_env_init_state(args.init_state_path, env_states)),
                             total, ignore_terminations=True, record_metrics=True)
    obs, _ = env.reset(seed=args.seed)
    success = []
    while len(success) < args.episodes_per_env:
        obs, _, _, _, info = env.step(policy(obs))
        if "final_info" in info:
            assert info["_final_info"].all(), "episodes should end together (ignore_terminations=True)"
            success.append(info["final_info"]["episode"]["success_once"].bool().cpu())
            print(f"round {len(success)}/{args.episodes_per_env}", end="\r", flush=True)
    success = torch.stack(success, dim=1).view(len(indices), args.num_envs, args.episodes_per_env)

    rows, matrices = [], {}
    for j, idx in enumerate(indices):
        s, info = success[j], load_init_state(args.init_state_path, idx)["info"]
        row = dict(index=idx, tee_x=info["tee_xy"][0], tee_y=info["tee_xy"][1], tee_yaw=info["tee_yaw"][0])
        for k in args.ks:
            if k == 1:
                row["success@1"] = s[0].float().mean().item()
            else:
                row[f"success@{k}"] = s[:, :k].any(dim=1).float().mean().item()
        row["success_rate_all"] = s.float().mean().item()
        matrices[idx] = s
        rows.append(row)
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    out = args.out or os.path.join(os.path.dirname(args.init_state_path),
                                   f"success_at_k_{os.path.basename(os.path.dirname(args.checkpoint))}"
                                   f"{'_det' if args.deterministic else ''}.csv")
    with open(out, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    torch.save(dict(success=matrices, config=vars(args)), out.replace(".csv", "_episodes.pt"))
    print("wrote", out, "and per-episode success (num_envs x episodes) to", out.replace(".csv", "_episodes.pt"))
    os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
