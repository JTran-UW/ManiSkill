"""Concatenate fastsac.py --collect-replay-buffer exports along the env axis (equal row counts), e.g. a 50/50 goal-A /
goal-B expert buffer for PushT2Goal-v1.

    python reinforcement_learning/scripts/merge_replay_buffers.py runs/expA/replay_buffer.pt runs/expB/replay_buffer.pt \
        --out runs/expert_2goal_ab/replay_buffer.pt
"""
import os
from dataclasses import dataclass
from typing import List

import torch
import tyro

from reinforcement_learning.utils.buffers import ReplayBuffer


@dataclass
class Args:
    inputs: tyro.conf.Positional[List[str]]
    out: str


def main(args: Args):
    parts = [torch.load(p, weights_only=False) for p in args.inputs]
    rows = {p["buffer"]["observations"].shape[0] for p in parts}
    assert len(rows) == 1, f"row counts differ: {rows}"
    buf = {k: torch.cat([p["buffer"][k] for p in parts], dim=1) for k in ReplayBuffer.EXPORT_KEYS}
    first = parts[0]["buffer"]
    buf.update(n_envs=sum(p["buffer"]["n_envs"] for p in parts), n_steps=first["n_steps"], gamma=first["gamma"],
               pos=first["pos"], full=first["full"])
    metas = [p["metadata"] for p in parts]
    for path, m in zip(args.inputs, metas):
        print(f"{path}: {m['num_transitions']} transitions, {m['num_episodes']} episodes, success_once {m['success_once']:.3f}, "
              f"task kwargs {m['config']['env'].get('kwargs')}")
    meta = dict(metas[0], checkpoint=[m["checkpoint"] for m in metas], sources=args.inputs,
                num_transitions=sum(m["num_transitions"] for m in metas), num_episodes=sum(m["num_episodes"] for m in metas),
                success_once=sum(m["success_once"] * m["num_episodes"] for m in metas) / max(1, sum(m["num_episodes"] for m in metas)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"buffer": buf, "metadata": meta}, args.out)
    print(f"saved {args.out}: n_envs {buf['n_envs']}, rows {buf['observations'].shape[0]}, success_once {meta['success_once']:.3f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
