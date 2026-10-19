"""Reset a ManiSkill env to one saved initial state (from reinforcement_learning/scripts/save_init_states.py)."""
from typing import Any, Dict, List

import gymnasium as gym
import torch


def load_init_state(path: str, index: int) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    n = len(data["tee_xy"]) if "tee_xy" in data else next(iter(data["env_state"]["actors"].values())).shape[0]
    if not 0 <= index < n:
        raise IndexError(f"init_state_index {index} out of range for {path} ({n} states)")
    state = {group: {k: v[index:index + 1] for k, v in data["env_state"][group].items()}
             for group in ("actors", "articulations") if group in data["env_state"]}
    info = {k: [round(x, 4) for x in data[k][index].reshape(-1).tolist()] for k in ("tee_xy", "tee_yaw") if k in data}
    return dict(state=state, info=info, path=path, index=index)


def per_env_init_state(path: str, env_indices: List[int]) -> Dict[str, Any]:
    """One saved state per env: env i resets to saved state env_indices[i]."""
    states = [load_init_state(path, i)["state"] for i in env_indices]
    state = {g: {k: torch.cat([s[g][k] for s in states]) for k in states[0][g]} for g in states[0]}
    return dict(state=state, info=None, path=path, index=env_indices)


class FixedInitStateWrapper(gym.Wrapper):
    """Every reset (including ManiSkillVectorEnv auto-resets) restores a saved state instead of sampling one: the same
    state for all envs (1 saved row) or env i's own row (num_envs saved rows, see per_env_init_state)."""

    def __init__(self, env: gym.Env, init_state: Dict[str, Any]):
        super().__init__(env)
        base = env.unwrapped
        current = base.get_state_dict()
        self.state = {}
        for group, saved in init_state["state"].items():
            present = current.get(group, {})
            missing = sorted(set(saved) - set(present))
            if missing:
                print(f"[init_state] skipping {group} not in this env: {missing}")
            self.state[group] = {k: v.to(base.device) for k, v in saved.items() if k in present}
        rows = next(iter(next(iter(self.state.values())).values())).shape[0]
        if rows not in (1, base.num_envs):
            raise ValueError(f"init state has {rows} rows; need 1 or num_envs={base.num_envs}")
        unset = sorted(set(current.get("actors", {})) - set(self.state.get("actors", {})))
        if unset:
            print(f"[init_state] actors not in the saved state (keep their pose from before the reset): {unset}")

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        env_idx = options.get("env_idx")
        n = self.env.unwrapped.num_envs if env_idx is None else len(env_idx)
        def rows(v):
            if v.shape[0] == 1:
                return v.expand(n, *v.shape[1:])
            return v if env_idx is None else v[env_idx.to(v.device)]
        states = {g: {k: rows(v) for k, v in d.items()} for g, d in self.state.items()}
        options["reset_to_env_states"] = dict(env_states=states)
        out = self.env.reset(seed=seed, options=options)
        self._flush_contacts()
        return out

    def _flush_contacts(self):
        """GPU PhysX keeps the previous frame's contacts across a state restore, which jolts restored bodies on the next
        step (e.g. the T after an episode that ended in stick contact). Step once, then restore every env's state."""
        base = self.env.unwrapped
        if not base.gpu_sim_enabled:
            return
        state = base.get_state_dict()
        base.scene.step()
        base.set_state_dict(state)
        base.scene._gpu_apply_all()
        base.scene.px.gpu_update_articulation_kinematics()
        base.scene._gpu_fetch_all()
