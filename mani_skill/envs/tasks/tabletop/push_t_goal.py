from typing import Any, List, Optional, Sequence, Union

import torch

from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_axis_angle
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array

RESET_DATASET_FIELDS = ("tee_xy", "tee_yaw", "goal_xy", "goal_yaw", "robot_qpos")


@register_env("PushTGoal-v1", max_episode_steps=100)
class PushTGoalEnv(PushTEnv):
    """
    **Task Description:**
    Goal-conditioned PushT: each episode samples the goal T pose from the same distribution as the initial T
    (xy in the PushT spawnbox, yaw uniform in [0, 2pi)). Initial poses that already satisfy the goal are resampled.

    With reset_datasets (paths to reset_states.pt files from collect_pusht_goal_resets.py), each reset instead picks a
    dataset per env (uniformly, or by reset_dataset_weights) and a random entry from it, setting the goal, T and arm.
    info["reset_dataset"] is the dataset index of each env's episode (-1 for default resets).

    **Observations (state):** arm qpos/qvel, plus relative poses as position + axis-angle (like UWLab OmniReset):
    tcp_pose (stick tip in the robot base frame), obj_pose_in_goal (T in the goal T frame), tcp_pose_in_obj (tip in the T frame).

    **Reward:** sparse by default (+1 while the T covers >= 90% of the goal T).
    """

    SUPPORTED_REWARD_MODES = ("sparse", "normalized_dense", "dense", "none")
    max_init_resamples = 10

    def __init__(self, *args, reset_datasets: Optional[Union[str, Sequence[str]]] = None,
                 reset_dataset_weights: Optional[Sequence[float]] = None, **kwargs):
        if isinstance(reset_datasets, str):
            reset_datasets = [p for p in reset_datasets.split(",") if p]
        self.reset_dataset_paths: List[str] = list(reset_datasets or [])
        self._reset_data = [
            {k: torch.as_tensor(v).float() for k, v in torch.load(p, weights_only=False).items() if k in RESET_DATASET_FIELDS}
            for p in self.reset_dataset_paths
        ]
        n = len(self._reset_data)
        weights = torch.as_tensor(reset_dataset_weights if reset_dataset_weights is not None else [1.0] * n, dtype=torch.float)
        assert len(weights) == n, "reset_dataset_weights needs one weight per reset dataset"
        self._reset_weights = weights / weights.sum() if n else weights
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.reset_dataset_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._reset_data = [{k: v.to(self.device) for k, v in d.items()} for d in self._reset_data]
        self._reset_weights = self._reset_weights.to(self.device)

    @property
    def goal_yaw(self) -> torch.Tensor:
        return self.quat_to_z_euler(self.goal_tee.pose.q)

    def goal_world_to_goal(self) -> torch.Tensor:
        return self._world_to_goal(self.goal_tee.pose.p[:, :2], self.goal_yaw)

    def _sample_tee_xy_yaw(self, b: int):
        xy = torch.zeros((b, 2), device=self.device)
        xy[:, 0] = self.goal_offset[0] + torch.rand(b, device=self.device) * self.tee_spawnbox_xlength + self.tee_spawnbox_xoffset
        xy[:, 1] = self.goal_offset[1] + torch.rand(b, device=self.device) * self.tee_spawnbox_ylength + self.tee_spawnbox_yoffset
        yaw = torch.rand(b, device=self.device) * (2 * torch.pi)
        return xy, yaw

    @staticmethod
    def _yaw_to_quat(yaw: torch.Tensor):
        q = torch.zeros((yaw.shape[0], 4), device=yaw.device)
        q[:, 0] = (yaw / 2).cos()
        q[:, 3] = (yaw / 2).sin()
        return q

    @staticmethod
    def _world_to_goal(xy: torch.Tensor, yaw: torch.Tensor):
        goal_to_world = torch.zeros((xy.shape[0], 3, 3), device=xy.device)
        goal_to_world[:, 0, 0] = yaw.cos()
        goal_to_world[:, 0, 1] = -yaw.sin()
        goal_to_world[:, 1, 0] = yaw.sin()
        goal_to_world[:, 1, 1] = yaw.cos()
        goal_to_world[:, 0:2, 2] = xy
        goal_to_world[:, 2, 2] = 1
        return torch.linalg.inv(goal_to_world)

    def _sample_default(self, b: int):
        goal_xy, goal_yaw = self._sample_tee_xy_yaw(b)
        world_to_goal = self._world_to_goal(goal_xy, goal_yaw)
        tee_xy, tee_yaw = self._sample_tee_xy_yaw(b)
        for _ in range(self.max_init_resamples):
            solved = self.pseudo_render_intersection(
                torch.cat([tee_xy, torch.zeros((b, 1), device=self.device)], dim=1), self._yaw_to_quat(tee_yaw), world_to_goal
            ) >= self.intersection_thresh
            if not solved.any():
                break
            new_xy, new_yaw = self._sample_tee_xy_yaw(int(solved.sum()))
            tee_xy[solved], tee_yaw[solved] = new_xy, new_yaw
        return goal_xy, goal_yaw, tee_xy, tee_yaw

    def _sample_datasets(self, b: int):
        ids = torch.multinomial(self._reset_weights, b, replacement=True)
        out = {k: torch.zeros((b,) + self._reset_data[0][k].shape[1:], device=self.device) for k in RESET_DATASET_FIELDS}
        for i, d in enumerate(self._reset_data):
            rows = (ids == i).nonzero().squeeze(1)
            if len(rows):
                entries = torch.randint(len(d["tee_xy"]), (len(rows),), device=self.device)
                for k in RESET_DATASET_FIELDS:
                    out[k][rows] = d[k][entries]
        return ids, out

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            if self._reset_data:
                ids, s = self._sample_datasets(b)
                goal_xy, goal_yaw, tee_xy, tee_yaw = s["goal_xy"], s["goal_yaw"], s["tee_xy"], s["tee_yaw"]
                self.agent.robot.set_qpos(s["robot_qpos"])
                self.agent.robot.set_qvel(torch.zeros_like(s["robot_qpos"]))
                self.reset_dataset_id[env_idx] = ids
            else:
                goal_xy, goal_yaw, tee_xy, tee_yaw = self._sample_default(b)
            goal_p = torch.cat([goal_xy, torch.full((b, 1), 1e-3)], dim=1)
            self.goal_tee.set_pose(Pose.create_from_pq(p=goal_p, q=self._yaw_to_quat(goal_yaw)))
            tee_p = torch.cat([tee_xy, torch.full((b, 1), 0.04 / 2 + 1e-3)], dim=1)
            self.tee.set_pose(Pose.create_from_pq(p=tee_p, q=self._yaw_to_quat(tee_yaw)))

    def evaluate(self):
        inter_area = self.pseudo_render_intersection(world_to_goal_trans=self.goal_world_to_goal())
        return {"success": inter_area >= self.intersection_thresh, "reset_dataset": self.reset_dataset_id.clone()}

    @staticmethod
    def _pose_pos_axis_angle(pose: Pose) -> torch.Tensor:
        return torch.cat([pose.p, quaternion_to_axis_angle(pose.q)], dim=-1)

    def _get_obs_extra(self, info: dict):
        tcp = self.agent.tcp.pose
        obs = dict(tcp_pose=self._pose_pos_axis_angle(self.agent.robot.pose.inv() * tcp))
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose_in_goal=self._pose_pos_axis_angle(self.goal_tee.pose.inv() * self.tee.pose),
                tcp_pose_in_obj=self._pose_pos_axis_angle(self.tee.pose.inv() * tcp),
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        tee_z_eulers = self.quat_to_z_euler(self.tee.pose.q)
        rot_rew = (tee_z_eulers - self.goal_yaw).cos()
        reward = (((rot_rew + 1) / 2) ** 2) / 2

        tee_to_goal_pose = self.tee.pose.p[:, 0:2] - self.goal_tee.pose.p[:, 0:2]
        tee_to_goal_pose_dist = torch.linalg.norm(tee_to_goal_pose, axis=1)
        reward += ((1 - torch.tanh(5 * tee_to_goal_pose_dist)) ** 2) / 2

        tcp_to_push_pose = self.tee.pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reward += ((1 - torch.tanh(5 * tcp_to_push_pose_dist)).sqrt()) / 20

        reward[info["success"]] = 3
        return reward

    def compute_sparse_reward(self, obs: Any, action: Array, info: dict):
        return info["success"].float()
