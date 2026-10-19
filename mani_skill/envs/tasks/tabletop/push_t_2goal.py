import math
from typing import Any, Optional, Sequence, Union

import sapien
import torch

from mani_skill.envs.tasks.tabletop.push_t_goal import PushTGoalEnv
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array

# default stick-tip rest point straight behind the T (world xy)
DEFAULT_TIP_XY = (-0.30, 0.0)
KEYFRAME_QPOS = (0.0, 0.392699, 0.0, -1.963495, 0.0, 2.356194, 0.785398)  # TableSceneBuilder panda_stick
TIP_Z = 0.024


def solve_tip_ik(env, tip_xy: torch.Tensor, iters: int = 200):
    """Panda-stick qpos putting the stick tip at world tip_xy (B, 2), z TIP_Z, keyframe tip orientation. Returns (qpos, err)."""
    from mani_skill.agents.controllers.utils.kinematics import Kinematics
    if getattr(env, "_kin", None) is None:
        arm = env.agent.controller.controllers["arm"]
        env._kin = Kinematics(env.agent.urdf_path, env.agent.ee_link_name, env.agent.robot, arm.active_joint_indices)
        q0 = torch.tensor([KEYFRAME_QPOS], device=env.device)
        env._tip_quat = matrix_to_quaternion(env._kin.pk_chain.forward_kinematics(q0).get_matrix()[:, :3, :3])
    kin, b = env._kin, tip_xy.shape[0]
    target_p = torch.cat([tip_xy, torch.full((b, 1), TIP_Z, device=tip_xy.device)], 1) - env.agent.robot.pose.p[:1]
    target = Pose.create_from_pq(target_p, env._tip_quat.expand(b, 4))
    lim = torch.tensor(kin.pk_chain.get_joint_limits(), device=tip_xy.device)
    q = torch.tensor([KEYFRAME_QPOS], device=tip_xy.device).expand(b, -1).clone()
    for _ in range(iters):
        q = torch.clamp(kin.compute_ik(target, q), lim[0], lim[1])
    err = torch.linalg.norm(kin.pk_chain.forward_kinematics(q).get_matrix()[:, :3, 3] - target_p, dim=1)
    return q, err


@register_env("PushT2Goal-v1", max_episode_steps=100)
class PushT2GoalEnv(PushTGoalEnv):
    """
    **Task Description:**
    Two-goal sparse PushT. The T starts near START_XY with its stem pointing away from the robot (+x), mirror-symmetric
    about the world x axis, and the stick tip starts straight behind it. Two goal T poses with the start yaw sit at
    START_XY +- (0, goal_dist): goal A (+y) and goal B (-y). Reward is sparse (+1 per step while the T covers >= 90% of
    a valid goal); observations carry no goal information, so the optimal policy is bimodal.

    valid_goals: "ab" (both rewarded), "a" or "b" (only that goal rewarded, for per-goal experts; obs unchanged).
    reset_datasets entries need tee_xy, tee_yaw, robot_qpos.

    **Observations (state):** arm qpos/qvel, tcp_pose (tip in the robot base frame), obj_pose_in_start (T in the start
    frame), tcp_pose_in_obj (tip in the T frame); poses as position + axis-angle.
    """

    SUPPORTED_REWARD_MODES = ("sparse", "none")
    START_XY = (-0.156, 0.0)
    START_YAW = -math.pi / 2

    def __init__(self, *args, valid_goals: str = "ab", goal_dist: float = 0.15, start_xy_noise: float = 0.02,
                 start_yaw_noise_deg: float = 10.0, tip_xy_noise: float = 0.01,
                 reset_datasets: Optional[Union[str, Sequence[str]]] = None,
                 reset_dataset_weights: Optional[Sequence[float]] = None, **kwargs):
        assert valid_goals in ("ab", "a", "b"), valid_goals
        self.valid_goals = valid_goals
        self.goal_dist = goal_dist
        self.start_xy_noise = start_xy_noise
        self.start_yaw_noise = math.radians(start_yaw_noise_deg)
        self.tip_xy_noise = tip_xy_noise
        super().__init__(*args, reset_datasets=reset_datasets, reset_dataset_weights=reset_dataset_weights, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        builder = self.scene.create_actor_builder()
        for half_size, y in (([0.1, 0.025], -0.0375), ([0.025, 0.075], 0.0625)):
            builder.add_box_visual(pose=sapien.Pose([0.0, y, 0.0]), half_size=[*half_size, 1e-4],
                                   material=sapien.render.RenderMaterial(base_color=[0.35, 0.35, 0.8, 1]))
        builder.initial_pose = sapien.Pose([0, 0, 0.1])
        self.goal_tee_b = builder.build_kinematic(name="goal_Tee_b")
        self._start_xy = torch.tensor(self.START_XY, device=self.device)
        self._goal_xy = torch.tensor([[self.START_XY[0], self.START_XY[1] + self.goal_dist],
                                      [self.START_XY[0], self.START_XY[1] - self.goal_dist]], device=self.device)
        yaw = torch.full((1,), self.START_YAW, device=self.device)
        self._w2g = [self._world_to_goal(self._goal_xy[i:i + 1], yaw) for i in range(2)]
        self._w2s = self._world_to_goal(self._start_xy[None], yaw)
        self._start_pose = Pose.create_from_pq(torch.tensor([*self.START_XY, 1e-3], device=self.device), self._yaw_to_quat(yaw))
        self._default_qpos = None

    def _default_robot_qpos(self, b: int):
        """Symmetric rest qpos with the tip at DEFAULT_TIP_XY (IK from the table keyframe on first use), plus init noise."""
        if self._default_qpos is None:
            self._default_qpos = solve_tip_ik(self, torch.tensor([DEFAULT_TIP_XY], device=self.device))[0][0]
        return self._default_qpos.expand(b, -1) + torch.randn(b, 7, device=self.device) * self.robot_init_qpos_noise

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super(PushTGoalEnv, self)._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            yaw0 = torch.full((b,), self.START_YAW)
            for actor, xy in ((self.goal_tee, self._goal_xy[0]), (self.goal_tee_b, self._goal_xy[1])):
                p = torch.cat([xy.expand(b, 2), torch.full((b, 1), 1e-3)], dim=1)
                actor.set_pose(Pose.create_from_pq(p=p, q=self._yaw_to_quat(yaw0)))
            if self._reset_data:
                ids, s = self._sample_datasets(b)
                tee_xy, tee_yaw, qpos = s["tee_xy"], s["tee_yaw"], s["robot_qpos"]
                self.reset_dataset_id[env_idx] = ids
            else:
                tee_xy = self._start_xy + (torch.rand(b, 2) * 2 - 1) * self.start_xy_noise
                tee_yaw = yaw0 + (torch.rand(b) * 2 - 1) * self.start_yaw_noise
                qpos = self._default_robot_qpos(b)
            if qpos is not None:
                self.agent.robot.set_qpos(qpos)
                self.agent.robot.set_qvel(torch.zeros_like(qpos))
            tee_p = torch.cat([tee_xy, torch.full((b, 1), 0.04 / 2 + 1e-3)], dim=1)
            self.tee.set_pose(Pose.create_from_pq(p=tee_p, q=self._yaw_to_quat(tee_yaw)))

    def _sample_datasets(self, b: int):
        ids = torch.multinomial(self._reset_weights, b, replacement=True)
        fields = ("tee_xy", "tee_yaw", "robot_qpos")
        out = {k: torch.zeros((b,) + self._reset_data[0][k].shape[1:], device=self.device) for k in fields}
        for i, d in enumerate(self._reset_data):
            rows = (ids == i).nonzero().squeeze(1)
            if len(rows):
                entries = torch.randint(len(d["tee_xy"]), (len(rows),), device=self.device)
                for k in fields:
                    out[k][rows] = d[k][entries]
        return ids, out

    def goal_overlaps(self, tee_p=None, tee_q=None):
        n = self.num_envs if tee_p is None else tee_p.shape[0]
        return [self.pseudo_render_intersection(tee_p, tee_q, w.expand(n, 3, 3)) for w in self._w2g]

    def evaluate(self):
        ov_a, ov_b = self.goal_overlaps()
        success_a, success_b = ov_a >= self.intersection_thresh, ov_b >= self.intersection_thresh
        success = {"ab": success_a | success_b, "a": success_a, "b": success_b}[self.valid_goals]
        return {"success": success, "success_a": success_a, "success_b": success_b,
                "reset_dataset": self.reset_dataset_id.clone()}

    def _get_obs_extra(self, info: dict):
        tcp = self.agent.tcp.pose
        obs = dict(tcp_pose=self._pose_pos_axis_angle(self.agent.robot.pose.inv() * tcp))
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose_in_start=self._pose_pos_axis_angle(self._start_pose.inv() * self.tee.pose),
                tcp_pose_in_obj=self._pose_pos_axis_angle(self.tee.pose.inv() * tcp),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: Array, info: dict):
        return info["success"].float()
