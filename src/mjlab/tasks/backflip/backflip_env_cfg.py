"""Backflip task environment configuration."""

from copy import deepcopy
import math

from mjlab.asset_zoo.robots.unitree_go2 import (
  GO2_BACKFLIP_ACTION_SCALE,
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_root_state_uniform, reset_scene_to_default
from mjlab.envs.mdp.rewards import action_rate_l2, joint_pos_limits
from mjlab.envs.mdp.terminations import root_height_below_minimum, time_out
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  CommandTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.backflip import mdp
from mjlab.tasks.backflip.mdp import BackflipCommandCfg
from mjlab.tasks.velocity.mdp.observations import (
  base_ang_vel,
  base_lin_vel,
  joint_pos_rel,
  joint_vel_rel,
  last_action,
  projected_gravity,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

SCENE_CFG = SceneCfg(
  terrain=None,
  num_envs=1,
  extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="trunk",
  distance=3.0,
  elevation=-10.0,
  azimuth=90.0,
)

SIM_CFG = SimulationCfg(
  nconmax=40,
  njmax=300,
  mujoco=MujocoCfg(
    timestep=0.0025,
    iterations=20,
    ls_iterations=40,
  ),
)


def create_backflip_env_cfg() -> ManagerBasedRlEnvCfg:
  scene = deepcopy(SCENE_CFG)
  scene.entities = {"robot": get_go2_robot_cfg()}

  viewer = deepcopy(VIEWER_CONFIG)

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      asset_name="robot",
      actuator_names=(".*",),
      scale=GO2_BACKFLIP_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "backflip": BackflipCommandCfg(asset_name="robot"),
  }

  policy_terms: dict[str, ObservationTermCfg] = {
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=joint_vel_rel,
      noise=Unoise(n_min=-1.0, n_max=1.0),
    ),
    "base_lin_vel": ObservationTermCfg(
      func=base_lin_vel,
      noise=Unoise(n_min=-0.4, n_max=0.4),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "last_action": ObservationTermCfg(
      func=last_action, params={"action_name": "joint_pos"}
    ),
    "command": ObservationTermCfg(
      func=lambda env: env.command_manager.get_term("backflip").command
    ),
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=policy_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  events = {
    "reset_scene": EventTermCfg(
      func=reset_scene_to_default,
      mode="reset",
    ),
    "reset_base": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.05, 0.05),
          "y": (-0.05, 0.05),
          "z": (-0.02, 0.02),
          "roll": (-0.05, 0.05),
          "pitch": (-0.05, 0.05),
          "yaw": (-0.2, 0.2),
        },
        "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
      },
    ),
  }

  rewards = {
    "takeoff_upward_velocity": RewardTermCfg(
      func=mdp.takeoff_upward_velocity,
      weight=3.5,
      params={
        "target_vel": 4.2,
        "std": math.sqrt(0.6),
        "command_name": "backflip",
      },
    ),
    "phase_progress": RewardTermCfg(
      func=mdp.phase_progress,
      weight=0.3,
      params={"command_name": "backflip"},
    ),
    "track_pitch": RewardTermCfg(
      func=mdp.track_pitch,
      weight=3.0,
      params={"std": math.sqrt(0.15), "command_name": "backflip"},
    ),
    "track_height": RewardTermCfg(
      func=mdp.track_height,
      weight=1.75,
      params={"std": math.sqrt(0.05), "command_name": "backflip"},
    ),
    "spin_rate": RewardTermCfg(
      func=mdp.spin_rate,
      weight=2.5,
      params={
        "target_rate": -7.0,
        "std": math.sqrt(0.6),
        "command_name": "backflip",
      },
    ),
    "landing_upright": RewardTermCfg(
      func=mdp.landing_upright,
      weight=3.0,
      params={"std": math.sqrt(0.1), "command_name": "backflip"},
    ),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.05),
    "dof_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-0.5),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "low_height": TerminationTermCfg(
      func=root_height_below_minimum,
      time_out=False,
      params={"minimum_height": 0.08},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    events=events,
    episode_length_s=3.0,
    sim=SIM_CFG,
    viewer=viewer,
    decimation=2,
  )