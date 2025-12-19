from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  euler_xyz_from_quat,
  quat_from_euler_xyz,
  wrap_to_pi,
)


def _smoothstep(x: torch.Tensor, start: float, end: float) -> torch.Tensor:
  # Smooth ramp from 0 to 1 between [start, end].
  x = torch.clamp((x - start) / max(end - start, 1e-6), 0.0, 1.0)
  return x * x * (3 - 2 * x)


def _triangular_window(x: torch.Tensor, start: float, peak: float, end: float) -> torch.Tensor:
  """Symmetric triangular weight that peaks at ``peak`` within [start, end]."""
  rise = torch.clamp((x - start) / max(peak - start, 1e-6), 0.0, 1.0)
  fall = torch.clamp((end - x) / max(end - peak, 1e-6), 0.0, 1.0)
  return torch.minimum(rise, fall)


def _height_profile(
  phase: torch.Tensor,
  *,
  stance_height: float,
  crouch_height: float,
  apex_height: float,
  crouch_phase_end: float,
  launch_phase_end: float,
  flight_phase_end: float,
) -> torch.Tensor:
  """Piecewise height reference with a crouch, launch, flight hold, and landing."""

  crouch_interp = stance_height + (crouch_height - stance_height) * _smoothstep(
    phase, 0.0, crouch_phase_end
  )
  launch_interp = crouch_height + (apex_height - crouch_height) * _smoothstep(
    phase, crouch_phase_end, launch_phase_end
  )
  landing_interp = apex_height - (apex_height - stance_height) * _smoothstep(
    phase, flight_phase_end, 1.0
  )

  height = torch.where(phase < crouch_phase_end, crouch_interp, launch_interp)
  height = torch.where(phase < launch_phase_end, height, apex_height)
  height = torch.where(phase < flight_phase_end, height, landing_interp)
  return height


class BackflipCommand(CommandTerm):
  """Simple phase-based command for a single backflip maneuver."""

  cfg: BackflipCommandCfg

  def __init__(self, cfg: BackflipCommandCfg, env):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.asset_name]
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self._command = torch.zeros(self.num_envs, 3, device=self.device)
    self._target_upward_velocity = torch.zeros(self.num_envs, device=self.device)
    self._target_pitch_rate = torch.zeros(self.num_envs, device=self.device)
    self.takeoff_weight = torch.zeros(self.num_envs, device=self.device)
    self.spin_weight = torch.zeros(self.num_envs, device=self.device)
    self.landing_weight = torch.zeros(self.num_envs, device=self.device)

  def _update_metrics(self) -> None:
    self.metrics["phase"] = self.phase
    self.metrics["takeoff_weight"] = self.takeoff_weight
    self.metrics["spin_weight"] = self.spin_weight
    self.metrics["landing_weight"] = self.landing_weight

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.phase[env_ids] = 0.0
    self._command[env_ids] = 0.0
    self._target_upward_velocity[env_ids] = 0.0
    self._target_pitch_rate[env_ids] = 0.0
    self.takeoff_weight[env_ids] = 0.0
    self.spin_weight[env_ids] = 0.0
    self.landing_weight[env_ids] = 0.0

  def _update_command(self) -> None:
    dt = self._env.step_dt
    advance = dt / self.cfg.flip_duration
    self.phase = torch.clamp(self.phase + advance, max=1.0)
    phase = self.phase

    target_height = _height_profile(
      phase,
      stance_height=self.cfg.stance_height,
      crouch_height=self.cfg.crouch_height,
      apex_height=self.cfg.apex_height,
      crouch_phase_end=self.cfg.crouch_phase_end,
      launch_phase_end=self.cfg.launch_phase_end,
      flight_phase_end=self.cfg.flight_phase_end,
    )

    target_pitch = -2.0 * math.pi * _smoothstep(
      phase, self.cfg.spin_up_phase_start, self.cfg.landing_phase_start
    )

    self.takeoff_weight = _triangular_window(
      phase,
      start=self.cfg.takeoff_phase[0],
      peak=self.cfg.takeoff_phase[1],
      end=self.cfg.takeoff_phase[2],
    )
    self.spin_weight = _triangular_window(
      phase,
      start=self.cfg.spin_phase[0],
      peak=self.cfg.spin_phase[1],
      end=self.cfg.spin_phase[2],
    )
    self.landing_weight = torch.clamp(
      (phase - self.cfg.landing_phase_start) / (1.0 - self.cfg.landing_phase_start),
      min=0.0,
      max=1.0,
    )

    self._target_upward_velocity = self.cfg.takeoff_target_up_vel * self.takeoff_weight
    self._target_pitch_rate = self.cfg.spin_target_rate * self.spin_weight
    self._command[:, 0] = self.phase
    self._command[:, 1] = target_height
    self._command[:, 2] = target_pitch

  @property
  def command_tensor(self) -> torch.Tensor:
    return self._command

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def target_upward_velocity(self) -> torch.Tensor:
    return self._target_upward_velocity

  @property
  def target_pitch_rate(self) -> torch.Tensor:
    return self._target_pitch_rate

  def _debug_vis_impl(self, visualizer):
    # Visualize target base height and orientation arrow for the selected env.
    idx = visualizer.env_idx
    if idx >= self.num_envs:
      return
    target_height = self.command[idx, 1].item()
    target_pitch = self.command[idx, 2]
    target_quat = quat_from_euler_xyz(
      torch.tensor([0.0], device=self.device),
      torch.tensor([target_pitch], device=self.device),
      torch.tensor([0.0], device=self.device),
    )[0]
    base_pos = self.robot.data.root_link_pos_w[idx].cpu().numpy()
    base_pos[2] = target_height
    visualizer.add_frame(pos=base_pos, quat=target_quat.cpu().numpy(), scale=0.3)


@dataclass(kw_only=True)
class BackflipCommandCfg(CommandTermCfg):
  asset_name: str
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  flip_duration: float = 1.6
  crouch_height: float = 0.25
  stance_height: float = 0.30
  apex_height: float = 0.70
  crouch_phase_end: float = 0.18
  launch_phase_end: float = 0.38
  flight_phase_end: float = 0.82
  landing_phase_start: float = 0.86
  takeoff_phase: tuple[float, float, float] = (0.10, 0.24, 0.40)
  spin_phase: tuple[float, float, float] = (0.25, 0.50, 0.78)
  spin_up_phase_start: float = 0.18
  takeoff_target_up_vel: float = 4.2
  spin_target_rate: float = -7.0
  class_type: type[CommandTerm] = BackflipCommand


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_pitch(env, std: float, command_name: str = "backflip") -> torch.Tensor:
  command: BackflipCommand = env.command_manager.get_term(command_name)
  target_pitch = command.command_tensor[:, 2]
  asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
  _, current_pitch, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)
  pitch_error = wrap_to_pi(current_pitch - target_pitch)
  return torch.exp(-(pitch_error**2) / (2 * std**2))


def track_height(env, std: float, command_name: str = "backflip") -> torch.Tensor:
  command: BackflipCommand = env.command_manager.get_term(command_name)
  target_height = command.command_tensor[:, 1]
  asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
  height_error = asset.data.root_link_pos_w[:, 2] - target_height
  return torch.exp(-(height_error**2) / (2 * std**2))


def phase_progress(env, command_name: str = "backflip") -> torch.Tensor:
  command: BackflipCommand = env.command_manager.get_term(command_name)
  return command.command_tensor[:, 0]


def landing_upright(
  env,
  std: float,
  phase_start: float = 0.9,
  command_name: str = "backflip",
) -> torch.Tensor:
  command: BackflipCommand = env.command_manager.get_term(command_name)
  weight = torch.maximum(command.landing_weight, torch.tensor(0.0, device=command.device))
  asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
  _, current_pitch, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)
  pitch_error = wrap_to_pi(current_pitch)
  return weight * torch.exp(-(pitch_error**2) / (2 * std**2))


def takeoff_upward_velocity(
  env,
  target_vel: float,
  std: float,
  command_name: str = "backflip",
) -> torch.Tensor:
  command: BackflipCommand = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
  up_vel = asset.data.root_link_lin_vel_w[:, 2]
  vel_error = up_vel - torch.where(command.takeoff_weight > 0, command.target_upward_velocity, target_vel)
  return command.takeoff_weight * torch.exp(-(vel_error**2) / (2 * std**2))


def spin_rate(
  env,
  target_rate: float,
  std: float,
  command_name: str = "backflip",
) -> torch.Tensor:
  command: BackflipCommand = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
  pitch_rate = asset.data.root_link_ang_vel_b[:, 1]
  desired_rate = torch.where(command.spin_weight > 0, command.target_pitch_rate, target_rate)
  rate_error = pitch_rate - desired_rate
  return command.spin_weight * torch.exp(-(rate_error**2) / (2 * std**2))