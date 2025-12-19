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


class BackflipCommand(CommandTerm):
  """Simple phase-based command for a single backflip maneuver."""

  cfg: BackflipCommandCfg

  def __init__(self, cfg: BackflipCommandCfg, env):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.asset_name]
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.command = torch.zeros(self.num_envs, 3, device=self.device)

  def _update_metrics(self) -> None:
    self.metrics["phase"] = self.phase

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.phase[env_ids] = 0.0
    self.command[env_ids] = 0.0

  def _update_command(self) -> None:
    dt = self._env.step_dt
    advance = dt / self.cfg.flip_duration
    self.phase = torch.clamp(self.phase + advance, max=1.0)
    target_pitch = -2.0 * math.pi * self.phase
    target_height = self.cfg.stance_height + (
      self.cfg.apex_height - self.cfg.stance_height
    ) * torch.sin(math.pi * self.phase)
    self.command[:, 0] = self.phase
    self.command[:, 1] = target_height
    self.command[:, 2] = target_pitch

  @property
  def command_tensor(self) -> torch.Tensor:
    return self.command

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
  flip_duration: float = 1.6
  stance_height: float = 0.30
  apex_height: float = 0.65
  class_type: type[CommandTerm] = BackflipCommand


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_pitch(env, std: float, command_name: str = "backflip") -> torch.Tensor:
  command: BackflipCommand = env.command_manager.get_command(command_name)
  target_pitch = command.command_tensor[:, 2]
  asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
  current_pitch = euler_xyz_from_quat(asset.data.root_link_quat_w)[:, 1]
  pitch_error = wrap_to_pi(current_pitch - target_pitch)
  return torch.exp(-(pitch_error**2) / (2 * std**2))


def track_height(env, std: float, command_name: str = "backflip") -> torch.Tensor:
  command: BackflipCommand = env.command_manager.get_command(command_name)
  target_height = command.command_tensor[:, 1]
  asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
  height_error = asset.data.root_link_pos_w[:, 2] - target_height
  return torch.exp(-(height_error**2) / (2 * std**2))


def phase_progress(env, command_name: str = "backflip") -> torch.Tensor:
  command: BackflipCommand = env.command_manager.get_command(command_name)
  return command.command_tensor[:, 0]


def landing_upright(
  env,
  std: float,
  phase_start: float = 0.9,
  command_name: str = "backflip",
) -> torch.Tensor:
  command: BackflipCommand = env.command_manager.get_command(command_name)
  weight = torch.clamp(
    (command.command_tensor[:, 0] - phase_start) / (1.0 - phase_start),
    min=0.0,
    max=1.0,
  )
  asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
  current_pitch = euler_xyz_from_quat(asset.data.root_link_quat_w)[:, 1]
  pitch_error = wrap_to_pi(current_pitch)
  return weight * torch.exp(-(pitch_error**2) / (2 * std**2))