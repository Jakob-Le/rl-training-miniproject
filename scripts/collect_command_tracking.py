"""Collect commanded vs. measured velocity traces for deliverable #4.

This script runs a single-environment rollout with the scripted velocity
profile (by default the deliverable sequence) and writes a CSV file with
commanded and actual linear/angular velocities for downstream plotting.
Use it from Colab with `uv run python scripts/collect_command_tracking.py ...`
so imports resolve via the installed project package.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import sys

import torch
import tyro

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp.velocity_command import (
  ScriptedVelocityCommandCfg,
  UniformVelocityCommandCfg,
)


@dataclass
class Args:
  """CLI arguments for collecting command-tracking data."""

  checkpoint_file: Path
  """Path to a saved policy checkpoint (TorchScript or state dict)."""

  output_csv: Path = Path("logs/command_tracking.csv")
  """Where to save the commanded vs. measured velocity traces."""

  task: str = "Mjlab-Velocity-Flat-Unitree-Go1"
  """Registered task name to load the environment configuration."""

  velocity_command_profile: str = "deliverable"
  """Which command profile to run ("deliverable" follows the assignment)."""

  num_steps: int = 500
  """Number of environment steps to roll out (500 covers 4×125-step segments)."""

  device: str = "cuda:0"
  """Device for the environment and policy."""

  render: bool = False
  """Enable on-screen rendering (not needed for offline CSV capture)."""

  viewer: Optional[str] = None
  """Override viewer backend if rendering (e.g., "viser")."""


@torch.no_grad()
def _apply_velocity_command_profile(env_cfg, profile: str) -> None:
  """Swap the twist command to the scripted deliverable sequence if requested."""

  if profile == "none":
    return

  twist_cfg = env_cfg.commands.get("twist") if env_cfg.commands else None
  if twist_cfg is None:
    raise ValueError("Velocity command profile requires a 'twist' command in the config.")
  if not isinstance(twist_cfg, UniformVelocityCommandCfg):
    raise TypeError("Velocity command profile only supports UniformVelocityCommandCfg tasks.")

  deliverable_sequence = (
    (0.6, 0.0, 0.0, 125),
    (0.0, 0.4, 0.0, 125),
    (0.0, 0.0, 0.4, 125),
    (0.5, 0.0, 0.3, 125),
  )

  ranges = UniformVelocityCommandCfg.Ranges(
    lin_vel_x=twist_cfg.ranges.lin_vel_x,
    lin_vel_y=twist_cfg.ranges.lin_vel_y,
    ang_vel_z=twist_cfg.ranges.ang_vel_z,
    heading=None,
  )

  env_cfg.commands["twist"] = ScriptedVelocityCommandCfg(
    asset_name=twist_cfg.asset_name,
    heading_command=False,
    heading_control_stiffness=twist_cfg.heading_control_stiffness,
    rel_standing_envs=0.0,
    rel_heading_envs=0.0,
    init_velocity_prob=0.0,
    resampling_time_range=(0.0, 0.0),
    ranges=ranges,
    sequence=deliverable_sequence,
    loop_sequence=True,
  )


def _strip_randomization(env_cfg) -> None:
  """Disable startup/interval perturbations for clean evaluation."""

  if env_cfg.events is None:
    return

  # Remove pushes and friction randomization to avoid unintended resets or
  # contact variability while gathering tracking curves.
  env_cfg.events.pop("push_robot", None)
  env_cfg.events.pop("foot_friction", None)


def main(args: Args) -> None:
  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = 1
  _apply_velocity_command_profile(env_cfg, args.velocity_command_profile)
  _strip_randomization(env_cfg)

  render_mode: Optional[str]
  if args.render:
    render_mode = args.viewer or "human"
    env_cfg.viewer.render = True
    if args.viewer:
      env_cfg.viewer.viewer_type = args.viewer
  else:
    render_mode = None
    env_cfg.viewer.render = False

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=render_mode)

  agent_cfg = load_rl_cfg(args.task)
  runner_cls = load_runner_cls(args.task)
  if runner_cls is None:
    from rsl_rl.runners import OnPolicyRunner

    runner_cls = OnPolicyRunner

  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = runner_cls(vec_env, asdict(agent_cfg), device=args.device)
  runner.load(str(args.checkpoint_file), map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)

  observations, _ = vec_env.reset()
  rows: list[dict[str, float | int]] = []

  for step in range(args.num_steps):
    # The inference policy already returns batched actions with shape
    # (num_envs, action_dim); pass them through directly so the vec-env
    # wrapper keeps the expected 2D layout.
    action = policy(observations)
    observations, _, _, _ = vec_env.step(action)

    command = env.command_manager.get_command("twist").cpu().numpy()[0]
    lin_vel = env.scene["robot"].data.root_link_lin_vel_b.cpu().numpy()[0]
    yaw_vel = env.scene["robot"].data.root_link_ang_vel_b.cpu().numpy()[0, 2]

    rows.append(
      {
        "step": step,
        "cmd_vx": float(command[0]),
        "cmd_vy": float(command[1]),
        "cmd_wz": float(command[2]),
        "vel_vx": float(lin_vel[0]),
        "vel_vy": float(lin_vel[1]),
        "vel_wz": float(yaw_vel),
      }
    )

  vec_env.close()

  args.output_csv.parent.mkdir(parents=True, exist_ok=True)
  with args.output_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
      f, fieldnames=["step", "cmd_vx", "cmd_vy", "cmd_wz", "vel_vx", "vel_vy", "vel_wz"]
    )
    writer.writeheader()
    writer.writerows(rows)

  print(f"Saved command-tracking traces to: {args.output_csv}")


if __name__ == "__main__":
  # Accept both ``--checkpoint_file`` (notebook-style) and ``--checkpoint-file``
  # (Tyro's default) by normalizing underscores to hyphens before parsing.
  normalized_args = []
  for arg in sys.argv[1:]:
    if arg.startswith("--"):
      name, *rest = arg[2:].split("=", 1)
      name = name.replace("_", "-")
      arg = "--" + name + ("=" + rest[0] if rest else "")
    normalized_args.append(arg)

  tyro.cli(main, args=normalized_args)