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
def main(args: Args) -> None:
  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = 1
  env_cfg.commands["twist"].profile = args.velocity_command_profile

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
    action = policy(observations)[0]
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