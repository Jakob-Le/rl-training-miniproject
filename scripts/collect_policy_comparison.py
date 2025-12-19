"""Collect comparison traces for deliverable section 2.

This helper runs two policies (A and B) on the same velocity-command profile
and dumps a CSV containing:
- commanded linear/angular velocities
- front-left foot position (world frame) for each policy
- slip velocity (contact-weighted foot speed) for each policy
- linear tracking error (norm of commanded vs. measured vx/vy in body frame)

Use this to generate plots comparing gait-shaping rewards or privileged critic
observations by supplying the appropriate checkpoints for policy A and B.
"""
from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import sys

import torch
import tyro

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


@dataclass
class Args:
  """CLI arguments for running the comparison rollout."""

  policy_a: Path
  """Checkpoint path for policy A (e.g., with gait shaping)."""

  policy_b: Path
  """Checkpoint path for policy B (e.g., without gait shaping)."""

  policy_a_privileged_critic: bool = True
  """Whether policy A was trained with privileged critic observations enabled."""

  policy_b_privileged_critic: bool = True
  """Whether policy B was trained with privileged critic observations enabled."""

  policy_a_label: str = "policy_a"
  """Column prefix label for policy A."""

  policy_b_label: str = "policy_b"
  """Column prefix label for policy B."""

  output_csv: Path = Path("logs/policy_comparison.csv")
  """Where to store the combined traces."""

  task: str = "Mjlab-Velocity-Flat-Unitree-Go1"
  """Registered task name."""

  velocity_command_profile: str = "deliverable"
  """Command profile to run (use "deliverable" for the assignment sequence)."""

  num_steps: int = 500
  """Number of environment steps for each rollout (500 covers 4×125 steps)."""

  device: str = "cuda:0"
  """Device for environment and policy execution."""

  render: bool = False
  """Enable on-screen rendering while collecting traces."""

  viewer: Optional[str] = None
  """Viewer override if rendering (e.g., "viser")."""


def _apply_common_env_edits(env_cfg, args: Args):
  """Make shared environment edits used by both rollouts."""

  env_cfg.scene.num_envs = 1
  env_cfg.commands["twist"].profile = args.velocity_command_profile

  # Avoid joint name/id consistency checks on reset that can trip when regex
  # patterns are used. Clearing the names lets the manager operate on all
  # joints without providing both names and ids simultaneously.
  reset_cfg = env_cfg.events.get("reset_robot_joints") if env_cfg.events else None
  if reset_cfg is not None:
    asset_cfg = reset_cfg.params.get("asset_cfg") if reset_cfg.params else None
    if asset_cfg is not None:
      asset_cfg.joint_names = None

  # Disable pushes for clean comparisons.
  if env_cfg.events is not None and "push_robot" in env_cfg.events:
    env_cfg.events.pop("push_robot")


def _run_single_rollout(
  base_cfg, policy_path: Path, label: str, privileged_critic: bool, args: Args
):
  env_cfg = deepcopy(base_cfg)
  if not privileged_critic:
    env_cfg.observations["critic"].terms = env_cfg.observations["policy"].terms

  if args.render:
    render_mode: Optional[str] = args.viewer or "human"
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
  runner.load(str(policy_path), map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)

  observations, _ = vec_env.reset()

  robot = env.scene["robot"]
  fl_cfg = SceneEntityCfg("robot", site_names=("FL",))
  fl_cfg.resolve(env.scene)
  assert isinstance(fl_cfg.site_ids, list)
  fl_id = fl_cfg.site_ids[0]

  rows: list[dict[str, float]] = []
  for step in range(args.num_steps):
    action = policy(observations)
    observations, _, _, _ = vec_env.step(action)

    command = env.command_manager.get_command("twist")
    assert command is not None
    cmd_np = command.cpu().numpy()[0]

    foot_pos = robot.data.site_pos_w[:, fl_id].cpu().numpy()[0]

    contact_sensor = env.scene["feet_ground_contact"]
    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()
    foot_vel_xy = robot.data.site_lin_vel_w[:, fl_cfg.site_ids, :2]  # [B, N, 2]
    vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
    slip = (
      torch.sum(vel_xy_norm * in_contact)
      / torch.clamp(torch.sum(in_contact), min=1)
    ).item()

    lin_vel_b = robot.data.root_link_lin_vel_b.cpu().numpy()[0]
    tracking_error = float(
      torch.norm(
        torch.as_tensor(cmd_np[:2])
        - torch.as_tensor(lin_vel_b[:2]),
        dim=0,
      )
    )

    rows.append(
      {
        "step": step,
        "cmd_vx": float(cmd_np[0]),
        "cmd_vy": float(cmd_np[1]),
        "cmd_wz": float(cmd_np[2]),
        f"{label}_foot_x": float(foot_pos[0]),
        f"{label}_foot_y": float(foot_pos[1]),
        f"{label}_foot_z": float(foot_pos[2]),
        f"{label}_slip_vel": float(slip),
        f"{label}_lin_tracking_error": tracking_error,
      }
    )

  vec_env.close()
  return rows


@torch.no_grad()
def _rollout(policy_path: Path, label: str, args: Args) -> list[dict[str, float]]:
  base_cfg = deepcopy(load_env_cfg(args.task))
  _apply_common_env_edits(base_cfg, args)

  desired_privileged = (
    args.policy_a_privileged_critic
    if label == args.policy_a_label
    else args.policy_b_privileged_critic
  )

  try:
    return _run_single_rollout(
      base_cfg, policy_path, label, privileged_critic=desired_privileged, args=args
    )
  except RuntimeError as exc:
    # If the checkpoint has a smaller critic input (policy-only), retry without
    # privileged critic terms so comparison still works.
    if desired_privileged and "critic.0.weight" in str(exc):
      print(
        f"[WARN] {label}: critic shape mismatch; retrying without privileged critic observations"
      )
      return _run_single_rollout(
        base_cfg, policy_path, label, privileged_critic=False, args=args
      )
    raise


def main(args: Args) -> None:
  rows_a = _rollout(args.policy_a, args.policy_a_label, args)
  rows_b = _rollout(args.policy_b, args.policy_b_label, args)

  if len(rows_a) != len(rows_b):
    raise ValueError("Rollouts have different lengths; check num_steps")

  args.output_csv.parent.mkdir(parents=True, exist_ok=True)
  with args.output_csv.open("w", newline="") as f:
    fieldnames = [
      "step",
      "cmd_vx",
      "cmd_vy",
      "cmd_wz",
      f"{args.policy_a_label}_foot_x",
      f"{args.policy_a_label}_foot_y",
      f"{args.policy_a_label}_foot_z",
      f"{args.policy_a_label}_slip_vel",
      f"{args.policy_a_label}_lin_tracking_error",
      f"{args.policy_b_label}_foot_x",
      f"{args.policy_b_label}_foot_y",
      f"{args.policy_b_label}_foot_z",
      f"{args.policy_b_label}_slip_vel",
      f"{args.policy_b_label}_lin_tracking_error",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row_a, row_b in zip(rows_a, rows_b):
      merged = {
        **{k: row_a[k] for k in ("step", "cmd_vx", "cmd_vy", "cmd_wz")},
        f"{args.policy_a_label}_foot_x": row_a[f"{args.policy_a_label}_foot_x"],
        f"{args.policy_a_label}_foot_y": row_a[f"{args.policy_a_label}_foot_y"],
        f"{args.policy_a_label}_foot_z": row_a[f"{args.policy_a_label}_foot_z"],
        f"{args.policy_a_label}_slip_vel": row_a[f"{args.policy_a_label}_slip_vel"],
        f"{args.policy_a_label}_lin_tracking_error": row_a[
          f"{args.policy_a_label}_lin_tracking_error"
        ],
        f"{args.policy_b_label}_foot_x": row_b[f"{args.policy_b_label}_foot_x"],
        f"{args.policy_b_label}_foot_y": row_b[f"{args.policy_b_label}_foot_y"],
        f"{args.policy_b_label}_foot_z": row_b[f"{args.policy_b_label}_foot_z"],
        f"{args.policy_b_label}_slip_vel": row_b[f"{args.policy_b_label}_slip_vel"],
        f"{args.policy_b_label}_lin_tracking_error": row_b[
          f"{args.policy_b_label}_lin_tracking_error"
        ],
      }
      writer.writerow(merged)

  print(f"Saved comparison traces to: {args.output_csv}")


if __name__ == "__main__":
  normalized_args = []
  for arg in sys.argv[1:]:
    if arg.startswith("--"):
      name, *rest = arg[2:].split("=", 1)
      name = name.replace("_", "-")
      arg = "--" + name + ("=" + rest[0] if rest else "")
    normalized_args.append(arg)

  tyro.cli(main, args=normalized_args)