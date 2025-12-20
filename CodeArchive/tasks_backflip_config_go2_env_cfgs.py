"""Unitree Go2 backflip environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.backflip.backflip_env_cfg import create_backflip_env_cfg
from mjlab.utils.retval import retval


@retval
def UNITREE_GO2_BACKFLIP_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 backflip task configuration."""
  return create_backflip_env_cfg()