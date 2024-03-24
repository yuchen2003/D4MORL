from gym.envs.registration import register
from .ant import AntEnv
from .half_cheetah import HalfCheetahEnv
from .hopper import HopperEnv
from .hopper_v3 import HopperEnv as HopperEnv_v3
from .swimmer import SwimmerEnv
from .walker2d import Walker2dEnv
from .humanoid import HumanoidEnv

# Wrap with d4rl Offline Env Wrapper
from d4rl.offline_env import OfflineEnv

def env_wrapper(env):
    class MO_OfflineEnv(env, OfflineEnv):
        def __init__(self, **kwargs):
            env.__init__(self,)
            OfflineEnv.__init__(self, **kwargs)
            
    return MO_OfflineEnv

register(
    id="MO-Ant-v2",
    entry_point=env_wrapper(AntEnv),
    max_episode_steps=500,
)

register(
    id = 'MO-Hopper-v2',
    entry_point = env_wrapper(HopperEnv),
    max_episode_steps=500,
)

register(
    id = 'MO-Hopper-v3',
    entry_point = env_wrapper(HopperEnv_v3),
    max_episode_steps=500,
)

register(
    id = 'MO-HalfCheetah-v2',
    entry_point = env_wrapper(HalfCheetahEnv),
    max_episode_steps=500,
)

register(
    id = 'MO-Walker2d-v2',
    entry_point = env_wrapper(Walker2dEnv),
    max_episode_steps=500,
)

register(
    id = 'MO-Swimmer-v2',
    entry_point = env_wrapper(SwimmerEnv),
    max_episode_steps=500,
)

register(
    id = 'MO-Humanoid-v2',
    entry_point = env_wrapper(HumanoidEnv),
    max_episode_steps=1000,
)