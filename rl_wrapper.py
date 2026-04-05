import gymnasium as gym
from gymnasium import spaces
import numpy as np
from adaptive_cache.env import AdaptiveCacheEnv, Action

class GymCacheEnv(gym.Env):
    """Wraps the OpenEnv into a strict Gymnasium standard for SB3."""
    def __init__(self, task_level="medium"):
        super().__init__()
        self.open_env = AdaptiveCacheEnv(task_level=task_level)
        self.capacity = self.open_env.capacity
        
        # Action Space: Which index to evict (0 to 9)
        self.action_space = spaces.Discrete(self.capacity)
        
        # Observation Space: 1D Array -> [incoming_request, cache_state(10), idle_times(10)]
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=10000.0, 
            shape=(1 + self.capacity * 2,), 
            dtype=np.float32
        )

    def _flatten_obs(self, obs):
        req = [obs.incoming_request]
        cache = obs.cache_state
        idle = obs.idle_times
        return np.array(req + cache + idle, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.open_env.reset()
        return self._flatten_obs(obs), {}

    def step(self, action):
        open_action = Action(evict_index=int(action))
        obs, reward, done, info = self.open_env.step(open_action)
        truncated = False # Gym requirement, we don't use it
        return self._flatten_obs(obs), reward, done, truncated, info