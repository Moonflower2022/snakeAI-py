from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np

class SnakeEnv(Env):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.action_space = Discrete(3) # turn counter-clockwise, do nothing, turn clockwise
        self.grid = np.zeros((width, height), dtype=np.int32)
        self.observation_space = Box(low=0, high=3, shape=(width, height), dtype=np.int32)
        pass
    def step(self, action):
        # step(action) -> ObseravtionType, Float, Bool, Bool

        return observation, reward, terminated, truncated
    def reset(self):
        pass
    def render(self):
        pass
    def close(self):
        pass