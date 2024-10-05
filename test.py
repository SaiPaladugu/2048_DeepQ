import gymnasium as gym
from gym_examples.envs.grid_world import GridWorldEnv

env = GridWorldEnv(size=5)
obs = env.reset()
print("Initial observation:", obs)