import numpy as np
from game_code import Game2048Env

env = Game2048Env()
observation, info = env.reset()
done = False
total_score = 0

while not done:
    action = env.action_space.sample()  # select a random action
    observation, reward, done, truncated, info = env.step(action)
    total_score += reward
    env.render()
    print(f"Action taken: {['Up', 'Down', 'Left', 'Right'][action]}")
    print(f"Reward obtained: {reward}")
    print(f"Total score: {total_score}\n")
    if done:
        print("Game over!")
        break
