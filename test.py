import numpy as np
from game_code import Game2048Env

env = Game2048Env()
observation, info = env.reset()
env.render()
print('Initial state\n')
done = False
total_score = 0
i = 0
while not done:
    i += 1
    action = env.action_space.sample()  # select a random action
    observation, reward, done, truncated, info = env.step(action)
    total_score += reward
    env.render()
    print(f"Action taken: {['Up', 'Down', 'Left', 'Right', 'Up Left', 'Up Right', 'Down Left', 'Down Right'][action]}")
    print(f"Reward obtained: {reward}")
    print(f"Total score: {total_score}\n")
    if done:
        print("Game over!")
        break
