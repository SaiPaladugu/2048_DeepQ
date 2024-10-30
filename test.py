import numpy as np
from game_env import Game2048Env
import time

# environment demo file
# two options:
#   - play x games and show the best 3 final results
#   - show one game and all its steps
# 
# comment/uncomment the test functions below

def best_x_games(x):
    env = Game2048Env()
    results = []  # Will store tuples of (total_score, final_grid)

    for _ in range(x):
        observation, info = env.reset()
        done = False
        total_score = 0

        while not done:
            action = env.action_space.sample()  # select a random action
            observation, score, done, truncated, info = env.step(action)
            total_score += score
            if done:
                final_grid = observation.copy()  # Copy the final grid
                results.append((total_score, final_grid))
                break

    # Sort the results by total_score in descending order
    results.sort(key=lambda x: x[0], reverse=True)

    # Get the top 3 scores and their grids
    top_results = results[:3]

    print(f"Top 3 scores after {len(results)} games:")
    for i, (score, grid) in enumerate(top_results, start=1):
        print(f"\nRank {i}: Score = {score}")
        print("Final Grid:")
        print(grid)

def step_by_step():
    env = Game2048Env()

    observation, info = env.reset()
    done = False
    total_score = 0

    while not done:
        print(f'score: {total_score}')
        env.render()
        print()
        action = env.action_space.sample()  # select a random action
        observation, score, done, truncated, info = env.step(action)
        total_score += score
        if done:
            print(f'final score: {total_score}')
            env.render()
            break

# best_x_games(100)
# step_by_step()