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
    start = time.time()
    env = Game2048Env()
    results = []  # Will store tuples of (total_score, final_grid)
    avgScore = 0

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
                avgScore += total_score
                break

    # Sort the results by total_score in descending order
    results.sort(key=lambda x: x[0], reverse=True)

    # Get the top 3 scores and their grids
    top_results = results[:3]
    
    end = time.time()

    print(f'average score: {avgScore/x}')
    print(f'Time taken: {end - start}')
    print(f"Top 3 scores after {len(results)} games:")
    for i, (score, grid) in enumerate(top_results, start=1):
        print(f"\nRank {i}: Score = {score}")
        print("Final Grid:")
        print(grid)

def step_by_step():
    
    # action taken is from the previous grid, and the grid below is its transformation
    
    env = Game2048Env()
    directions = ['up ↑', 'down ↓', 'left ←', 'right →', 'up-left ↖', 'up-right ↗', 'down-left ↙', 'down-right ↘']

    observation, info = env.reset()
    done = False
    total_score = 0
    print("Initial state")
    env.render()
    print()

    while not done:
        action = env.action_space.sample()  # select a random action
        print(f'action: {action}')
        observation, score, done, _, info = env.step(action)
        total_score += score
        
        print(f'Action taken: {directions[action]}')
        print(f'Score: {total_score}')
        env.render()
        print()

        if done:
            print("Game Over!\n")
            break


best_x_games(100)
# step_by_step()