import numpy as np
from game_env import Game2048Env
import time
import matplotlib.pyplot as plt

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

def benchmarkX(x):
    env = Game2048Env()
    total_scores = []
    max_tiles = []

    start = time.time()

    for _ in range(x):
        observation, info = env.reset()
        done = False
        total_score = 0
        max_tile = 0

        while not done:
            action = env.action_space.sample()  # select a random action
            observation, score, done, truncated, info = env.step(action)
            total_score += score
            max_tile = max(max_tile, np.max(observation))
            if done:
                total_scores.append(total_score)
                max_tiles.append(max_tile)
                break

    end = time.time()

    print(f'Ran {x} games in {end - start:.2f} seconds.')
    print(f'Average total score: {np.mean(total_scores):.2f}')
    print(f'Average max tile: {np.mean(max_tiles):.2f}')

    # Plot total score distribution
    plt.figure(figsize=(10, 5))
    plt.hist(total_scores, bins=20, edgecolor='black')
    plt.title('Total Score Distribution')
    plt.xlabel('Total Score')
    plt.ylabel('Frequency')
    plt.savefig('total_score_distribution.png')
    plt.show()

    # Plot max tile distribution
    # Convert max tiles to log2 scale for better visualization
    log_max_tiles = [int(np.log2(tile)) if tile > 0 else 0 for tile in max_tiles]
    unique_tiles = sorted(set(log_max_tiles))
    tile_labels = [2**i for i in unique_tiles]

    plt.figure(figsize=(10, 5))
    plt.hist(log_max_tiles, bins=range(min(unique_tiles), max(unique_tiles)+2), edgecolor='black', align='left', rwidth=0.8)
    plt.xticks(unique_tiles, tile_labels)
    plt.title('Max Tile Distribution')
    plt.xlabel('Max Tile')
    plt.ylabel('Frequency')
    plt.savefig('max_tile_distribution.png')
    plt.show()

# Uncomment the function you want to run

# best_x_games(10000)
# step_by_step()
benchmarkX(1000)
