from game_env import Game2048Env
import time
import matplotlib.pyplot as plt
import numpy as np

def getGreedyAction(env):
    # Simulate all possible actions
    action_scores = env.simulateActions()
    # Filter out actions that do not result in any move
    valid_actions = {action: score for action, score in action_scores.items() if score > 0}
    if valid_actions:
        # Select the action with the maximum expected score
        action = max(valid_actions, key=valid_actions.get)
    else:
        # If no valid actions with positive score, pick any possible move
        action = env.action_space.sample()
    return action      
  
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
            action = getGreedyAction(env)
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

benchmarkX(1000)