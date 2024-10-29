import numpy as np
from game_code import Game2048Env
import time

# env = Game2048Env()
# observation, info = env.reset()
# done = False
# total_score = 0

# while not done:
#     action = env.action_space.sample()  # select a random action
#     observation, reward, done, truncated, info = env.step(action)
#     total_score += reward
#     env.render()
#     print(f"Action taken: {['Up', 'Down', 'Left', 'Right'][action]}")
#     print(f"Reward obtained: {reward}")
#     print(f"Total score: {total_score}\n")
#     if done:
#         print("Game over!")
#         break

def playX():
    env = Game2048Env()
    results = []  # Will store tuples of (total_score, final_grid)

    for _ in range(1):
        print('init')
        observation, info = env.reset()
        env.render()
        done = False
        total_score = 0
        step = 0

        while not done:
            step += 1
            if step == 10: exit()
            action = env.action_space.sample()  # select a random action
            observation, score, done, truncated, info, reward = env.step(action)
            print(f'RLscore: {reward}')
            env.render()
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

def play_continuous():
    env = Game2048Env()
    
    max_tile = 0
    max_tile_board = None
    max_score = 0
    max_score_board = None
    
    start_time = time.time()
    games_played = 0

    while True:
        observation, _ = env.reset()
        done = False
        total_score = 0
        
        while not done:
            action = env.action_space.sample()  # select a random action
            observation, reward, done, truncated, _ = env.step(action)
            total_score += reward

        # After game ends, check for max tile and max score
        current_max_tile = np.max(observation)
        if current_max_tile > max_tile:
            max_tile = current_max_tile
            max_tile_board = observation.copy()

        if total_score > max_score:
            max_score = total_score
            max_score_board = observation.copy()

        games_played += 1

        # Every 100,000 games, report stats
        if games_played % 100000 == 0:
            elapsed_time = time.time() - start_time
            print(f"\n--- Report after {games_played} games ---")
            print(f"Max Tile: {max_tile}")
            print("Board with Max Tile:")
            print(max_tile_board)
            print(f"Max Score: {max_score}")
            print("Board with Max Score:")
            print(max_score_board)
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")
            print(f"Games per second: {games_played / elapsed_time:.2f}")

# play_continuous()    
playX()