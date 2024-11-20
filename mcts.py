import numpy as np
from copy import deepcopy
from game_env import Game2048Env
import time
from threading import Thread

def mcts_best_move(env:Game2048Env, depth=5, simulation_count=10):
  threads = [[None] * simulation_count for _ in range(env.action_space.n)]
  value_sums = np.zeros(env.action_space.n)

  for action in range(env.action_space.n):
    print("Simulating:", action)
    simulation_env:Game2048Env = deepcopy(env)

    _, _, _, _, info = simulation_env.step(action)

    if not info['moved']:
      continue

    for i in range(simulation_count):
      print(f'Starting action {action} simulation {i}')
      threads[action][i] = Thread(target=random_game_thread, args=(simulation_env, value_sums, action, depth))
      threads[action][i].start()

  for action in range(env.action_space.n):
    for i in range(simulation_count):
      print(f'Joining action {action} simulation {i}')
      if threads[action][i]: threads[action][i].join()

  print(value_sums)
  
  best_move = np.argmax(value_sums)
  return best_move

def random_game_thread(env, value_sums, action, depth=5):
  simulation_env:Game2048Env = deepcopy(env)

  done = False
  simulated_moves = 0

  while not done and simulated_moves < depth:
    action = simulation_env.action_space.sample()
    _, _, done, _, _ = simulation_env.step(action)

  merge_score, max_tile, tile_sum = simulation_env.merge_score, simulation_env.max_tile(), simulation_env.tile_sum()
  del simulation_env

  value_sums[action] += merge_score

def random_game(env, depth=5):
  simulation_env:Game2048Env = deepcopy(env)

  done = False
  simulated_moves = 0

  while not done and simulated_moves < depth:
    action = simulation_env.action_space.sample()
    _, _, done, _, _ = simulation_env.step(action)

  merge_score, max_tile, tile_sum = simulation_env.merge_score, simulation_env.max_tile(), simulation_env.tile_sum()
  del simulation_env

  return merge_score, max_tile, tile_sum

def best_x_games(x):
    env = Game2048Env()
    results = []  # Will store tuples of (total_score, final_grid)

    for _ in range(x):
        observation, info = env.reset()
        done = False
        total_score = 0

        while not done:
            action = mcts_best_move(env, 5, 10)
            observation, score, done, truncated, info = env.step(action)
            total_score += score
            if done:
                final_grid = observation.copy()  # Copy the final grid
                results.append((total_score, final_grid))
                break

    # Sort the results by total_score in descending order
    results.sort(key=lambda x: x[0], reverse=True)

    # Get the top 3 scores and their grids
    top_results = results[:1]

    print(f"Top {len(results)} scores after {len(results)} games:")
    for i, (score, grid) in enumerate(top_results, start=1):
        print(f"\nRank {i}: Score = {score}")
        print("Final Grid:")
        print(grid)

start = time.time()
best_x_games(1)
end = time.time()

print("Time elapsed:", end - start)

  