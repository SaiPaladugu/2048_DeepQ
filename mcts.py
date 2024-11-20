import numpy as np
from copy import deepcopy
from game_env import Game2048Env
import time
from threading import Thread, Lock
import random

value_sums_lock = Lock()
board_lock = Lock()

def mcts_best_move(env:Game2048Env, depth=5, simulation_count=10):
  threads = [[None] * simulation_count for _ in range(env.action_space.n)]
  value_sums = np.zeros(env.action_space.n)

  for action in range(env.action_space.n):
    #print("Simulating:", action)

    simulation_env:Game2048Env = Game2048Env(env.board)
    _, reward, _, _, info = simulation_env.step(action)

    if not info['moved']:
      continue

    value_sums[action] += reward

    for i in range(simulation_count):
      #print(f'Starting action {action} simulation {i}')
      threads[action][i] = Thread(target=random_game_thread, args=(simulation_env, value_sums, action, depth, value_sums_lock, board_lock))
      threads[action][i].start()
    

  for action in range(env.action_space.n):
    for i in range(simulation_count):
      #print(f'Joining action {action} simulation {i}')
      if threads[action][i]: threads[action][i].join()

  #print(value_sums)
  
  best_move = np.argmax(value_sums)
  if value_sums.all() == 0: return random.sample(range(env.action_space.n), 1)[0]
  return best_move

def random_game_thread(env, value_sums, action, depth=5, value_sums_lock=None, board_lock=None):
  simulation_env:Game2048Env = Game2048Env(env.board)

  done = False
  simulated_moves = 0
  simulated_score = 0

  while not done and simulated_moves < depth:
    action_to_take = simulation_env.action_space.sample()
    _, reward, done, _, _ = simulation_env.step(action_to_take)
    simulated_score += reward

  merge_score, max_tile, tile_sum = simulated_score, simulation_env.max_tile(), simulation_env.tile_sum()
  del simulation_env

  with value_sums_lock:
    value_sums[action] += merge_score

def best_x_games(x):
    env = Game2048Env()
    results = []  # Will store tuples of (total_score, final_grid)
    for _ in range(x):
        print(f"Game {_} started")
        observation, info = env.reset()
        env.render()
        done = False
        total_score = 0

        while not done:
            action = mcts_best_move(env, 2, 4)
            print("Action:", action)
            observation, score, done, truncated, info = env.step(action)
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

    print(f"Top {len(results)} scores after {len(results)} games:")
    for i, (score, grid) in enumerate(top_results, start=1):
        print(f"\nRank {i}: Score = {score}")
        print("Final Grid:")
        print(grid)

start = time.time()
best_x_games(100)
end = time.time()

print("Time elapsed:", end - start)

  