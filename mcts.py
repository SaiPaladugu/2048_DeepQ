import numpy as np
from copy import deepcopy
from game_env import Game2048Env
import time
from threading import Thread, Lock
import random
from matplotlib import pyplot as plt

def benchmarkX(total_score_list, max_tile_list, start, end):

  print(f'Ran {len(total_score_list)} games in {end - start} seconds')
  print(f'Average total score: {np.mean(total_score_list):.2f}')
  print(f'Average max tile: {np.mean(max_tile_list):.2f}')

  # Plot total score distribution
  plt.figure(figsize=(10, 5))
  plt.hist(total_score_list, bins=20, edgecolor='black')
  plt.title('Total Score Distribution')
  plt.xlabel('Total Score')
  plt.ylabel('Frequency')
  plt.savefig('total_score_distribution.png')
  plt.show()

  # Plot max tile distribution
  # Convert max tiles to log2 scale for better visualization
  log_max_tiles = [int(np.log2(tile)) if tile > 0 else 0 for tile in max_tile_list]
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

value_sums_lock = Lock()
board_lock = Lock()
total_score_list_lock = Lock()
max_tile_list_lock = Lock()

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

def mcts_game_thread(thread_number, total_score_list, max_tile_list, total_score_list_lock=None, max_tile_list_lock=None):
  env = Game2048Env()
  observation, info = env.reset()
  done = False
  total_score = 0

  while not done:
    action = mcts_best_move(env, 1, 4)
    observation, score, done, truncated, info = env.step(action)
    print(f'Thread {thread_number} took action {action}')
    total_score += score

    if done:
      with total_score_list_lock:
        total_score_list.append(total_score)

      with max_tile_list_lock:
        max_tile_list.append(env.max_tile())
      break

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

def best_x_games_threaded(x):
  threads = [None for _ in range(x)]
  total_score_list = []
  max_tile_list = []
  
  start = time.time()

  for i in range(len(threads)):
    threads[i] = Thread(target=mcts_game_thread, args=(i, total_score_list, max_tile_list, total_score_list_lock, max_tile_list_lock))
    threads[i].start()
    
  for i in range(len(threads)):
    threads[i].join()
    
  end = time.time()

  benchmarkX(total_score_list, max_tile_list, start, end)

best_x_games_threaded(100)


  