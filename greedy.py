import numpy as np
from game_env import Game2048Env
import time
import copy

def getGreedyAction(env):
    # Try all possible actions and see which one yields the highest score
    best_score = 0
    best_action = env.action_space.sample()
    for action in range(env.action_space.n):
        new_env = copy.deepcopy(env)
        # Perform the action and get the resulting board and score
        _, score, _, _, _ = new_env.step(action)
        # Keep track of the best action based on the score
        if score > best_score:
            best_score = score
            best_action = action
    return best_action    
        
def greedyAlgo(x):
    env = Game2048Env()
    results = []
    
    for _ in range(x):
        observation, info = env.reset()
        done = False
        total_score = 0

        # env.render()
        while not done:
            action = getGreedyAction(env)
            # print("chosen action: ", action)
            observation, score, done, truncated, info = env.step(action)
            # env.render()
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

start = time.time()
greedyAlgo(100)
end = time.time()
print("Time elapsed: ", end-start)