from game_env import Game2048Env
import time

def greedyAlgo(x):
    '''
    Run a greedy algorithm for the 2048 game, selecting actions based on the highest simulated score.

    Parameters:
    - x (int): Number of episodes to run.

    Prints the top 3 scores and their final grids after all episodes are completed.
    '''
    env = Game2048Env()
    results = []
    episode_count = 0

    for _ in range(x):
        episode_count += 1
        print(f"Completing Episode {episode_count}/{x} episodes")
        observation, info = env.reset()
        done = False
        total_score = 0

        while not done:
            # Simulate all possible actions
            action_scores = env.simulateActions(env.board)
            # Filter out actions that do not result in any move
            valid_actions = {action: score for action, score in action_scores.items() if score > 0}
            if valid_actions:
                # Select the action with the maximum expected score
                action = max(valid_actions, key=valid_actions.get)
            else:
                # If no valid actions with positive score, pick any possible move
                action = env.action_space.sample()
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

    print(f"\nTop 3 scores after {len(results)} games:")
    for i, (score, grid) in enumerate(top_results, start=1):
        print(f"\nRank {i}: Score = {score}")
        print("Final Grid:")
        print(grid)

start = time.time()
greedyAlgo(10000)
end = time.time()
print("Time elapsed: ", end-start)