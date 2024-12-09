# deepQ.py

from game_env import Game2048Env

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

cudnn.benchmark = True  # Optimize computation for your hardware

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIZE = 4  # Board size
GOAL = 2 ** 17  # Goal tile value (131072)
NUM_CLASSES = int(math.log2(GOAL)) + 1  # Number of tile classes (including zero)

def encode_state(board):
    board_flat = [0 if e == 0 else int(math.log2(e)) for e in board.flatten()]
    board_flat = torch.LongTensor(board_flat).to(device)
    board_flat = F.one_hot(board_flat, num_classes=NUM_CLASSES).float()
    board_flat = board_flat.view(1, SIZE, SIZE, NUM_CLASSES).permute(0, 3, 1, 2)
    return board_flat

# Define the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # Save a transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # Randomly sample a batch of transitions
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        d = output_dim // 4
        self.conv1 = nn.Conv2d(input_dim, d, kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(input_dim, d, kernel_size=2, padding='same')
        self.conv3 = nn.Conv2d(input_dim, d, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(input_dim, d, kernel_size=4, padding='same')

    def forward(self, x):
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        output3 = self.conv3(x)
        output4 = self.conv4(x)
        return torch.cat((output1, output2, output3, output4), dim=1)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = ConvBlock(NUM_CLASSES, 512)
        self.conv2 = ConvBlock(512, 512)
        self.conv3 = ConvBlock(512, 512)
        self.dense1 = nn.Linear(512 * SIZE * SIZE, 256)
        self.dense2 = nn.Linear(256, 8)  # 8 possible actions

    def forward(self, x):
        x = F.relu(self.conv1(x.to(device)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.dropout(F.relu(self.dense1(x)), training=self.training)
        return self.dense2(x)

# Neural Network Initialization and utilities
BATCH_SIZE = 64  # Reduced batch size
GAMMA = 0.98
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2000  # Faster decay for exploration
TARGET_UPDATE = 10
n_actions = 8

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net.train()

# Load saved model weights if available
try:
    policy_net.load_state_dict(torch.load('policy_net.pth'))
    target_net.load_state_dict(torch.load('target_net.pth'))
    print("Loaded saved model weights.")
except FileNotFoundError:
    print("No saved model weights found. Starting from scratch.")

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
memory = ReplayMemory(100000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    # Adjusted epsilon decay for faster convergence
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # Exploit learned policy
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # Explore action space
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Mask of non-final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None], dim=0)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # V(s_{t+1})
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if len(non_final_next_states) > 0:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Loss calculation
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Clipping gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# env = Game2048Env(size=SIZE, goal=GOAL)
# total_scores, max_tiles = [], []

# num_episodes = 10000
# for i_episode in range(num_episodes):
#     print(f"Episode {i_episode}")
#     state_np, info = env.reset()
#     state = encode_state(state_np).float()
#     non_valid_count, valid_count = 0, 0
#     episode_score = 0
#     state_np = state_np  # Keep track of the numpy state

#     for t in count():
#         # Select and perform an action
#         action = select_action(state)
#         action_value = action.item()

#         # Save old max tile for reward calculation
#         old_max_tile = state_np.max()

#         # Perform the action in the environment
#         observation, reward, done, _, info = env.step(action_value)
#         reward_tensor = torch.tensor([reward], device=device).float()
#         episode_score += reward

#         # Adjust the reward function to encourage higher tiles
#         new_max_tile = observation.max()
#         if new_max_tile > old_max_tile:
#             # Reward for increasing the max tile
#             reward_tensor += (math.log2(new_max_tile) - math.log2(old_max_tile)) * 10

#         # Penalize invalid moves
#         if not info['moved']:
#             non_valid_count += 1
#             reward_tensor -= 10
#         else:
#             valid_count += 1

#         # Encode the new state
#         if not done:
#             next_state = encode_state(observation).float()
#         else:
#             next_state = None

#         # Store the transition in memory
#         memory.push(state, action, next_state, reward_tensor)

#         # Move to the next state
#         state = next_state
#         state_np = observation

#         # Optimize the model at every step
#         optimize_model()

#         if done:
#             total_scores.append(episode_score)
#             max_tile = env.board.max()
#             max_tiles.append(max_tile)
#             if i_episode % 50 == 0 and i_episode > 0:
#                 average_score = sum(total_scores[-50:]) / 50
#                 average_tile = sum(max_tiles[-50:]) / 50
#                 print(f"Average score over last 50 episodes: {average_score}")
#                 print(f"Average max tile over last 50 episodes: {average_tile}")
#             break

#     # Update the target network
#     if i_episode % TARGET_UPDATE == 0:
#         target_net.load_state_dict(policy_net.state_dict())
#         policy_net.train()

#     # Save the model
#     if i_episode % 100 == 0:
#         torch.save(policy_net.state_dict(), 'policy_net.pth')
#         torch.save(target_net.state_dict(), 'target_net.pth')

# print('Complete')

def benchmark_results(total_scores, max_tiles):
    '''
    Generates histograms and outputs average total score and max tile.
    '''
    import matplotlib.pyplot as plt

    print(f'Ran {len(total_scores)} games.')
    print(f'Average total score: {np.mean(total_scores):.2f}')
    print(f'Average max tile: {np.mean(max_tiles):.2f}')
    print(f'Highest score: {max(total_scores)}')
    print(f'Highest tile: {max(max_tiles)}')
    print(f'Winning games (2048): {len([1 for i in max_tiles if i >= 2048])}')

    # Plot total score distribution
    plt.figure(figsize=(10, 5))
    plt.hist(total_scores, bins=20, edgecolor='black')
    plt.title('Total Score Distribution')
    plt.xlabel('Total Score')
    plt.ylabel('Frequency')
    plt.savefig('total_score_distribution_dqn.png')
    plt.show()

    # Plot max tile distribution
    log_max_tiles = [int(np.log2(tile)) if tile > 0 else 0 for tile in max_tiles]
    unique_tiles = sorted(set(log_max_tiles))
    tile_labels = [2**i for i in unique_tiles]

    plt.figure(figsize=(10, 5))
    plt.hist(log_max_tiles, bins=range(min(unique_tiles), max(unique_tiles)+2),
             edgecolor='black', align='left', rwidth=0.8)
    plt.xticks(unique_tiles, tile_labels)
    plt.title('Max Tile Distribution')
    plt.xlabel('Max Tile')
    plt.ylabel('Frequency')
    plt.savefig('max_tile_distribution_dqn.png')
    plt.show()

def sample_game_DQN():
    policy_net.eval()
    game = Game2048Env(size=SIZE, goal=GOAL)
    state_np, _ = game.reset()
    state = encode_state(state_np).float()
    done = False
    total_score = 0

    while not done:
        with torch.no_grad():
            q_values = policy_net(state)
        q_values_np = q_values.cpu().numpy()[0]
        sorted_actions = np.argsort(q_values_np)[::-1]

        state_backup = game.get_state()
        moved = False
        for action in sorted_actions:
            next_state_np, reward, done, _, info = game.step(action)
            if info['moved']:
                moved = True
                total_score += reward
                state = encode_state(next_state_np).float()
                break
            else:
                game.set_state(state_backup)  # Restore state if action is invalid

        if not moved:
            done = True

    best_tile = game.board.max()
    return total_score, best_tile

# Testing the trained agent
scores_DQN, best_tiles_DQN = [], []
for i in range(1000):
    if i % 1 == 0:
        print(f"Iteration {i}")
    total_score, best_tile = sample_game_DQN()
    scores_DQN.append(total_score)
    best_tiles_DQN.append(best_tile)
print("Finish")
benchmark_results(scores_DQN, best_tiles_DQN)
