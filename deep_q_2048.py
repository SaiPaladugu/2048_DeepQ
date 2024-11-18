# deep_q_2048.py

import numpy as np
import random
from collections import deque
import time  # For logging purposes

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from game_env import Game2048Env  # Import the game environment

# Hyperparameters
MEMORY_SIZE = 50000
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_END = 0.01
EPSILON_DECAY = 0.9999
TARGET_UPDATE = 1000  # How often to update the target network

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = ConvBlock(input_channels, 128)
        self.conv2 = ConvBlock(128, 128)
        self.conv3 = ConvBlock(128, 128)
        self.fc1 = nn.Linear(128 * 16, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvBlock, self).__init__()
        d = output_channels // 4
        # Using odd kernel sizes and appropriate padding
        self.conv1 = nn.Conv2d(input_channels, d, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(input_channels, d, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels, d, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(input_channels, d, kernel_size=7, padding=3)
    
    def forward(self, x):
        x = x.to(device)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)

# Function to convert the game board to a tensor
def board_to_tensor(board):
    '''
    Convert the game board to a tensor representation suitable for the neural network.

    Each tile is represented as a one-hot vector.
    '''
    # The maximum possible value of a tile, to determine the depth of one-hot encoding
    max_power = 17  # Adjust this based on the maximum tile value possible
    # Since tiles can be from 0 (empty) to 2^17
    depth = max_power + 1  # from 0 to 2^17

    # Use logarithms to compute the powers efficiently
    board = board.copy()
    board[board == 0] = 1  # Avoid log2(0)
    board_log2 = np.log2(board).astype(int)

    tensor = np.zeros((board.shape[0], depth, board.shape[1], board.shape[2]), dtype=np.float32)
    for i in range(board.shape[0]):
        for y in range(board.shape[1]):
            for x in range(board.shape[2]):
                power = board_log2[i, y, x]
                tensor[i, power, y, x] = 1.0
    return torch.from_numpy(tensor)

# Main training loop
def train():
    env = Game2048Env()
    num_actions = env.action_space.n  # 8 actions
    memory = ReplayMemory(MEMORY_SIZE)

    policy_net = DQN(input_channels=18, num_actions=num_actions).to(device)  # 18 channels for one-hot encoding
    target_net = DQN(input_channels=18, num_actions=num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    epsilon = EPSILON_START
    steps_done = 0

    num_episodes = 20000
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.expand_dims(state, axis=0)
        state_tensor = board_to_tensor(state)
        total_score = 0
        done = False
        step_count = 0  # For logging steps per episode

        start_time = time.time()  # Timing each episode

        while not done:
            step_count += 1
            # Select action
            sample = random.random()
            if sample < epsilon:
                # Random action (exploration)
                action = random.randrange(num_actions)
            else:
                # Greedy action (exploitation)
                with torch.no_grad():
                    q_values = policy_net(state_tensor.to(device))
                    action = q_values.max(1)[1].item()

            # Take action
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            next_state_tensor = board_to_tensor(next_state)

            total_score += reward

            # Store transition in memory
            memory.push(state_tensor, action, next_state_tensor, reward, done)

            # Move to the next state
            state_tensor = next_state_tensor

            # Perform optimization
            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*transitions)

                batch_state = torch.cat(batch_state).to(device)
                batch_action = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(1).to(device)
                batch_next_state = torch.cat(batch_next_state).to(device)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1).to(device)
                batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1).to(device)

                # Compute Q(s_t, a)
                state_action_values = policy_net(batch_state).gather(1, batch_action)

                # Compute V(s_{t+1}) for all next states.
                with torch.no_grad():
                    next_state_values = target_net(batch_next_state).max(1)[0].unsqueeze(1)

                # Compute expected Q values
                expected_state_action_values = batch_reward + (1 - batch_done) * GAMMA * next_state_values

                # Compute loss
                loss = F.mse_loss(state_action_values, expected_state_action_values)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            steps_done += 1

            # Update target network
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Logging every 100 steps
            if step_count % 100 == 0:
                print(f"Episode {episode}, Step {step_count}, Total Score: {total_score}, Epsilon: {epsilon:.5f}")

        # Decay epsilon
        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY

        # Print progress after each episode
        duration = time.time() - start_time
        print(f"Episode {episode} completed in {duration:.2f} seconds with total steps {step_count}. Total Score: {total_score}, Epsilon: {epsilon:.5f}")

    # Save the trained model
    torch.save(policy_net.state_dict(), 'dqn_2048.pth')

if __name__ == '__main__':
    train()
