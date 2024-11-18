# deep_q_2048.py

import numpy as np
import random
from collections import deque
import time  # For timing purposes

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from game_env import Game2048Env  # Import the game environment

# Hyperparameters
MEMORY_SIZE = 50000
BATCH_SIZE = 16  # Reduced batch size for faster training
LEARNING_RATE = 5e-5
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_END = 0.01
EPSILON_DECAY = 0.9999
TARGET_UPDATE = 1000  # How often to update the target network

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        # Simplified network for faster computation
        self.conv1 = ConvBlock(input_channels, 32)
        self.conv2 = ConvBlock(32, 32)
        self.fc1 = nn.Linear(32 * 16, 128)
        self.fc2 = nn.Linear(128, num_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvBlock, self).__init__()
        d = output_channels // 2
        # Using smaller kernel sizes for efficiency
        self.conv1 = nn.Conv2d(input_channels, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels, d, kernel_size=5, padding=2)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        return torch.cat([out1, out2], dim=1)

# Optimized board_to_tensor function
def board_to_tensor(board):
    '''
    Convert the game board to a tensor representation suitable for the neural network.
    Each tile is represented as a one-hot vector.
    '''
    max_power = 17
    depth = max_power + 1

    board = board.copy()
    board[board == 0] = 1
    board_log2 = np.log2(board).astype(int)

    batch_size, height, width = board.shape
    tensor = np.zeros((batch_size, depth, height, width), dtype=np.float32)

    # Vectorized assignment
    batch_indices = np.arange(batch_size)[:, None, None]
    height_indices = np.arange(height)[None, :, None]
    width_indices = np.arange(width)[None, None, :]

    tensor[batch_indices, board_log2, height_indices, width_indices] = 1.0

    return torch.from_numpy(tensor).to(device)

# Main training loop
def train():
    env = Game2048Env()
    num_actions = env.action_space.n  # 8 actions
    memory = ReplayMemory(MEMORY_SIZE)

    policy_net = DQN(input_channels=18, num_actions=num_actions).to(device)
    target_net = DQN(input_channels=18, num_actions=num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    epsilon = EPSILON_START
    steps_done = 0

    num_episodes = 20000
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.expand_dims(state, axis=0)
        state_tensor = board_to_tensor(state)
        total_score = 0
        done = False
        step_count = 0

        episode_start_time = time.time()

        while not done:
            step_count += 1

            # Action Selection
            sample = random.random()
            if sample < epsilon:
                action = random.randrange(num_actions)
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = q_values.max(1)[1].item()

            # Environment Step
            next_state, reward, done, _, _ = env.step(action)

            # State Tensor Conversion
            next_state = np.expand_dims(next_state, axis=0)
            next_state_tensor = board_to_tensor(next_state)

            total_score += reward

            # Store Transition in Memory
            memory.push(state_tensor, action, next_state_tensor, reward, done)

            # Move to the Next State
            state_tensor = next_state_tensor

            # Perform Optimization
            if len(memory) > BATCH_SIZE:
                # Sample from Replay Memory
                transitions = memory.sample(BATCH_SIZE)
                batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*transitions)

                batch_state = torch.cat(batch_state)
                batch_action = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(1).to(device)
                batch_next_state = torch.cat(batch_next_state)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1).to(device)
                batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1).to(device)

                # Optimization Step
                with torch.cuda.amp.autocast():
                    # Compute Q(s_t, a)
                    q_values = policy_net(batch_state)
                    state_action_values = q_values.gather(1, batch_action)

                    # Compute V(s_{t+1}) for all next states
                    next_q_values = target_net(batch_next_state)
                    next_state_values = next_q_values.max(1)[0].unsqueeze(1)

                    # Compute Expected Q Values
                    expected_state_action_values = batch_reward + (1 - batch_done) * GAMMA * next_state_values

                    # Compute Loss
                    loss = F.mse_loss(state_action_values, expected_state_action_values)

                # Optimize the Model
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            steps_done += 1

            # Update Target Network
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Logging every 100 steps
            if step_count % 100 == 0:
                print(f"Episode {episode}, Step {step_count}, Total Score: {total_score}, Epsilon: {epsilon:.5f}")

        # Decay epsilon
        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY

        # Print progress after each episode
        duration = time.time() - episode_start_time
        print(f"Episode {episode} completed in {duration:.2f} seconds with total steps {step_count}. Total Score: {total_score}, Epsilon: {epsilon:.5f}")

        # Optionally, save the model every N episodes
        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), f'dqn_2048_episode_{episode}.pth')

    # Save the final trained model
    torch.save(policy_net.state_dict(), 'dqn_2048_final.pth')

if __name__ == '__main__':
    train()
