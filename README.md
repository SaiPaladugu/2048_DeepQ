# Deep Q-Learning Agent for 2048

![GitHub](https://img.shields.io/github/license/SaiPaladugu/deep_q_2048)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10.0-brightgreen)

Current highest achieving score grid <br>
<img width="269" alt="Screenshot 2024-11-26 at 12 55 30 PM" src="https://github.com/user-attachments/assets/ebd5b6f8-70c5-4384-a309-7aa17fb4f5fc">


## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Agent](#training-the-agent)
  - [Evaluating the Agent](#evaluating-the-agent)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

Welcome to the **Deep Q-Learning Agent for 2048** project! This repository contains a reinforcement learning (RL) agent designed to master the popular puzzle game [2048](https://play2048.co/) using Deep Q-Learning (DQN). The agent autonomously learns and optimizes strategies to achieve high scores, demonstrating significant improvements over baseline performance.

**Key Achievements:**
- **Control Phase:** Achieved a control score of approximately **6,500** with the agent reaching the **2048 tile** once every **1,000 games**.
- **Optimized Phase:** Enhanced performance to reach scores up to **15,000**, with the agent obtaining the **2048 tile** in **70%** of the games.

This project showcases the application of deep reinforcement learning techniques in game AI, highlighting the potential for autonomous agents to learn and excel in complex environments.

## Features

- **Deep Q-Network (DQN):** Utilizes a convolutional neural network to approximate Q-values for state-action pairs.
- **Experience Replay:** Implements a replay memory to store and sample past experiences, improving learning stability.
- **Epsilon-Greedy Strategy:** Balances exploration and exploitation during training, with dynamic epsilon decay.
- **Target Network:** Incorporates a separate target network to stabilize training by reducing correlations between target and prediction.
- **One-Hot Encoding:** Converts the game board into a tensor representation suitable for neural network input.
- **Performance Logging:** Tracks and logs performance metrics, including scores and training progress.

## Technologies Used

- **Python 3.8+**
- **PyTorch:** For building and training the neural network.
- **NumPy:** For numerical operations and data manipulation.
- **OpenAI Gym:** Custom environment for the 2048 game.
- **CUDA:** Optional GPU acceleration for faster training.

## Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

- **Python 3.8 or higher**
- **Git**
- **CUDA-enabled GPU (optional but recommended for faster training)**

### Clone the Repository

```bash
git clone https://github.com/SaiPaladugu/deep_q_2048.git
cd deep_q_2048
```

### Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

*If `requirements.txt` is not provided, install the necessary packages manually:*

```bash
pip install numpy torch
```

### Verify GPU Availability (Optional)

To utilize GPU acceleration, ensure that PyTorch is installed with CUDA support.

```python
import torch
print(torch.cuda.is_available())
```

If `True`, CUDA is available. If `False`, the training will proceed on the CPU.

## Usage

### Training the Agent

To start training the Deep Q-Learning agent, execute the `deep_q_2048.py` script.

```bash
python deep_q_2048.py
```

**Training Parameters:**

- **Memory Size:** 50,000 experiences
- **Batch Size:** 64
- **Learning Rate:** 5e-5
- **Discount Factor (Gamma):** 0.99
- **Epsilon Start:** 0.9
- **Epsilon End:** 0.01
- **Epsilon Decay:** 0.9999
- **Target Network Update Frequency:** Every 1,000 steps
- **Number of Episodes:** 20,000
- **Image Size:** 640 (if applicable)

*These parameters can be adjusted within the script to experiment with different training configurations.*

### Evaluating the Agent

After training, the model parameters are saved as `dqn_2048.pth`. You can load this model to evaluate the agent's performance.

```python
import torch
from deep_q_2048 import DQN, board_to_tensor, Game2048Env

# Load the trained model
model = DQN(input_channels=18, num_actions=8)
model.load_state_dict(torch.load('dqn_2048.pth'))
model.eval()

# Initialize the environment
env = Game2048Env()

# Example evaluation run
state, _ = env.reset()
state_tensor = board_to_tensor(state)
done = False
total_score = 0

while not done:
    with torch.no_grad():
        q_values = model(state_tensor.to(device))
        action = q_values.max(1)[1].item()
    next_state, reward, done, _, _ = env.step(action)
    state_tensor = board_to_tensor(next_state)
    total_score += reward

print(f"Total Score: {total_score}")
```

*Modify the evaluation script as needed to run multiple episodes and gather performance metrics.*

## Results

### Training Performance

- **Control Phase:**
  - **Average Score:** ~6,500
  - **2048 Tile Achievement:** 1 per 1,000 games

- **Optimized Phase:**
  - **Average Score:** ~15,000
  - **2048 Tile Achievement:** 70% of games

### Performance Metrics

- **Mean Average Precision (mAP):** Evaluated to assess the model's accuracy in selecting optimal moves.
- **Training Duration:** Varies based on hardware, with GPU acceleration significantly reducing training time.
- **Loss Trends:** Monitored using Mean Squared Error (MSE) loss between predicted Q-values and target Q-values.

### Visualizations

*Include graphs or charts showcasing training progress, score improvements, and loss reduction over episodes.*

## Project Structure

```
deep_q_2048/
├── deep_q_2048.py         # Main training script
├── game_env.py            # Custom 2048 game environment
├── dqn_2048.pth           # Trained model parameters (after training)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── LICENSE                # License information
```

- **deep_q_2048.py:** Contains the implementation of the DQN agent, training loop, and hyperparameters.
- **game_env.py:** Defines the 2048 game environment compatible with OpenAI Gym's interface.
- **dqn_2048.pth:** Saved state dictionary of the trained model.
- **requirements.txt:** Lists all Python dependencies required to run the project.
- **LICENSE:** Specifies the licensing terms for the project.

## Contributing

Contributions are welcome! If you have suggestions for improvements or encounter any issues, feel free to open an [issue](https://github.com/SaiPaladugu/deep_q_2048/issues) or submit a pull request.

### How to Contribute

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -m "Add some feature"
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this code as per the license terms.

## Acknowledgments

- **[PyTorch](https://pytorch.org/):** An open-source machine learning library used for developing and training the neural network.
- **[OpenAI Gym](https://gym.openai.com/):** Provides the framework for the custom 2048 game environment.
- **[NumPy](https://numpy.org/):** Utilized for efficient numerical operations and data manipulation.
- **Inspiration:** The 2048 game serves as a challenging environment for reinforcement learning applications, inspiring the development of this project.

---

Feel free to explore the repository, train your own agent, and contribute to advancing the capabilities of reinforcement learning in game AI!
