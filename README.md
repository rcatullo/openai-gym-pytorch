# openai-gym-pytorch

Implementation of PyTorch's DQN algorithm which uses OpenAI's CartPole gym environment, and extension to curiosity-driven exploration to solve continuous mountaincart (usually solved by DDPG).

## Overview

This repository provides a PyTorch implementation of the Deep Q-Network (DQN) algorithm, a popular reinforcement learning method for solving control tasks in environments provided by OpenAI Gymnasium. The codebase includes:

- A standard DQN agent for the CartPole-v1 environment.
- Extensions for curiosity-driven exploration and forward dynamics modeling in the MountainCarContinuous-v0 environment.
- Utilities for generating transition data and training forward dynamics models.

## Features

- **Replay Memory**: Experience replay buffer for stable training.
- **Target Network**: Soft updates for improved stability.
- **Curiosity Module**: Intrinsic reward via prediction error (for MountainCarContinuous).
- **Video Recording**: Automatic recording of agent performance.
- **Modular Code**: Easily extensible for other environments.

## Directory Structure

```
openai-gym-pytorch/
├── models/
│   ├── cartpole-v0/
│   │   └── dqn.py
│   └── mountain-cart-v0/
│       ├── curiosity-dqn.py
│       └── forward-dynamics-model/
│           ├── dynamics.py
│           └── generate-transition-data.py
├── README.md
```

- `models/cartpole-v0/dqn.py`: DQN agent for CartPole-v1.
- `models/mountain-cart-v0/curiosity-dqn.py`: DQN agent with curiosity for MountainCarContinuous-v0.
- `models/mountain-cart-v0/forward-dynamics-model/`: Scripts for training and using a forward dynamics model.

## Setup

### Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/) (for dynamics model)
- (Optional) IPython for inline plotting

Install dependencies with:

```bash
pip install torch gymnasium[all] matplotlib pandas scikit-learn
```

### Video Support

To record videos, ensure you have `ffmpeg` installed:

```bash
# On Ubuntu
sudo apt-get install ffmpeg
# On MacOS
brew install ffmpeg
```

## Usage

### CartPole DQN

Train a DQN agent on CartPole-v1:

```bash
python models/cartpole-v0/dqn.py
```

Videos and training statistics will be saved in the `cartpole-agent` directory.

### MountainCarContinuous with Curiosity

1. **Generate Transition Data** (for the forward dynamics model):

    ```bash
    python models/mountain-cart-v0/forward-dynamics-model/generate-transition-data.py
    ```

2. **Train the Forward Dynamics Model**:

    ```bash
    python models/mountain-cart-v0/forward-dynamics-model/dynamics.py
    ```

3. **Train the Curiosity-Driven DQN Agent**:

    ```bash
    python models/mountain-cart-v0/curiosity-dqn.py
    ```

Videos and training statistics will be saved in the `mountain-cart-agent` directory.

## Customization

- **Change Environment**: Modify the `env = gym.make(...)` line in the relevant script.
- **Hyperparameters**: Adjust batch size, learning rate, and other parameters at the top of each script.
- **Model Architecture**: Edit the `model` class in each script to change the neural network.

## References

- [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [OpenAI Gymnasium](https://gymnasium.farama.org/)
- [Curiosity-driven Exploration](https://pathak22.github.io/noreward-rl/)

## More Information

See https://rcatullo.com/stacks/dqn for more details on this specific implementation.

---
