import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "forward-dynamics-model"))
import dynamics

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
    training_period = 100
else:
    num_episodes = 250
    training_period = 50


env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
env = RecordVideo(env, video_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mountain-cart-agent"), name_prefix="training",
                   episode_trigger=lambda x: x % training_period == 0)
env = RecordEpisodeStatistics(env)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display  # type: ignore

plt.ion()

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    # Add transition to memory
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    # Sample batch from memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_dim)
    
    def init_weights(self):
        if isinstance(self, nn.Linear):
            init.kaiming_normal_(self.weight)
    
    # Compute either one element to determine action 
    # or batch during training. 
    # Returns tensor([[Qleft,Qright]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# ETA is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
ETA = 1e-4
BINS = 20
BETA_START = 0.9
BETA_END = 0.001
BETA_DECAY = 10
low = env.action_space.low[0]
high = env.action_space.high[0]

def get_action(bin):
    action = low + (float(bin) / (BINS-1)) * (high - low)
    return action

def get_bin(action):
    bin = int((action - low) / (high - low) * (BINS-1))
    return bin

# Get number of actions from gym action space
n_actions = BINS
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

online_net = model(n_observations, n_actions).apply(model.init_weights).to(device)
target_net = model(n_observations, n_actions).apply(model.init_weights).to(device)
# Copy parameters from online to target
target_net.load_state_dict(online_net.state_dict())

# Load the dynamics model
dynamics_model = dynamics.model(input_dim=n_observations + 1, output_dim=n_observations).to(device)
dynamics_model.load_state_dict(torch.load("models/mountain-cart-v0/forward-dynamics-model/dynamics_model.pt"))
dynamics_model.eval()

# Optimize with Adam/AMSGrad
optimizer = optim.AdamW(online_net.parameters(), lr=ETA, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def compute_intrinsic_reward(state, action, reward, next_state, i_episode):
    if next_state == None:
        return reward
    input_tensor = torch.cat((state, action.unsqueeze(1)), dim=1).to(device)
    with torch.no_grad():
        # Get the predicted next state from the dynamics model
        next_state_pred = dynamics_model(input_tensor)
        intrinsic_reward = torch.mean(torch.abs(next_state - next_state_pred)).item()
    weight = BETA_END + (BETA_START - BETA_END) * math.exp(-1 * i_episode / BETA_DECAY)
    return reward + weight * intrinsic_reward

# Select an action with highest Q_value (returns bin, not float)
def select_action(state):
    global steps_done
    steps_done += 1
    with torch.no_grad():
        # Get bin with highest Q-value
        bin = online_net(state).max(1).indices.view(1,1)
        return torch.tensor([[bin.item()]], dtype=torch.long, device=device)

episode_reward = []
episode_int_reward = []

def plot_reward(show_result=False):
    rewards_t = torch.tensor(episode_reward, dtype=torch.float, device=device)
    int_rewards_t = torch.tensor(episode_int_reward, dtype=torch.float, device=device)
    plt.figure(1)
    plt.clf()
    if show_result:
        plt.title('Result')
    else:
        plt.title('Training...')
    plt.plot(rewards_t.numpy(), label='Reward')
    if len(rewards_t) >= 100:
        means_reward = rewards_t.unfold(0, 100, 1).mean(1)
        means_reward = torch.cat((torch.zeros(99), means_reward))
        plt.plot(means_reward.numpy(), label='100 Episode Average Reward')
    plt.plot(int_rewards_t.numpy(), label='Intrinsic Reward')
    if len(int_rewards_t) >= 100:
        means_int = int_rewards_t.unfold(0, 100, 1).mean(1)
        means_int = torch.cat((torch.zeros(99), means_int))
        plt.plot(means_int.numpy(), label='100 Episode Average Intrinsic Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Perform one step of the optimization (on the online network)
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # Transpose array of transitions to Transition of arrays
    # E.g. batch.state, batch.action, ... are arrays of Tensors representing Transitions. 
    batch = Transition(*zip(*transitions))

    # Tensor of next_states in batch which are not final
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s,a) for each state-action pair in the batch
    state_action_values = online_net(state_batch).gather(1, action_batch)

    # Compute max Q(s',a') for all next states s', or 0 if s' is terminal
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # Bitmask of booleans for next_state's s' in batch that are not final
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    # target = r + gamma * max Q(s', a')
    target_state_action_values = reward_batch + GAMMA * next_state_values

    # Compute Huber loss
    huber = nn.HuberLoss()
    J = huber(state_action_values, target_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    J.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(online_net.parameters(), 100)
    optimizer.step()


# Main episodic training loop
for i_episode in range(num_episodes):
    if (i_episode + 1) % training_period == 0:
        print(f"Episode {i_episode + 1} of {num_episodes}")
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    cumulative_reward = 0
    cumulative_intrinsic_reward = 0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(np.array([get_action(action.item())], dtype=np.float32))
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        intrinsic_reward = compute_intrinsic_reward(state, torch.tensor([get_action(action.item())], dtype=torch.float32, device=device), reward, next_state, i_episode)
        cumulative_reward += reward
        cumulative_intrinsic_reward += intrinsic_reward

        reward = torch.tensor([intrinsic_reward], device=device)

        # Fix: Push the reward before next_state (order: state, action, reward, next_state)
        memory.push(state, action, reward, next_state)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the online network)
        optimize_model()

        # Soft update of the target network's weights
        target_net_state_dict = target_net.state_dict()
        online_net_state_dict = online_net.state_dict()
        for key in online_net_state_dict:
            target_net_state_dict[key] = online_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_reward.append(cumulative_reward)
            episode_int_reward.append(cumulative_intrinsic_reward)
            plot_reward()
            break

print('Complete')
plot_reward(show_result=True)
plt.ioff()
plt.show()