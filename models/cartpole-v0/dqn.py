import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

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
    num_episodes = 300
    training_period = 25


env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="training",
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
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

online_net = model(n_observations, n_actions).apply(model.init_weights).to(device)
target_net = model(n_observations, n_actions).apply(model.init_weights).to(device)
# Copy parameters from online to target
target_net.load_state_dict(online_net.state_dict())

# Optimize with Adam/AMSGrad
optimizer = optim.AdamW(online_net.parameters(), lr=ETA, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


# Select an action based on epsilon-greedy policy
def select_action(state):
    global steps_done
    eps = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1
    if eps > eps_threshold:
        with torch.no_grad():
            return online_net(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=device)

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
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
for i_episode in range(1, num_episodes+1):
    if i_episode % training_period == 0:
        print(f"Episode {i_episode} of {num_episodes}")
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

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
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()