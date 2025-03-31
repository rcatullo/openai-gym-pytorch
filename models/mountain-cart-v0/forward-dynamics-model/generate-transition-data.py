import numpy as np
import gymnasium as gym
import pandas as pd
from collections import namedtuple, deque
import pathlib

BINS = 20
N_SAMPLES = 100000

env = gym.make("MountainCarContinuous-v0")

low = env.action_space.low[0]
high = env.action_space.high[0]

def get_action(bin):
    action = low + (float(bin) / (BINS-1)) * (high - low)
    return action

def get_bin(action):
    bin = int((action - low) / (high - low) * (BINS-1))
    return bin

n_actions = BINS
state, info = env.reset()
n_obserations = len(state)

transitions = {
    'state': np.empty((0, n_obserations), dtype=np.float32),
    'action': np.empty((0, 1), dtype=np.float32),
    'next_state': np.empty((0, n_obserations), dtype=np.float32)
}

for s in range(N_SAMPLES):
    if (s+1) % 10000 == 0:
        print(f"Sampled {s} transitions")
    state = env.observation_space.sample()
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    transitions['state'] = np.append(transitions['state'], np.array([state]), axis=0)
    transitions['action'] = np.append(transitions['action'], np.array([action.item()]))
    transitions['next_state'] = np.append(transitions['next_state'], np.array([next_state]), axis=0)
    if terminated or truncated:
        state, info = env.reset()

df_dict = {}
for key, arr in transitions.items():
    # If the value is a 2D array, create a column for each index
    if arr.ndim == 2:
        for i in range(arr.shape[1]):
            df_dict[f"{key}_{i}"] = arr[:, i]
    else:
        df_dict[key] = arr
df = pd.DataFrame(df_dict)
path = pathlib.Path.joinpath(pathlib.Path(__file__).parent.resolve(), "transitions.csv")
df.to_csv(path, index=False)
print("Transition data generated and saved to ", path)
