import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.model_selection import train_test_split

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Read transitions with corrected columns: input as state and action, output as next_state
data = pd.read_csv(pathlib.Path.joinpath(pathlib.Path(__file__).parent.resolve(), "transitions.csv"))
# Changed: input now comprises state and action, output now comprises next_state values
input = data[['state_0', 'state_1', 'action']]
output = data[['next_state_0', 'next_state_1']]

# Split the data into train and test sets (80/20 split)
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    input.to_numpy(), output.to_numpy(), test_size=0.2, random_state=42, shuffle=True
)

# Convert the split data to torch tensors (both as float)
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).to(device)

class model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define the neural network layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

        # Initialize weights
        init.kaiming_normal_(self.fc1.weight)
        init.kaiming_normal_(self.fc2.weight)
        init.kaiming_normal_(self.fc3.weight)
        init.kaiming_normal_(self.fc4.weight)
    
    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Update: output_dim is now 2 for next_state values
dynamics_model = model(input_dim=X_train_tensor.shape[1], output_dim=2).to(device)

# Change loss and optimizer for regression task
criterion = nn.MSELoss()
optimizer = optim.Adam(dynamics_model.parameters(), lr=1e-3)

if __name__ == "__main__":
    epochs = 1000
    train_loss_history, test_loss_history = [], []
    for epoch in range(epochs):
        dynamics_model.train()
        optimizer.zero_grad()
        outputs = dynamics_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        dynamics_model.eval()
        with torch.no_grad():
            train_loss = criterion(dynamics_model(X_train_tensor), y_train_tensor)
            test_loss = criterion(dynamics_model(X_test_tensor), y_test_tensor)
        
        train_loss_history.append(train_loss.item())
        test_loss_history.append(test_loss.item())
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    # Plot train and test loss
    plt.figure()
    plt.plot(range(epochs), train_loss_history, label='Train Loss')
    plt.plot(range(epochs), test_loss_history, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.legend()
    plt.show()

    # Save the model state_dict
    torch.save(dynamics_model.state_dict(), pathlib.Path.joinpath(pathlib.Path(__file__).parent.resolve(), "dynamics_model.pt"))
