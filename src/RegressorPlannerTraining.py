import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from RegressorHelpers import RegressorNet, MotionSequenceDataset

recording_file="./data/walk.amc"
dataset = MotionSequenceDataset(recording_file)

X = torch.tensor(dataset.X, dtype=torch.float32)
Y_target = torch.tensor(dataset.Y, dtype=torch.float32)

net = RegressorNet(input_size=X.size(dim=1), output_size=Y_target.size(dim=1))
optimizer = optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

losses = []
epochs = 1000
for i in range(epochs):
    Y = net(X)
    loss = loss_fn(Y, Y_target)
    losses.append(loss.detach())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % int(epochs/4) == 0:
        plt.figure()
        plt.plot(losses)
        plt.show()

# Plot trained vs target comparison
Y = net(X)

fig, axs = plt.subplots(8, 8, sharex='col', sharey='row', figsize=(12, 12))
axs = axs.flatten()
plt.tight_layout()
for joint in range(0, dataset.Y.shape[1]):
    axs[joint].plot(Y_target[:, joint].detach().numpy())
    axs[joint].plot(Y[:, joint].detach().numpy())
    axs[joint].set_ylim([-1, 1])
plt.show()

# Plot network output at specific frame and compare with target
X_test = torch.tensor(dataset.X[10, :], dtype=torch.float32)
print(f'Input size is: {X_test.size(0)}')
Y_test = torch.tensor(dataset.Y[10, :], dtype=torch.float32)
print(f'Target size is: {Y_test.size(0)}')
with torch.no_grad():
    Y_pred = net(X_test)
print(f'Output size is: {Y_pred.size(0)}')
error = (Y_test - Y_pred)**2
print(f'Error is: {error}')