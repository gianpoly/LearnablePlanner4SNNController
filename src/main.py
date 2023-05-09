import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from RegressorHelpers import RegressorNet, MotionSequenceDataset

recording_file="./data/walk.amc"
dataset = MotionSequenceDataset(recording_file)
# print(dataset.X.shape)
# print(dataset.Y.shape)


# x = torch.linspace(-5, 5, 100).view(100, 1)
# y_target = np.zeros((100, 1), dtype=np.float32)
# y_target[50:, 0] = 1
# y_target = torch.from_numpy(y_target)

X = torch.tensor(dataset.X, dtype=torch.float32)
Y_target = torch.tensor(dataset.Y, dtype=torch.float32)

net = RegressorNet(input_size=X.size(dim=1), output_size=Y_target.size(dim=1))
optimizer = optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


Y = net(X)

# plt.figure()
# plt.plot(x.detach().numpy(), y.detach().numpy())
# plt.show()

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

Y = net(X)

# plt.figure()
# plt.plot(X.detach().numpy(), y.detach().numpy())
# plt.plot(x.detach().numpy(), y_target.detach().numpy())
# plt.show()

fig, axs = plt.subplots(8, 8, sharex='col', sharey='row', figsize=(12, 12))
axs = axs.flatten()
plt.tight_layout()
for joint in range(0, dataset.Y.shape[1]):
    axs[joint].plot(Y_target[:, joint].detach().numpy())
    axs[joint].plot(Y[:, joint].detach().numpy())
    axs[joint].set_ylim([-1, 1])
plt.show()
