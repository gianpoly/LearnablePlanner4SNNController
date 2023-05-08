import sys
sys.path.append('./IO')
from amc_processing import construct_CMU_train_set

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from RegressorNet import RegressorNet

recording_file="./data/walk.amc"
construct_CMU_train_set(recording_file, './data/training_sequence.txt')

recording = './data/training_sequence.txt'
motion_seq = np.loadtxt(recording)
print(motion_seq.shape)

x = torch.linspace(-5, 5, 100).view(100, 1)
y_target = np.zeros((100, 1), dtype=np.float32)
y_target[50:, 0] = 1
y_target = torch.from_numpy(y_target)


net = RegressorNet(input_size=1, output_size=1)
optimizer = optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

y = net(x)

plt.figure()
plt.plot(x.detach().numpy(), y.detach().numpy())
plt.show()

losses = []
epochs = 1000
for i in range(epochs):
    y = net(x)
    loss = loss_fn(y, y_target)
    losses.append(loss.detach())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % int(epochs/10) == 0:
        plt.figure()
        plt.plot(losses)
        plt.show()

y = net(x)

plt.figure()
plt.plot(x.detach().numpy(), y.detach().numpy())
plt.plot(x.detach().numpy(), y_target.detach().numpy())
plt.show()

# print(y.size())
# print(y_target.size())