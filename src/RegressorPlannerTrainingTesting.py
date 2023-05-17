import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from RegressorHelpers import RegressorNet, MotionSequenceDataset, setup_regressor_testing_env, evaluate_regressor

def train_regressor(motion_file, oversampling, hidden_neurons):

    recording_file = motion_file
    dataset = MotionSequenceDataset(recording_file, oversampling=oversampling)

    X = torch.tensor(dataset.X, dtype=torch.float32)
    Y_target = torch.tensor(dataset.Y, dtype=torch.float32)

    net = RegressorNet(input_size=X.size(dim=1), hidden_size=hidden_neurons, output_size=Y_target.size(dim=1))
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

        # if i % int(epochs/4) == 0:
        #     plt.figure()
        #     plt.plot(losses)
        #     plt.show()

    plt.figure()
    plt.plot(losses)
    plt.grid()
    plt.ylim([0, 0.1])
    plt.ylabel("MSE Loss")
    plt.xlabel("# Epochs")
    plt.savefig(f'./figs/Loss_hid{net.hidden_neurons}_ov{dataset.oversampling}.svg')
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
        axs[joint].text(0.5, -0.9, f"Joint {joint}")
    plt.show()

    PATH = f"./models/planner/Regressor_hid{net.hidden_neurons}_ov{dataset.oversampling}.pt"
    torch.save(net.state_dict(), PATH)


def test_regressor(motion_file, model_path, sampling_rate):
    # Sample dataset at multiple timesteps and evaluate regressor and compare with ground truth
    dataset, net = setup_regressor_testing_env(motion_file, model_path)

    frame_step = int(120/sampling_rate)
    query_frames = np.arange(0, dataset.X.shape[0], frame_step)
    preds = np.zeros((len(query_frames), dataset.Y.shape[1]))
    
    for i, frame in enumerate(query_frames):
        t = time.time()
        Y = evaluate_regressor(dataset, net, frame).numpy()
        dt = time.time() - t
        print(f'Query time: {dt} s')
        preds[i, :] = Y
        print(f'------------------')
    
    ctr = 0
    preds_zh = np.zeros_like(dataset.Y)
    for j in range(preds_zh.shape[0]):
        if j in query_frames:
            preds_zh[j, :] = preds[ctr, :]
            ctr += 1
        else:
            preds_zh[j, :] = preds[ctr-1, :]
    fig, axs = plt.subplots(8, 8, sharex='col', sharey='row', figsize=(12, 12))
    axs = axs.flatten()
    plt.tight_layout()
    for joint in range(0, dataset.Y.shape[1]):
        axs[joint].plot(dataset.targets[:, joint])
        # axs[joint].plot(preds_zh[:, joint])
        axs[joint].scatter(query_frames, preds[:, joint], c='r', s=2)
        axs[joint].set_ylim([-1, 1])
        axs[joint].text(0.5, -0.9, f"Joint {joint}")
    plt.show()

    plt.figure()
    plt.plot(dataset.targets[:, 51])
    plt.plot(preds_zh[:, 51])
    plt.scatter(query_frames, preds[:, 51], c='r', s=10)
    plt.ylim([-1, 1])
    plt.show()

    return preds_zh

if __name__ == "__main__":
    # train_regressor("./data/walk.amc", oversampling=0, hidden_neurons=256)
    
    learned_targets = test_regressor(motion_file="./data/walk.amc", model_path="./models/planner/Regressor_hid128_ov0.pt", sampling_rate=7.5)
    np.savetxt("./data/learned_targets.txt", learned_targets, delimiter=",")
