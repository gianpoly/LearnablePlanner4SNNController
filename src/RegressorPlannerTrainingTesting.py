import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from RegressorHelpers import RegressorNet, MotionSequenceDataset, setup_regressor_testing_env, evaluate_regressor
from controllers import AgentController
from optimal_search_utils import read_controller_specs
from dataset_utils import estim_deriv

def simulate_SNN_states(dataset: MotionSequenceDataset, specs_path: str, fps: int = 120, dt: float = 0.1) -> np.ndarray:
    """
    Run the SNN controllers open-loop on the planner's training targets and return the states they produce,
    normalized the same way as dataset.X (speeds computed exactly as in the real-time closed loop)

    :param dataset: Motion sequence dataset providing the targets and the normalization factors
    :param specs_path: Path to the optimal controller specs found by the SNN controller search
    :param fps: Frame rate of the motion sequence
    :param dt: SNN integration timestep (ms)
    :return: (np.array) frames x 2*DOF array of normalized angles and speeds, same layout as dataset.X
    """
    DOF_no = dataset.Y.shape[1]
    targets = dataset.norm_fact*dataset.Y  # SNN works in degrees
    angles = np.zeros_like(targets)
    angles[0, :] = dataset.norm_fact*dataset.X[0, :DOF_no]
    SNN_ctrlr = AgentController(dt, angles[0, :], specs_list=read_controller_specs(specs_path))

    steps_per_frame = 1000/(fps*dt)
    for frame_ind in range(angles.shape[0] - 1):
        n_steps = round((frame_ind + 1)*steps_per_frame) - round(frame_ind*steps_per_frame)
        for _ in range(n_steps):
            SNN_ctrlr.update_state(targets[frame_ind, :])
        angles[frame_ind + 1, :] = SNN_ctrlr.get_state()

    norm_angles = angles/dataset.norm_fact
    speeds = estim_deriv(norm_angles)/dataset.derivs_norm_fact
    return np.concatenate((norm_angles, speeds), axis=1)


def train_regressor(motion_file: str, oversampling: int, hidden_neurons: int, snn_specs_path: str | None = None, snn_angles: bool = False) -> None:

    recording_file = motion_file
    dataset = MotionSequenceDataset(recording_file, oversampling=oversampling)

    # Optionally train on the states the SNN produces when reproducing the motion instead of the GT ones (targets stay GT)
    model_suffix = ""
    if snn_specs_path is not None:
        DOF_no = dataset.Y.shape[1]
        snn_states = simulate_SNN_states(dataset, snn_specs_path)
        if snn_angles:
            dataset.X = snn_states
            model_suffix = "_snnstate"
        else:
            dataset.X[:, DOF_no:] = snn_states[:, DOF_no:]
            model_suffix = "_snnspeed"

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
    plt.savefig(f'./figs/Loss_hid{net.hidden_neurons}_ov{dataset.oversampling}{model_suffix}.svg')
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

    PATH = f"./models/planner/Regressor_hid{net.hidden_neurons}_ov{dataset.oversampling}{model_suffix}.pt"
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

    return preds_zh, dataset

if __name__ == "__main__":
    train_regressor("./data/walk.amc", oversampling=0, hidden_neurons=128)
    
    # learned_targets = test_regressor(motion_file="./data/walk.amc", model_path="./models/planner/Regressor_hid128_ov0.pt", sampling_rate=7.5)
    # np.savetxt("./data/learned_targets.txt", learned_targets, delimiter=",")
