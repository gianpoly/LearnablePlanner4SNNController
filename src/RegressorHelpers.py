import sys
sys.path.append('./IO')
from amc_processing import build_CMU_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from dataset_utils import zero_crossing, estim_deriv, oversample_trigger_list

class RegressorNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_neurons = hidden_size
        super(RegressorNet, self).__init__()
        self.fc1 = nn.Linear(input_size, self.hidden_neurons)
        self.fc2 = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        self.fc3 = nn.Linear(self.hidden_neurons, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MotionSequenceDataset():
    def __init__(self, recording_file, oversampling) -> None:
        self.oversampling = oversampling
        # Read amc file and create txt equivalent in /data foler
        build_CMU_sequence(recording_file, './data/training_sequence.txt')

        # Read txt file and preprocess (normalize)
        self.motion_seq, self.norm_fact = self.preprocess_recorded_motion_seq('./data/training_sequence.txt')

        # Specify trigger points
        self.derivs, self.trigs = self.set_triggers(oversampling=oversampling)

        # Specify target angles
        self.targets = self.set_targets()

        self.X = np.concatenate((self.motion_seq, self.derivs), axis=1)
        self.Y = self.targets
         
    def preprocess_recorded_motion_seq(self, recording):
            """
            Procedure:
            1. Cuts the sequence to the desired length 'self.ep_len' by selecting a random starting frame and taking
                'self.ep_len' number of frames starting from that frame.
            2. Shuts all joints with less than 2 degree variation, by computing the range of motion for each joint and
                checking if it is less than 2 degrees.
            """

            # Load motion sequence from recording and drop frame numbers
            motion_seq = np.loadtxt(recording)

            ep_len = motion_seq.shape[0]

            # Shut all joints with less than 2 degree variation
            # motion_ranges = np.abs(motion_seq.max(axis=0) - motion_seq.min(axis=0))
            # idle = motion_ranges < 2*np.pi/180
            # motion_seq[:, idle] = motion_seq[0, idle]

            # Normalize motion sequence data to feed the NN
            norm_factor = np.abs(motion_seq).max()
            motion_seq = motion_seq/norm_factor
            
            return motion_seq, norm_factor
    
    def set_triggers(self, oversampling):
        """
        Define the trigger points for the target angles of the motion sequence

        :param oversampling: Number of intermediate samples to be chosen between the automatically generated

        :return:
        derivs: (np.array) contains the estimates of the angle derivatives
        triggers: (list of lists) contains the trigger times
        """

        # Estimate derivative of joint angles
        derivs = estim_deriv(self.motion_seq)

        # Normalize speed data to feed the NN
        derivs = derivs/np.abs(derivs).max()
        # eps = 1e-6
        # derivs = derivs/(np.abs(derivs).max(axis=0) + eps)

        # derivs = (derivs - derivs.mean())/derivs.std()

        # Detect the zero-crossings of the derivative
        zero_crossings = zero_crossing(derivs)

        # Set motion triggers at zero crossings
        triggers = zero_crossings
        # and also add the first and last timesteps
        triggers = [[0] + t + [self.motion_seq.shape[0]-1] for t in triggers]

        # Oversample triggers
        for _ in range(oversampling):
            triggers = [oversample_trigger_list(t) for t in triggers]
        
        # Oversample joint 2 from CMU dataset to accurately capture translation of the skeleton
        if oversampling < 5:
            for _ in range(5):
                triggers[2] = oversample_trigger_list(triggers[2])

        return derivs, triggers

    def set_targets(self):
        """
        Compute the target state for each time step of the motion sequence.

        :return: targets (np.ndarray): Array with shape `(n_timesteps, n_dims)` containing the target state for each
        time step.
        """
        # Initialize targets as the initial state of actual
        targets = np.multiply(np.ones_like(self.motion_seq), self.motion_seq[0, :][np.newaxis, :])

        # At each trigger, set targets to the value of the next trigger
        temp_trigs = deepcopy(self.trigs)
        for dim in range(targets.shape[1]):
            while len(temp_trigs[dim]) > 1:
                init = temp_trigs[dim].pop(0)
                fin = temp_trigs[dim][0]
                targets[init:fin, dim] = self.motion_seq[fin, dim]

        return targets


def setup_regressor_testing_env(motion_file, model_path):
    recording_file = motion_file
    model_specs = model_path.split('_')

    hidden_neurons = int(model_specs[1][3:])
    oversampling = int(model_specs[2][2])

    dataset = MotionSequenceDataset(recording_file, oversampling=oversampling)

    net = RegressorNet(input_size=dataset.X.shape[1], hidden_size=hidden_neurons, output_size=dataset.Y.shape[1])
    net.load_state_dict(torch.load(model_path))
    net.eval()

    return dataset, net


def query_regressor(state, network):
    # Query regressor network given the state of the character (angles and angular velocities)
    X_test = torch.tensor(state, dtype=torch.float32)
    print(f'Input size is: {X_test.size(0)}')
    with torch.no_grad():
        Y_pred = network(X_test)
    print(f'Output size is: {Y_pred.size(0)}')

    return Y_pred


def evaluate_regressor(dataset, network, query_frame):
    # Plot network output at specific frame and compare with target
    X_test = torch.tensor(dataset.X[query_frame, :], dtype=torch.float32)
    print(f'Input size is: {X_test.size(0)}')
    Y_test = torch.tensor(dataset.Y[query_frame, :], dtype=torch.float32)
    print(f'Target size is: {Y_test.size(0)}')
    with torch.no_grad():
        Y_pred = network(X_test)
    print(f'Output size is: {Y_pred.size(0)}')
    error = (Y_test - Y_pred)**2
    print(f'Error is: {error.numpy().mean()}')

    return Y_pred