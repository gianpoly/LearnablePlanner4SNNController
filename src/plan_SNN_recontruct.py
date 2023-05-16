import time
from RegressorHelpers import setup_regressor_testing_env, query_regressor
from optimal_search_utils import read_controller_specs
from controllers import AgentController

import numpy as np

fps = 120
f_query = 120
frame_interv_ms = round(1000/120, 1)

motion_sequence_file_path="./data/walk.amc"
trained_planner_path="./models/planner/Regressor_hid128_ov0.pt"

ground_truth_dataset, net = setup_regressor_testing_env(motion_sequence_file_path, trained_planner_path)
DOF_no = int(ground_truth_dataset.X.shape[1]/2)
seq_time_len = np.round(ground_truth_dataset.X.shape[0]/fps, 3)*1000
seq_frame_len = ground_truth_dataset.X.shape[0]

SNN_ctrlr_specs = read_controller_specs("./data/optimal_controller_specs.txt")
dt = 0.1 # SNN integration timestep
T = seq_time_len
ts = np.arange(0, (T - dt)/dt, 1).round(2) 
SNN_char_ctrlr = AgentController(dt, ground_truth_dataset.X[0, 0:DOF_no], specs_list=SNN_ctrlr_specs)

reconstr_frame_seq = np.zeros((seq_frame_len, ground_truth_dataset.X.shape[1]))
reconstr_time_seq = np.zeros((round(seq_time_len/dt), ground_truth_dataset.X.shape[1]))
reconstr_frame_seq[0, :] = ground_truth_dataset.X[0, :]
reconstr_time_seq[0, :] = ground_truth_dataset.X[0, :]

reconstr_targets = np.zeros((seq_frame_len, DOF_no))

for t in ts:
        frame_ind = t/(frame_interv_ms/dt)
        if t % (frame_interv_ms/dt) == 0:
              print(t)
              reconstr_frame_seq[int(frame_ind), :] = reconstr_time_seq[int(t), :]
              targets = query_regressor(reconstr_frame_seq[int(frame_ind), :], net)
              reconstr_targets[int(frame_ind), :] = targets

        SNN_char_ctrlr.update_state(targets)
        reconstr_time_seq[int(t), :] = SNN_char_ctrlr.get_state()






