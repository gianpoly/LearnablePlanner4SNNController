import time
from RegressorHelpers import setup_regressor_testing_env, query_regressor
from optimal_search_utils import read_controller_specs
from controllers import AgentController

import numpy as np

fps = 120
f_query = 120
frame_interv_ms = round(1000/f_query, 1)

motion_sequence_file_path="./data/walk.amc"
trained_planner_path="./models/planner/Regressor_hid128_ov0.pt"

ground_truth_dataset, net = setup_regressor_testing_env(motion_sequence_file_path, trained_planner_path)
DOF_no = int(ground_truth_dataset.X.shape[1]/2)
seq_time_len = np.round(ground_truth_dataset.X.shape[0]/fps, 3)*1000 + 1
seq_frame_len = ground_truth_dataset.X.shape[0]

SNN_ctrlr_specs = read_controller_specs("./data/optimal_controller_specs.txt")
dt = 0.1 # SNN integration timestep
T = seq_time_len
ts = np.arange(0, (T - dt)/dt, 1).round(2) 
SNN_char_ctrlr = AgentController(dt, ground_truth_dataset.X[0, 0:DOF_no], specs_list=SNN_ctrlr_specs)

reconstr_frame_seq = np.zeros((seq_frame_len, ground_truth_dataset.X.shape[1]))
reconstr_frame_seq[:, DOF_no:] = ground_truth_dataset.X[:, DOF_no:]
reconstr_time_seq = np.zeros((round(seq_time_len/dt), ground_truth_dataset.X.shape[1]))
reconstr_frame_seq[0, :] = ground_truth_dataset.X[0, :]
reconstr_time_seq[0, :] = ground_truth_dataset.X[0, :]
reconstr_frame_speeds = np.zeros((seq_frame_len, DOF_no))

reconstr_targets = np.zeros((seq_frame_len, DOF_no))

for t in np.arange(1, (T - dt)/dt, 1).round(2):        
      if ((t-1)) % int(frame_interv_ms/dt) == 0:
            print(f"Time: {(t-1)*dt} ms")
            frame_ind = int((t-1)/(int(frame_interv_ms/dt)))
            print(f"Frame: {frame_ind}")
            reconstr_frame_seq[frame_ind, :] = reconstr_time_seq[int((t-1)), :]  # Fetch state from renderer array
            norm_state = np.concatenate((reconstr_frame_seq[frame_ind, :DOF_no]/ground_truth_dataset.norm_fact, reconstr_frame_seq[frame_ind, DOF_no:]/ground_truth_dataset.derivs_norm_fact))
            targets = ground_truth_dataset.norm_fact*query_regressor(norm_state, net)  # Query regressor
            print("--------------------")
            reconstr_targets[frame_ind, :] = targets # Save new targets
            np.savetxt("./data/reconstructed_targets.txt", reconstr_targets, delimiter=",")
            np.savetxt("./data/reconstructed_angles.txt", reconstr_frame_seq, delimiter=",")
            reconstr_frame_speeds[frame_ind, :] = reconstr_frame_seq[frame_ind, DOF_no:]/ground_truth_dataset.derivs_norm_fact
            np.savetxt("./data/reconstructed_speeds.txt", reconstr_frame_speeds, delimiter=",")


      SNN_char_ctrlr.update_state(targets)  # Update SNN with new orr past targets
      reconstr_time_seq[int(t), :DOF_no] = SNN_char_ctrlr.get_state() # Save new SNN state







