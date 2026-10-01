import time
from RegressorHelpers import setup_regressor_testing_env, query_regressor
from optimal_search_utils import read_controller_specs
from controllers import AgentController
from dataset_utils import estim_deriv
from event_utils import TargetEventDetector

import numpy as np

fps = 120
deriv_window = 10  # same window as dataset_utils.estim_deriv (used to build the planner's training inputs)

motion_sequence_file_path="./data/walk.amc"
trained_planner_path="./models/planner/Regressor_hid128_ov0_snnstate.pt"
event_driven = True  # re-plan a DOF's target only when the SNN reaches it or stops approaching it (else every frame)
use_oracle = False  # take new targets from the GT targets of the current frame instead of the planner
dwell = 5  # minimum frames between two re-plans of the same DOF (event-driven only)
max_hold = 10  # maximum frames between two re-plans of the same DOF (event-driven only)
arrival_frac = 0.5  # a DOF has arrived once its error is below this fraction of the error when its target was set

ground_truth_dataset, net = setup_regressor_testing_env(motion_sequence_file_path, trained_planner_path)
norm_fact = ground_truth_dataset.norm_fact
derivs_norm_fact = ground_truth_dataset.derivs_norm_fact
DOF_no = int(ground_truth_dataset.X.shape[1]/2)
seq_frame_len = ground_truth_dataset.X.shape[0]

SNN_ctrlr_specs = read_controller_specs("./data/optimal_controller_specs.txt")
dt = 0.1 # SNN integration timestep
steps_per_frame = 1000/(fps*dt)  # 83.33 SNN steps per mocap frame
init_angles = norm_fact*ground_truth_dataset.X[0, 0:DOF_no]  # SNN works in degrees, dataset X is normalized
SNN_char_ctrlr = AgentController(dt, init_angles, specs_list=SNN_ctrlr_specs)

reconstr_frame_seq = np.zeros((seq_frame_len, ground_truth_dataset.X.shape[1]))  # angles (deg) + normalized speeds
reconstr_targets = np.zeros((seq_frame_len, DOF_no))
reconstr_events = np.zeros((seq_frame_len, DOF_no))
detector = TargetEventDetector(DOF_no, dwell=dwell, max_hold=max_hold, arrival_frac=arrival_frac)

state = init_angles
targets = init_angles.copy()
start = time.time()
for frame_ind in range(seq_frame_len):
      reconstr_frame_seq[frame_ind, :DOF_no] = state
      # Speed estimate computed exactly as for the training set: normalized angle difference over deriv_window frames
      angle_hist = reconstr_frame_seq[max(frame_ind - deriv_window, 0):frame_ind + 1, :DOF_no]/norm_fact
      reconstr_frame_seq[frame_ind, DOF_no:] = estim_deriv(angle_hist)[-1]/derivs_norm_fact
      norm_state = np.concatenate((state/norm_fact, reconstr_frame_seq[frame_ind, DOF_no:]))
      replan = detector.detect(frame_ind, state, targets) if event_driven else np.ones(DOF_no, dtype=bool)
      if replan.any():
            if use_oracle:
                  new_targets = norm_fact*ground_truth_dataset.Y[frame_ind, :]
            else:
                  new_targets = norm_fact*query_regressor(norm_state, net)  # Query regressor
            targets = np.where(replan, new_targets, targets)  # DOFs that didn't fire keep their target
            detector.start_segments(replan, frame_ind, state, targets)
      reconstr_targets[frame_ind, :] = targets # Save new targets
      reconstr_events[frame_ind, :] = replan

      # Advance the SNN to the next frame; rounding the cumulative step count avoids drift from 83.33 steps/frame
      n_steps = round((frame_ind + 1)*steps_per_frame) - round(frame_ind*steps_per_frame)
      for _ in range(n_steps):
            SNN_char_ctrlr.update_state(targets)
      state = SNN_char_ctrlr.get_state()

      if frame_ind % 50 == 0:
            print(f"Frame {frame_ind}/{seq_frame_len} ({time.time() - start:.1f} s)")

np.savetxt("./data/reconstructed_targets.txt", reconstr_targets, delimiter=",")
np.savetxt("./data/reconstructed_angles.txt", reconstr_frame_seq, delimiter=",")
np.savetxt("./data/reconstructed_speeds.txt", reconstr_frame_seq[:, DOF_no:], delimiter=",")
np.savetxt("./data/reconstructed_events.txt", reconstr_events, delimiter=",", fmt="%d")
