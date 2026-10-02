from RegressorHelpers import setup_regressor_testing_env
from optimal_search_utils import read_controller_specs
from closed_loop import run_closed_loop

import numpy as np

motion_sequence_file_path="./data/walk.amc"
trained_planner_path="./models/planner/Regressor_hid128_ov0_dagger.pt"
event_driven = True  # re-plan a DOF's target only when the SNN reaches it or stops approaching it (else every frame)
use_oracle = False  # take new targets from the GT targets of the current frame instead of the planner
dwell = 5  # minimum frames between two re-plans of the same DOF (event-driven only)
max_hold = 10  # maximum frames between two re-plans of the same DOF (event-driven only)
arrival_frac = 0.5  # a DOF has arrived once its error is below this fraction of the error when its target was set

ground_truth_dataset, net = setup_regressor_testing_env(motion_sequence_file_path, trained_planner_path)
DOF_no = int(ground_truth_dataset.X.shape[1]/2)
SNN_ctrlr_specs = read_controller_specs("./data/optimal_controller_specs.txt")

reconstr_frame_seq, reconstr_targets, reconstr_events = run_closed_loop(
      ground_truth_dataset, net, SNN_ctrlr_specs, event_driven=event_driven, use_oracle=use_oracle,
      dwell=dwell, max_hold=max_hold, arrival_frac=arrival_frac)

np.savetxt("./data/reconstructed_targets.txt", reconstr_targets, delimiter=",")
np.savetxt("./data/reconstructed_angles.txt", reconstr_frame_seq, delimiter=",")
np.savetxt("./data/reconstructed_speeds.txt", reconstr_frame_seq[:, DOF_no:], delimiter=",")
np.savetxt("./data/reconstructed_events.txt", reconstr_events, delimiter=",", fmt="%d")
