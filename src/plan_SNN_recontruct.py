import time
from RegressorHelpers import setup_regressor_testing_env, query_regressor
from optimal_search_utils import read_controller_specs
from controllers import AgentController

import numpy as np


motion_sequence_file_path="./data/walk.amc"
trained_planner_path="./models/planner/Regressor_hid128_ov0.pt"
SNN_ctrlr_specs = read_controller_specs(dir + "./data/optimal_controller_specs.txt")

ground_truth_dataset, net = setup_regressor_testing_env(motion_sequence_file_path, trained_planner_path)
SNN_char_ctrlr

eval_frames = ground_truth_dataset.X.shape[0]
reconstr_seq = np.zeros((eval_frames, ground_truth_dataset.X.shape[1]))
reconstr_seq[0, :] = ground_truth_dataset.X[0, :]


select_joints = [1, 10, 30, 36, 119, 163, 167, 189]
t = time.time()






