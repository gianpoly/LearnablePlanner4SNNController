import numpy as np
import matplotlib.pyplot as plt
from RegressorHelpers import setup_regressor_testing_env


motion_sequence_file_path="./data/walk.amc"
trained_planner_path="./models/planner/Regressor_hid128_ov0.pt"


ground_truth_dataset, net = setup_regressor_testing_env(motion_sequence_file_path, trained_planner_path)
DOF_no = int(ground_truth_dataset.X.shape[1]/2)

reconstr_targets = np.loadtxt("./data/reconstructed_targets.txt", delimiter=",")
reconstr_angles = np.loadtxt("./data/reconstructed_angles.txt", delimiter=",")
reconstr_speeds = np.loadtxt("./data/reconstructed_speeds.txt", delimiter=",")

joint = 10  # DOF to inspect (51 = rtibia)
plt.figure()
plt.plot(ground_truth_dataset.norm_fact*ground_truth_dataset.targets[:, joint])  # dataset targets are normalized, reconstructed ones are in degrees
plt.plot(reconstr_targets[:, joint])
# plt.ylim([-1, 1])
plt.show()