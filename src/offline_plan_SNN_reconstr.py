import time
from optimal_search_utils import read_controller_specs, automatic_state_sampling, perform_gesture, plot_optimal_controller, save_fig
from RegressorPlannerTrainingTesting import test_regressor
import numpy as np
from scipy.interpolate import interp1d
import sys 
sys.path.append('./IO')
from amc_processing import txt2amc

def frames2time(data, frame_rate):
    frame_numbers = np.arange(0, data.shape[0], 1)
    frame_times = np.floor(1000 * (1 / frame_rate) * np.array(frame_numbers))
    T = frame_times[-1]  # recording time in milliseconds
    time_steps = np.arange(0, T+1)

    data_rescaled = np.zeros((data.shape[1], len(time_steps)))

    for joint in range(data.shape[1]):
        f1 = interp1d(frame_times, data[:, joint], kind='cubic', fill_value='extrapolate')
        data_rescaled[joint, :] = f1(time_steps)
    
    return data_rescaled, frame_times



learned_targets, ground_truth_dataset = test_regressor(motion_file="./data/walk.amc", model_path="./models/planner/Regressor_hid128_ov0.pt", sampling_rate=7.5)
np.savetxt("./data/learned_targets.txt", learned_targets, delimiter=",")

ctrlr_specs = read_controller_specs("./data/optimal_controller_specs.txt")

DOF_no = int(ground_truth_dataset.X.shape[1]/2)
angles = ground_truth_dataset.norm_fact*ground_truth_dataset.X[:, :DOF_no]
angles, ft = frames2time(angles, 120)
ft = ft.astype(int)
optimal_controlled_angles = np.transpose([angles[:, 0]]*angles.shape[1])

learned_targets = learned_targets*ground_truth_dataset.norm_fact
learned_targets, ft2  = frames2time(learned_targets, 120)

t = time.time()
for joint in range(angles.shape[0]):
    t = time.time()
    temp = np.where((ctrlr_specs[:, 0] == joint))[0]  # get the specs of the trained controller
    if len(temp) > 0:
        ind = temp[0]
        joint_ctrlr_specs = ctrlr_specs[ind, 1:]
    else:
        joint_ctrlr_specs = [0, 0, 0, 0, 0, 3]

    joint_angle = angles[joint, :]
    moving_joint_threshold = 0.1
    if np.max(joint_angle) - np.min(joint_angle) > moving_joint_threshold and np.max(joint_angle) - np.min(joint_angle) < 200:
        # triggers, target_angles = automatic_state_sampling(joint_angle.reshape(1, len(joint_angle)), moving_joint_threshold, deriv_window=20, oversampling=0)

        target_angles = learned_targets[joint, :].reshape(1, learned_targets.shape[1])
        ctrl_angles = perform_gesture(joint_angle.reshape(1, joint_angle.shape[0]), target_angles.reshape(1, target_angles.shape[1]), joint_ctrlr_specs)

        optimal_controlled_angles[joint, :] = ctrl_angles

        handle = plot_optimal_controller(None, "Time (ms)", "Angle(deg)", "Angle for joint " + str(joint), joint_angle, target_angles, ctrl_angles)
        save_fig(handle, "./figs/learned_SNN_joints", "Joint" + str(joint))
    print("Joint testing time: ", time.time() - t)
    print("Recreating sequence, {:.1f} % done".format(100*(joint/angles.shape[0])))


optimal_controlled_angles_downsampled = optimal_controlled_angles.copy()
optimal_controlled_angles_downsampled = optimal_controlled_angles_downsampled[:, ft]
optimal_controlled_angles_downsampled = optimal_controlled_angles_downsampled.transpose()
print("Size of saved array is: ", optimal_controlled_angles_downsampled.shape)
np.savetxt("./data/walk_reconstr.txt", optimal_controlled_angles_downsampled, delimiter=",")

txt2amc("./data/walk_reconstr.txt", "./data/walk_reconstr.amc")