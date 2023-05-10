import os
import time
from optimal_search_utils import *

recording = "training_sequence.txt"
fps = 120
frame_ind = generate_SNN_training_set(recording, fps).astype(int)

angles = np.loadtxt('./data/SNN_train_data.txt')


optimal_controlled_angles = np.transpose([angles[:, 0]]*angles.shape[1])
print(optimal_controlled_angles.shape)
optimal_ctrlr_IDs = []
print(optimal_controlled_angles.shape)

t = time.time()
for joint in range(angles.shape[0]):
    print("Joint " + str(joint) + " out of " + str(angles.shape[0]))
    joint_angle = angles[joint, :]
    moving_joint_threshold = 0.1
    if np.max(joint_angle) - np.min(joint_angle) > moving_joint_threshold and np.max(joint_angle) - np.min(joint_angle) < 200:
        triggers, target_angles = automatic_state_sampling(joint_angle.reshape(1, len(joint_angle)), moving_joint_threshold, deriv_window=20, oversampling=0)
        ctrl_angles1, best_ctrlr_ID = search_ctrlrs_smart(joint_angle, target_angles)
        optimal_controlled_angles[joint, :] = ctrl_angles1
        best_ctrlr_label = np.insert(best_ctrlr_ID, 0, int(joint))
        optimal_ctrlr_IDs.append(best_ctrlr_label)

        handle = plot_optimal_controller(None, "Time (ms)", "Angle(deg)", "Angle for joint " + str(joint), joint_angle, target_angles, ctrl_angles1)
        save_fig(handle, "figs/SNN_joints/", "Joint" + str(joint))

print("Controller Search time: ", time.time() - t)

optimal_controlled_angles_downsampled = (np.pi/180)*optimal_controlled_angles.copy()
optimal_controlled_angles_downsampled = optimal_controlled_angles_downsampled[:, frame_ind]
optimal_controlled_angles_downsampled = optimal_controlled_angles_downsampled.transpose()
print("Size of saved array is: ", optimal_controlled_angles_downsampled.shape)
np.savetxt("/control_angles_full.txt", optimal_controlled_angles, delimiter=",")
np.savetxt("/control_angles.txt", optimal_controlled_angles_downsampled, delimiter=",")
np.savetxt("/optimal_controller_specs.txt", optimal_ctrlr_IDs, delimiter=",", fmt="%i")
