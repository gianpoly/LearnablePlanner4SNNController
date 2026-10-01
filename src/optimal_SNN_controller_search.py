import time
from multiprocessing import Pool
from optimal_search_utils import *

if __name__ == "__main__":  # required: worker processes re-import this script (spawn start method)
    recording = "training_sequence.txt"
    fps = 120
    frame_ind = generate_SNN_training_set(recording, fps).astype(int)

    angles = np.loadtxt('./data/SNN_train_data.txt')


    optimal_controlled_angles = np.transpose([angles[:, 0]]*angles.shape[1])
    print(optimal_controlled_angles.shape)
    optimal_ctrlr_IDs = []
    print(optimal_controlled_angles.shape)

    # Sample the targets of every moving joint, then search their controllers in parallel (searches are independent)
    moving_joint_threshold = 0.1
    moving_joints = []
    search_args = []
    for joint in range(angles.shape[0]):
        joint_angle = angles[joint, :]
        if np.max(joint_angle) - np.min(joint_angle) > moving_joint_threshold and np.max(joint_angle) - np.min(joint_angle) < 200:
            triggers, target_angles = automatic_state_sampling(joint_angle.reshape(1, len(joint_angle)), moving_joint_threshold, deriv_window=20, oversampling=0)
            moving_joints.append(joint)
            search_args.append((joint_angle, target_angles))

    t = time.time()
    with Pool() as pool:  # one worker per CPU core; chunksize=1 so idle workers pick up the next joint
        search_results = pool.starmap(search_ctrlrs_smart, search_args, chunksize=1)
    print("Controller Search time: ", time.time() - t)

    for joint, (joint_angle, target_angles), (ctrl_angles1, best_ctrlr_ID) in zip(moving_joints, search_args, search_results):
        print("Joint " + str(joint) + " out of " + str(angles.shape[0]) + ": best controller is " + str(best_ctrlr_ID))
        optimal_controlled_angles[joint, :] = ctrl_angles1
        best_ctrlr_label = np.insert(best_ctrlr_ID, 0, int(joint))
        optimal_ctrlr_IDs.append(best_ctrlr_label)

        handle = plot_optimal_controller(None, "Time (ms)", "Angle(deg)", "Angle for joint " + str(joint), joint_angle, target_angles, ctrl_angles1)
        save_fig(handle, "figs/SNN_joints/", "Joint" + str(joint))

    optimal_controlled_angles_downsampled = (np.pi/180)*optimal_controlled_angles.copy()
    optimal_controlled_angles_downsampled = optimal_controlled_angles_downsampled[:, frame_ind]
    optimal_controlled_angles_downsampled = optimal_controlled_angles_downsampled.transpose()
    print("Size of saved array is: ", optimal_controlled_angles_downsampled.shape)
    np.savetxt("./data/control_angles_full.txt", optimal_controlled_angles, delimiter=",")
    np.savetxt("./data/control_angles.txt", optimal_controlled_angles_downsampled, delimiter=",")
    np.savetxt("./data/optimal_controller_specs.txt", optimal_ctrlr_IDs, delimiter=",", fmt="%i")
