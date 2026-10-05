import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from RegressorHelpers import setup_regressor_testing_env


motion_sequence_file_path="./data/walk.amc"
trained_planner_path="./models/planner/Regressor_hid128_ov0_dagger.pt"


ground_truth_dataset, net = setup_regressor_testing_env(motion_sequence_file_path, trained_planner_path)
DOF_no = int(ground_truth_dataset.X.shape[1]/2)

reconstr_targets = np.loadtxt("./data/reconstructed_targets.txt", delimiter=",")
reconstr_angles = np.loadtxt("./data/reconstructed_angles.txt", delimiter=",")
reconstr_speeds = np.loadtxt("./data/reconstructed_speeds.txt", delimiter=",")

gt_targets = ground_truth_dataset.norm_fact*ground_truth_dataset.targets  # dataset targets are normalized, reconstructed ones are in degrees
gt_angles = ground_truth_dataset.norm_fact*ground_truth_dataset.motion_seq  # normalized → degrees

# One window, two stacked panels. Left/right arrows step through DOFs; the text box jumps to a typed index.
plt.rcParams["keymap.back"] = [k for k in plt.rcParams["keymap.back"] if k != "left"]  # free arrows from matplotlib's view-history shortcuts
plt.rcParams["keymap.forward"] = [k for k in plt.rcParams["keymap.forward"] if k != "right"]
fig, (ax_tgt, ax_ang) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
fig.subplots_adjust(bottom=0.15)

line_gt_tgt, = ax_tgt.plot(gt_targets[:, 0], label="ground-truth targets")
line_rc_tgt, = ax_tgt.plot(reconstr_targets[:, 0], label="planner targets")
ax_tgt.set_ylabel("Angle (deg)")
ax_tgt.legend(loc="upper right")

line_gt_ang, = ax_ang.plot(gt_angles[:, 0], label="raw sequence")
line_rc_ang, = ax_ang.plot(reconstr_angles[:, 0], label="SNN")  # already in degrees
ax_ang.set_xlabel("Frame")
ax_ang.set_ylabel("Angle (deg)")
ax_ang.legend(loc="upper right")

text_box = TextBox(fig.add_axes((0.45, 0.02, 0.1, 0.05)), "DOF (0-%d) " % (DOF_no - 1), initial="51")
joint = 51  # DOF to inspect (51 = rtibia)


def show_joint(j: int) -> None:
    global joint
    joint = j % DOF_no
    line_gt_tgt.set_ydata(gt_targets[:, joint])
    line_rc_tgt.set_ydata(reconstr_targets[:, joint])
    line_gt_ang.set_ydata(gt_angles[:, joint])
    line_rc_ang.set_ydata(reconstr_angles[:, joint])
    for ax in (ax_tgt, ax_ang):
        ax.relim()
        ax.autoscale_view()
    ax_tgt.set_title(f"Targets, DOF {joint}")
    ax_ang.set_title(f"Joint angles, DOF {joint}")
    text_box.eventson = False  # keep the box in sync with arrow-key navigation without re-triggering on_submit
    text_box.set_val(str(joint))
    text_box.eventson = True
    fig.canvas.draw_idle()


def on_submit(text: str) -> None:
    try:
        show_joint(int(text))
    except ValueError:
        print(f"Not a DOF index: {text!r}")


def on_key(event) -> None:
    if text_box.capturekeystrokes:  # arrows move the cursor while typing in the box
        return
    if event.key == "right":
        show_joint(joint + 1)
    elif event.key == "left":
        show_joint(joint - 1)


text_box.on_submit(on_submit)
fig.canvas.mpl_connect("key_press_event", on_key)
show_joint(joint)

plt.show()
