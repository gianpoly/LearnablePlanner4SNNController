import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider, TextBox
from mpl_toolkits.mplot3d.art3d import Line3D, Line3DCollection
from RegressorHelpers import setup_regressor_testing_env
sys.path.append('./IO')
from amc_parser import parse_amc, parse_asf


motion_sequence_file_path="./data/walk.amc"
skeleton_file_path="./data/skeleton.asf"
trained_planner_path="./models/planner/Regressor_hid128_ov0_dagger.pt"
joint = 51  # DOF shown at start (51 = rtibia)
gif_path = None  # e.g. "./reconstruction.gif": render every frame to a GIF instead of opening the interactive window


ground_truth_dataset, net = setup_regressor_testing_env(motion_sequence_file_path, trained_planner_path)
DOF_no = int(ground_truth_dataset.X.shape[1]/2)

reconstr_targets = np.loadtxt("./data/reconstructed_targets.txt", delimiter=",")
reconstr_angles = np.loadtxt("./data/reconstructed_angles.txt", delimiter=",")
reconstr_speeds = np.loadtxt("./data/reconstructed_speeds.txt", delimiter=",")

gt_targets = ground_truth_dataset.norm_fact*ground_truth_dataset.targets  # dataset targets are normalized, reconstructed ones are in degrees
gt_angles = ground_truth_dataset.norm_fact*ground_truth_dataset.motion_seq  # normalized → degrees
frames_no = min(len(gt_angles), len(reconstr_angles))


# ---- Skeleton forward kinematics (same code path as IO/3Dviewer.py) ----
joints = parse_asf(skeleton_file_path)
joint_names = list(joints)
parent_idx = [joint_names.index(j.parent.name) if j.parent is not None else -1 for j in joints.values()]

# Map each AMC bone to its column range in the flat 62-DOF vector, using the AMC's own channel order
dof_slices = {}
start = 0
for bone, channels in parse_amc(motion_sequence_file_path)[0].items():
    dof_slices[bone] = (start, start + len(channels))
    start += len(channels)
dof_to_bone = {d: bone for bone, (s, e) in dof_slices.items() for d in range(s, e)}


def skeleton_positions(seq: np.ndarray) -> np.ndarray:
    """Joint coordinates for every frame of a (frames × DOF) angle sequence, shape (frames, joints, 3)."""
    positions = np.zeros((len(seq), len(joint_names), 3))
    for f, frame in enumerate(seq):
        joints['root'].set_motion({bone: list(frame[s:e]) for bone, (s, e) in dof_slices.items()})
        positions[f] = [j.coordinate.ravel() for j in joints.values()]
    return positions[:, :, [2, 0, 1]]  # plot as (z, x, y) so the ASF's y-up axis is vertical, like Joint.draw


gt_pos = skeleton_positions(gt_angles[:frames_no, :DOF_no])
rc_pos = skeleton_positions(reconstr_angles[:frames_no, :DOF_no])
rc_pos_aligned = rc_pos - rc_pos[:, :1] + gt_pos[:, :1]  # SNN pose translated so its root sits on the ground-truth root
bones = [(c, p) for c, p in enumerate(parent_idx) if p >= 0]


# ---- Layout: time series on the left, skeleton on the right ----
# Left/right arrows step through DOFs and the text box jumps to a typed one;
# space plays/pauses, ',' / '.' step frames, the slider scrubs, 'r' toggles root alignment. Drag the 3D panel to rotate the view.
plt.rcParams["keymap.back"] = [k for k in plt.rcParams["keymap.back"] if k != "left"]  # free arrows from matplotlib's view-history shortcuts
plt.rcParams["keymap.forward"] = [k for k in plt.rcParams["keymap.forward"] if k != "right"]
plt.rcParams["keymap.home"] = [k for k in plt.rcParams["keymap.home"] if k != "r"]
fig = plt.figure(figsize=(16, 8))
grid = fig.add_gridspec(2, 2, width_ratios=(1.3, 1), bottom=0.17)
ax_tgt = fig.add_subplot(grid[0, 0])
ax_ang = fig.add_subplot(grid[1, 0], sharex=ax_tgt)
ax_skel = fig.add_subplot(grid[:, 1], projection="3d")

line_gt_tgt, = ax_tgt.plot(gt_targets[:, 0], label="ground-truth targets")
line_rc_tgt, = ax_tgt.plot(reconstr_targets[:, 0], label="planner targets")
ax_tgt.set_ylabel("Angle (deg)")
ax_tgt.legend(loc="upper right")

line_gt_ang, = ax_ang.plot(gt_angles[:, 0], label="raw sequence")
line_rc_ang, = ax_ang.plot(reconstr_angles[:, 0], label="SNN")  # already in degrees
ax_ang.set_xlabel("Frame")
ax_ang.set_ylabel("Angle (deg)")
ax_ang.legend(loc="upper right")

frame_cursors = [ax.axvline(0, color="grey", linestyle="--", linewidth=1) for ax in (ax_tgt, ax_ang)]

gt_bones = Line3DCollection([(gt_pos[0][c], gt_pos[0][p]) for c, p in bones], colors="C0", linewidths=2, label="raw sequence")
rc_bones = Line3DCollection([(rc_pos[0][c], rc_pos[0][p]) for c, p in bones], colors="C1", linewidths=2, label="SNN")
gt_points = Line3D([], [], [], color="C0", marker=".", linestyle="", markersize=5)
rc_points = Line3D([], [], [], color="C1", marker=".", linestyle="", markersize=5)
selected_bone = Line3D([], [], [], color="red", marker="o", linestyle="", markersize=9, label="selected DOF's bone")
for collection in (gt_bones, rc_bones):
    ax_skel.add_collection3d(collection)
for points in (gt_points, rc_points, selected_bone):
    ax_skel.add_line(points)
# Camera follows the ground-truth root horizontally; the window is wide enough to keep both skeletons in view
offsets = np.concatenate([gt_pos, rc_pos]) - np.concatenate([gt_pos[:, :1], gt_pos[:, :1]])
half_width = np.abs(offsets[:, :, :2]).max() + 2
z_lo = min(p[:, :, 2].min() for p in (gt_pos, rc_pos, rc_pos_aligned))
z_hi = max(p[:, :, 2].max() for p in (gt_pos, rc_pos, rc_pos_aligned))
ax_skel.set_zlim(z_lo, z_hi)
ax_skel.set_box_aspect((2*half_width, 2*half_width, z_hi - z_lo))  # equal scale on all axes so the skeleton isn't distorted
ax_skel.set_xlabel("z")
ax_skel.set_ylabel("x")
ax_skel.set_zlabel("y (up)")
ax_skel.legend(loc="upper left")

text_box = TextBox(fig.add_axes((0.08, 0.03, 0.05, 0.04)), "DOF (0-%d) " % (DOF_no - 1), initial=str(joint))
frame_slider = Slider(fig.add_axes((0.25, 0.035, 0.5, 0.03)), "Frame ", 0, frames_no - 1, valinit=0, valstep=1)
frame = 0
playing = False
align_root = False  # draw the SNN skeleton on the ground-truth root, to compare posture without the root drift
mocap_fps = 120
last_tick = 0.0  # wall-clock time of the previous playback tick
timer = fig.canvas.new_timer(interval=33)


def draw_skeleton() -> None:
    rc_shown = rc_pos_aligned if align_root else rc_pos
    for pos, bone_lines, points in ((gt_pos[frame], gt_bones, gt_points), (rc_shown[frame], rc_bones, rc_points)):
        bone_lines.set_segments([(pos[c], pos[p]) for c, p in bones])
        points.set_data_3d(*pos.T)
    bone = joints[dof_to_bone[joint]]  # the bone a DOF rotates is drawn from its joint to its child's
    end = bone.children[0].name if bone.children else bone.name
    sel = rc_shown[frame][[joint_names.index(bone.name), joint_names.index(end)]]
    selected_bone.set_data_3d(*sel.T)
    root = gt_pos[frame][0]
    ax_skel.set_xlim(root[0] - half_width, root[0] + half_width)
    ax_skel.set_ylim(root[1] - half_width, root[1] + half_width)
    ax_skel.set_title(f"Skeleton, frame {frame} / {frames_no - 1}" + (" (SNN root-aligned)" if align_root else ""))


def show_frame(f: int) -> None:
    global frame
    frame = f % frames_no
    for cursor in frame_cursors:
        cursor.set_xdata([frame, frame])
    draw_skeleton()
    frame_slider.eventson = False  # keep the slider in sync without re-triggering on_slide
    frame_slider.set_val(frame)
    frame_slider.eventson = True
    fig.canvas.draw_idle()


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
    ax_tgt.set_title(f"Targets, DOF {joint} ({dof_to_bone[joint]})")
    ax_ang.set_title(f"Joint angles, DOF {joint} ({dof_to_bone[joint]})")
    text_box.eventson = False  # keep the box in sync with arrow-key navigation without re-triggering on_submit
    text_box.set_val(str(joint))
    text_box.eventson = True
    draw_skeleton()
    fig.canvas.draw_idle()


def on_submit(text: str) -> None:
    try:
        show_joint(int(text))
    except ValueError:
        print(f"Not a DOF index: {text!r}")


def on_slide(value: float) -> None:
    show_frame(int(value))


def on_tick() -> None:
    global last_tick
    now = time.perf_counter()
    step = round((now - last_tick)*mocap_fps)  # advance by elapsed time so playback is real-time even if redraws are slow
    if step > 0:
        last_tick = now
        show_frame(frame + step)


def on_key(event) -> None:
    global playing, last_tick, align_root
    if text_box.capturekeystrokes:  # arrows move the cursor while typing in the box
        return
    if event.key == "right":
        show_joint(joint + 1)
    elif event.key == "left":
        show_joint(joint - 1)
    elif event.key == ".":
        show_frame(frame + 1)
    elif event.key == ",":
        show_frame(frame - 1)
    elif event.key == "r":
        align_root = not align_root
        draw_skeleton()
        fig.canvas.draw_idle()
    elif event.key == " ":
        playing = not playing
        if playing:
            last_tick = time.perf_counter()
            timer.start()
        else:
            timer.stop()


text_box.on_submit(on_submit)
frame_slider.on_changed(on_slide)
timer.add_callback(on_tick)
fig.canvas.mpl_connect("key_press_event", on_key)
show_joint(joint)
show_frame(frame)

if gif_path:
    gif_fps = 30  # GIFs can't play at the mocap's 120 fps, so this is 4x slow motion
    FuncAnimation(fig, show_frame, frames=frames_no).save(gif_path, writer=PillowWriter(fps=gif_fps), dpi=60)
else:
    plt.show()
