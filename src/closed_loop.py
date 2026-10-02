import time
import numpy as np

from RegressorHelpers import RegressorNet, MotionSequenceDataset, query_regressor
from controllers import AgentController
from dataset_utils import estim_deriv
from event_utils import TargetEventDetector

deriv_window = 10  # same window as dataset_utils.estim_deriv (used to build the planner's training inputs)


def run_closed_loop(dataset: MotionSequenceDataset, net: RegressorNet, specs_list: list, event_driven: bool = True,
                    use_oracle: bool = False, dwell: int = 5, max_hold: int = 10, arrival_frac: float = 0.5,
                    fps: int = 120, dt: float = 0.1, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reproduce the motion sequence in closed loop: every frame, the SNN's own state is fed back into the planner, and
    one AgentController tracks the planned targets

    :param dataset: Motion sequence dataset providing the initial state, the GT targets and the normalization factors
    :param net: Trained planner
    :param specs_list: SNN controller specs (see optimal_search_utils.read_controller_specs)
    :param event_driven: Re-plan a DOF's target only when the SNN reaches it or stops approaching it (else every frame)
    :param use_oracle: Take new targets from the GT targets of the current frame instead of the planner
    :param dwell: Minimum frames between two re-plans of the same DOF (event-driven only)
    :param max_hold: Maximum frames between two re-plans of the same DOF (event-driven only)
    :param arrival_frac: A DOF has arrived once its error is below this fraction of the error when its target was set
    :param fps: Frame rate of the motion sequence
    :param dt: SNN integration timestep (ms)
    :param verbose: Print progress every 50 frames
    :return: frames x 2*DOF angles (deg) + normalized speeds, frames x DOF targets (deg), frames x DOF re-plan mask
    """
    norm_fact = dataset.norm_fact
    derivs_norm_fact = dataset.derivs_norm_fact
    DOF_no = int(dataset.X.shape[1]/2)
    seq_frame_len = dataset.X.shape[0]

    steps_per_frame = 1000/(fps*dt)  # 83.33 SNN steps per mocap frame
    init_angles = norm_fact*dataset.X[0, 0:DOF_no]  # SNN works in degrees, dataset X is normalized
    SNN_char_ctrlr = AgentController(dt, init_angles, specs_list=specs_list)

    reconstr_frame_seq = np.zeros((seq_frame_len, dataset.X.shape[1]))  # angles (deg) + normalized speeds
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
                new_targets = norm_fact*dataset.Y[frame_ind, :]
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

        if verbose and frame_ind % 50 == 0:
            print(f"Frame {frame_ind}/{seq_frame_len} ({time.time() - start:.1f} s)")

    return reconstr_frame_seq, reconstr_targets, reconstr_events
