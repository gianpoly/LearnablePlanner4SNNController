import numpy as np


class TargetEventDetector:
    """
    Decide, per DOF, when the SNN is done with its current target and a new one should be planned.
    A DOF fires once at least `dwell` frames have passed since its last event and the joint has either arrived at the
    target or is no longer moving towards it (stalled, reversed or overshot), or after `max_hold` frames regardless
    (the SNN approaches targets asymptotically, so it may take very long to arrive). All DOFs fire on the first frame.
    """
    def __init__(self, DOF_no: int, dwell: int = 5, max_hold: int = 10, arrival_frac: float = 0.5, abs_tol: float = 0.5) -> None:
        """
        :param DOF_no: Number of DOFs
        :param dwell: Minimum number of frames between two events of the same DOF
        :param max_hold: Maximum number of frames between two events of the same DOF
        :param arrival_frac: The joint has arrived once the error is below this fraction of the error at segment start
        :param abs_tol: Arrival tolerance floor (deg), for small segments
        """
        self.dwell = dwell
        self.max_hold = max_hold
        self.arrival_frac = arrival_frac
        self.abs_tol = abs_tol
        self.last_event = np.full(DOF_no, -dwell)
        self.seg_amp = np.zeros(DOF_no)
        self.prev_angles = None

    def detect(self, frame_ind: int, angles: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        :param frame_ind: Current frame
        :param angles: Current joint angles (deg)
        :param targets: Currently held targets (deg)
        :return: (np.ndarray) boolean mask of the DOFs that need a new target
        """
        if self.prev_angles is None:
            self.prev_angles = angles.copy()
            return np.ones(angles.shape[0], dtype=bool)

        vel = angles - self.prev_angles
        self.prev_angles = angles.copy()
        err = targets - angles
        arrived = np.abs(err) <= np.maximum(self.abs_tol, self.arrival_frac*self.seg_amp)
        not_approaching = vel*err <= 0
        held = frame_ind - self.last_event
        return ((held >= self.dwell) & (arrived | not_approaching)) | (held >= self.max_hold)

    def start_segments(self, fired: np.ndarray, frame_ind: int, angles: np.ndarray, targets: np.ndarray) -> None:
        """
        Register the new targets of the DOFs that fired
        """
        self.last_event[fired] = frame_ind
        self.seg_amp[fired] = np.abs(targets - angles)[fired]
