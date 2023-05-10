import numpy as np


def v_theta2cur(vel):
    """
    Angular speed to current conversion
    :param vel: Speed (float)
    :return: Current vector I for the two PPC_speed neurons (vector)
    """
    I = np.zeros(2)
    if vel > 0:
        I[0] = min(3 * vel, 1)
    else:
        I[1] = min(-3 * vel, 1)

    return I


def ang2cur(ang):
    """
    Angle to current conversion
    :param ang: Angle (float)
    :return: Current value I (float)
    """
    if ang > 0:
        I = 0.05*ang
    else:
        I = 0
    return I


def PSI_wgt_clip(PSI_factor, max_wgt):
    """
    Clip PSI modulation factor in the required 0-1 range
    :param PSI_factor: Un-clipped PSI factor calculated based on synaptic activity
    :param max_wgt: Max weight of the synapse
    :return: Clipped PSI factor
    """
    min_wgt = 1.5
    if 1 - PSI_factor < 0:
        return min_wgt/max_wgt  # PSI_factor*max_wgt=min_wgt
    elif 1 - PSI_factor > 0:
        return max(min_wgt/max_wgt, 1 - PSI_factor)
    else:
        return 1
