import numpy as np

def estim_deriv(var_array):
    """
    Estimate the derivative of a given time-series array.
    The derivative of a time-series data is calculated by subtracting each time step from the previous time step.

    :param var_array: (numpy.ndarray) A 2D array with shape (time_steps, features) representing a time-series data.

    :return: (numpy.ndarray) A 2D array with the same shape as `var_array` representing the estimated derivative of
    the time-series.
    """

    der = np.zeros_like(var_array)

    estim_window = 10
    cur_ts = 1
    while cur_ts < var_array.shape[0]:
        if cur_ts >= estim_window:
            der[cur_ts, :] = var_array[cur_ts, :] - var_array[cur_ts-estim_window, :]
        else:
            der[cur_ts, :] = var_array[cur_ts, :] - var_array[0, :]
        cur_ts += 1
    return der


def zero_crossing(var_array):
    """
    This function computes the zero-crossings of an input 1D or 2D numpy array var_array. The zero-crossings are the
    points in time at which the sign of the signal changes from positive to negative or vice versa.

    :param var_array: (numpy array): The input numpy array of shape (time_steps, dimensions) where time_steps
    represents the number of time steps and dimensions represents the number of dimensions in the signal.

    :return: zcs (list of lists): A list of lists where each sublist corresponds to the zero-crossings in a given
    dimension.
    """
    zc_mask = np.zeros_like(var_array)
    zcs = [[] for _ in range(var_array.shape[1])]
    cur_ts = 1
    while cur_ts < var_array.shape[0]:
        zc_mask[cur_ts, :] = 1 * np.sign(var_array[cur_ts, :] * var_array[cur_ts - 1, :]) == -1
        zc_dims = np.where(zc_mask[cur_ts, :] == 1)[0].tolist()

        for dim in zc_dims:
            zcs[dim].append(cur_ts)

        cur_ts += 1

    return zcs


def oversample_trigger_list(lst):
    """
    This function oversamples a list of triggers by finding the average of adjacent triggers and adding them to the list.

    :param lst: (list): A list of trigger points, where each trigger point is an integer representing a time step.

    :return: The oversampled list of trigger points, where each trigger point is an integer representing a time step.

    """
    trigger_list = np.array(lst)
    extra_samples = ((trigger_list[1:] + trigger_list[:-1]) / 2).astype(int)

    return list(np.unique(np.concatenate((trigger_list, extra_samples))))