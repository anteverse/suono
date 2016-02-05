import math
import numpy as np
from scipy.signal import convolve


def find_peaks(fft_set, sign='+', alpha=0.15, threshold=2.0):
    """
    Method to find peaks on a fft set.
    
    :type threshold: object
    :param fft_set:
    :param sign:
    :param alpha: noise threshold
    :param threshold:
    :return:
    """

    if sign == '-':
        fft_set = -fft_set

    # Get derivative
    derivation_vector = [1, 0, -1]
    d_fft_set = convolve(fft_set, derivation_vector, 'same')

    # Checking for sign-flipping and derivative
    _sign = np.sign(d_fft_set)
    d_sign = convolve(_sign, derivation_vector, 'valid')

    candidates = np.where(d_fft_set > 0)[0] + (len(derivation_vector) - 1)

    peaks = sorted(set(candidates).intersection(np.where(d_sign == -2)[0] + 1))

    # Noise remover
    peaks = np.array(peaks)[fft_set[peaks] > alpha]

    return clean_adjacent_points(peaks, fft_set, float(threshold))


def clean_adjacent_points(arr, measure, threshold=2.0):
    """
    Returns an array without duplicates
    :param arr: array with duplicates
    :param measure: feature used to compare and determine duplicates
    :param threshold: indexes window to consider the duplicates.
        Value may vary depending on the resolution or size of the array.
    :return: The entry array where duplicates are filtered out.
    """
    _arr = arr
    for i in range(len(arr)-1):
        if math.fabs(arr[i] - arr[i+1]) <= threshold:
            if measure[arr[i]] - measure[arr[i+1]] <= 0.0:
                np.put(_arr, [i], [-1])
            else:
                np.put(_arr, [i + 1], [-1])
    return _arr[np.where(_arr > 0)[0]]
