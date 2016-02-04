"""
Proof of concept of the wave detection
"""

from scipy.fftpack import fft
from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
import math


def find_peaks(Y, sign='-', alpha=-0.1, threshold=2.0):
    if sign == '+':
        Y = -Y

    # Get derivative
    derivation_vector = [1, 0, -1]
    dY = convolve(Y, derivation_vector, 'same')

    # Checking for sign-flipping and derivative
    S = np.sign(dY)
    dS = convolve(S, derivation_vector, 'valid')

    candidates = np.where(dY < 0)[0] + (len(derivation_vector) - 1)

    peaks = sorted(set(candidates).intersection(np.where(dS == 2)[0] + 1))

    # Noise remover
    peaks = np.array(peaks)[Y[peaks] < alpha]

    return clean_adjacent_points(peaks, threshold)


def clean_adjacent_points(arr, threshold=2.0):
    _arr = arr
    for i in range(len(arr)-1):
        if math.fabs(arr[i] - arr[i+1]) <= threshold:
            if Y[arr[i]] - Y[arr[i+1]] <= 0.0:
                np.place(_arr, _arr == arr[i+1], -1)
            else:
                np.place(_arr, _arr == arr[i], -1)
    return _arr[np.where(_arr > 0)[0]]


def fit_curve(x, y, xf):
    # y would be either 1 / x or 1 / x^2, we run then polyfit on y^-2
    yc = -np.power(y, -2)

    # polyfit on a degree + 1
    z = np.polyfit(x, yc, 3)
    epsilon = 0.005
    e = np.power((1 - epsilon) * np.ones(len(z)), np.array(range(len(z), 0, -1)))
    # np.multiply(z, e)
    f = np.poly1d(np.multiply(z, e))

    # shouldn't be negative so far:
    # y = - f(np.power(y, .5))


    return xf, -np.power(-f(xf), -0.5)


# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2*np.pi*x) \
    + 0.5*np.sin(50.0 * 2*np.pi*x*2) \
    + 0.33*np.sin(50.0 * 2*np.pi*x*3) \
    + 0.25*np.sin(50.0 * 2*np.pi*x*4) \
    + 0.2*np.sin(50.0 * 2*np.pi*x*5)
yf = fft(y)
yf_1 = np.abs(fft(y)[0:N/2])
yf_filtered = 2.0/N * np.abs(fft(y)[0:N/2])
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)


Y = - yf_filtered

plt.plot(xf, Y)

peaks = find_peaks(Y)

plt.scatter(xf[peaks], Y[peaks], marker='*', color='r', s=40)

# print Y[peaks]
# print fit_curve(xf[peaks], Y[peaks], xf)[1][peaks]
fitted = fit_curve(xf[peaks], Y[peaks], xf[np.where(xf > xf[peaks[0]])])
plt.plot(fitted[0][np.where(fitted[0] <= xf[peaks][-1])[0]], fitted[1][np.where(fitted[0] <= xf[peaks][-1])[0]], color='g')
plt.plot(fitted[0][np.where(fitted[0] >= xf[peaks][-1])[0]], fitted[1][np.where(fitted[0] >= xf[peaks][-1])[0]], 'g--')

plt.show()