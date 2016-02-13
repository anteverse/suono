import numpy as np


def guess_sign(y):
    """
    :param y: FFT data - regressand
    :return:
    """
    return abs(y), '+'


def prepare_data(x, y, sign):
    """
    :param x: training sample
    :param y: regressand
    :param sign: sign of the regressand
    :return:
    """
    if sign is None or (sign != '+' and sign != '-'):
        y, sign = guess_sign(y)
    if sign == '+':
        return x, np.power(y, -1)
    elif sign == '-':
        return x, -np.power(y, -1)
    else:
        print 'not implemented yet'
        raise ValueError


def gradient_descent(alpha, x, y, iterations_max, stop, regularization=0.0):
    """
    :param alpha: Training coefficient
    :param x: training sample
    :param y: regressand
    :param iterations_max: number of iterations that stops the loop
    :param stop: threshold on the cost gap between to iterations.
        when reached, the loop is stopped
    :param regularization: regularization parameter. Also known as lambda
    :return:
    """

    # training sample size
    m = len(x[:])
    # features number
    n = len(x[0])

    # prepare theta
    theta = np.ones(n)

    # little trick to help convergence
    # to be replaced by a real normalization function
    theta *= y[0] / (x[1][0]**(n-1))

    # local variables
    x_transpose = x.transpose()
    i = 0
    j_previous = 0.0
    d_j = - (2*stop + 1)

    while i < iterations_max and abs(d_j) > stop:
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y

        # Regularization part could be removed from cost
        # since gradient is not set using j's derivative
        j = (np.sum(np.power(loss, np.array([2]))) +
             regularization * (np.sum(np.power(theta, np.array([2]))) - theta[0]**2)) / (2 * m)

        r = theta
        np.put(r, [0], [0])

        gradient = np.dot(x_transpose, loss) / m + regularization * r
        theta = theta - alpha * gradient
        if not i == 0:
            d_j = (j - j_previous) / float(j_previous)

        # Differential should be negative
        # Positive differential means the cost is increasing
        if d_j > 0:
            break

        j_previous = j
        i += 1
    return theta, j_previous
