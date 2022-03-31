import numpy as np


def angular_standard(theta):
    while theta < 0:
        theta += (2 * np.pi)
    while theta >= (2 * np.pi):
        theta -= (2 * np.pi)
    return theta


def Phi_body_angular_map(b, N):
    # 0 for North
    body_baseline = np.pi * 3 / 2  # Left
    angular_delta = np.pi * 2 / N
    return angular_standard(body_baseline + b * angular_delta)


def prior_function(N, prior_type='uniform'):
    if N <= 0:
        raise ValueError('Invalid number of body parts!')

    if prior_type in ['uniform', 'uni', 'u']:
        f = np.ones(N)
    else:
        raise ValueError('Prior type not in the list!')

    return 1. * f / np.sum(N)


