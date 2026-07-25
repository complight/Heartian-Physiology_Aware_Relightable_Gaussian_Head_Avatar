"""POS
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). 
Algorithmic principles of remote PPG. 
IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
"""

import math

import numpy as np
from scipy import signal

from rppg_toolbox import _detrend, _process_signal


def POS_WANG(fs, RGB):
    WinSec = 1.6
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H
    BVP = _detrend(np.mat(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))

    analytic = signal.hilbert(BVP)
    phase = np.angle(analytic)
    return phase, BVP

def POS_WANG_simplified(rgb_means, fps):
        
    R, G, B = rgb_means[:, 0], rgb_means[:, 1], rgb_means[:, 2]

    S1 = R - G
    S2 = R + G - 2 * B
    alpha = S1.std() / (S2.std() + 1e-6)
    P = S1 - alpha * S2 
    
    P = _process_signal(P, fps, use_bandpass = True)
    
    # Hilbert → instantaneous phase
    analytic = signal.hilbert(P)
    phase = np.angle(analytic)
    
    return phase, P


