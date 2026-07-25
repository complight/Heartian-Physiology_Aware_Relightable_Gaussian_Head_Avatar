"""rPPG-Toolbox
Liu, X., Narayanswamy, G., Paruchuri, A., Zhang, X., Tang, J., Zhang, Y., Sengupta, S., Patel, S., Wang, Y., & McDuff, D. (2023).
rPPG-Toolbox: Deep Remote PPG Toolbox.
Advances in Neural Information Processing Systems, 36, 68485-68510.
"""

import torch
import scipy
import numpy as np
from scipy import signal
from scipy.sparse import spdiags
from scipy.signal import butter

def _detrend_torch(signal, Lambda):

    T = signal.shape[0]
    device = signal.device

    D = (torch.diag(torch.ones(T,   device=device),  diagonal=0)
       - 2 * torch.diag(torch.ones(T-1, device=device), diagonal=1)
       + torch.diag(torch.ones(T-2, device=device), diagonal=2))[:T-2, :]

    I = torch.eye(T, device=device)
    A = I + (Lambda ** 2) * D.T @ D 
    
    detrended = signal - torch.linalg.solve(A, signal)
    return detrended

def _detrend(input_signal, lambda_value):

    signal_length = len(input_signal)
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, signal_length - 2, signal_length).toarray()

    # Apply smoothing
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal
    )
    return detrended_signal

def _process_signal(signal, fs, use_bandpass = False):

    # Detrend the original signal
    detrended_signal = _detrend(signal, 100)

    # Apply bandpass filter if needed
    if use_bandpass:
        b, a = butter(1, [0.75 / fs * 2, 3.0 / fs * 2], btype='bandpass')
        filtered_signal = scipy.signal.filtfilt(b, a, np.double(detrended_signal))
        return filtered_signal

    return detrended_signal

def _bandpass(sig, fps, low=0.75, high=3.0):
    b, a = butter(1, [low / fps * 2, high / fps * 2], btype='bandpass')
    return scipy.signal.filtfilt(b, a, np.double(sig))

def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def fft_hr(signal_window, fs, low=0.75, high=3.0):

    N = _next_power_of_2(len(signal_window))
    freqs, pxx = signal.periodogram(signal_window, fs=fs, nfft=N, detrend=False)
    valid = (freqs >= low) & (freqs <= high)
    if not np.any(valid):
        return np.nan
    peak_freq = freqs[valid][np.argmax(pxx[valid])]
    return peak_freq * 60