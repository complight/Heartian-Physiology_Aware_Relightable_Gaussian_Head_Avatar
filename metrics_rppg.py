import numpy as np
from scipy import signal
from copy import deepcopy
from scipy.signal import butter
import scipy
from rppg_toolbox import _detrend, _bandpass, _next_power_of_2, fft_hr

def sliding_windows(sig, fps, win_sec=10, overlap=0.5):
    win_size = int(win_sec * fps)
    step = int(win_size * (1 - overlap))
    windows = []
    for start in range(0, len(sig) - win_size + 1, step):
        windows.append(sig[start:start + win_size])
    return windows

def calculate_snr(signal_window, fps, target_hr):
    N = _next_power_of_2(len(signal_window))
    freqs, spectrum = signal.periodogram(signal_window, fs=fps, nfft=N, detrend=False)

    target_freq = target_hr / 60
    second_target_freq = 2 * target_freq
    band = 0.1 # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    freq_mask = (freqs >= 0.75) & (freqs <= 3.0)
    signal_mask = (freqs >= target_freq - band) & (freqs <= target_freq + band)
    second_signal_mask = (freqs >= second_target_freq - band) & (freqs <= second_target_freq + band)
    noise_mask = freq_mask & ~signal_mask & ~second_signal_mask

    signal_power = np.sum(spectrum[signal_mask])
    second_signal_power = np.sum(spectrum[second_signal_mask])
    noise_power = np.sum(spectrum[noise_mask])
    if noise_power == 0:
        return np.nan
    return 10 * np.log10((signal_power+second_signal_power) / noise_power)

def calculate_macc(pred_window, gt_window):
    pred = np.squeeze(deepcopy(pred_window))
    gt = np.squeeze(deepcopy(gt_window))
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]
    lags = np.arange(0, len(pred) - 1, 1)
    tlcc_list = []
    for lag in lags:
        cross_corr = np.abs(np.corrcoef(pred, np.roll(gt, lag))[0][1])
        tlcc_list.append(cross_corr)
    return max(tlcc_list)

def evaluate_rppg(gt_signal, pred_signal, fps=30, win_sec=10, overlap=0.5, band=(0.75, 3.0), preprocess_gt=False, use_sliding_window=False):

    if preprocess_gt:
        gt_proc = _detrend(gt_signal, 100)
        gt_proc = _bandpass(gt_proc, fps)
    else:
        gt_proc = gt_signal

    # already filtered before entering this function
    pred_proc = pred_signal

    if use_sliding_window:
        gt_windows = sliding_windows(gt_proc, fps, win_sec, overlap)
        pred_windows = sliding_windows(pred_proc, fps, win_sec, overlap)
    else:
        gt_windows = [gt_proc]
        pred_windows = [pred_proc]

    gt_hr_all, pred_hr_all, snr_all, macc_all = [], [], [], []

    for gt_w, pred_w in zip(gt_windows, pred_windows):
        if len(gt_w) < 9:
            continue

        gt_hr = fft_hr(gt_w, fps, band[0], band[1])
        pred_hr = fft_hr(pred_w, fps, band[0], band[1])

        if np.isnan(gt_hr) or np.isnan(pred_hr):
            continue

        gt_hr_all.append(gt_hr)
        pred_hr_all.append(pred_hr)
        snr_all.append(calculate_snr(pred_w, fps, gt_hr))
        macc_all.append(calculate_macc(pred_w, gt_w))

    gt_hr_all = np.array(gt_hr_all)
    pred_hr_all = np.array(pred_hr_all)

    # filter out the nan values
    snr_arr = np.array(snr_all)
    macc_arr = np.array(macc_all)
    snr_mean = np.nanmean(snr_arr)
    macc_mean = np.nanmean(macc_arr)

    mae = np.mean(np.abs(pred_hr_all - gt_hr_all))
    rmse = np.sqrt(np.mean((pred_hr_all - gt_hr_all) ** 2))
    mape = np.mean(np.abs((pred_hr_all - gt_hr_all) / gt_hr_all)) * 100

    if len(pred_hr_all) < 2:
        pearson_str = "N/A"
        print("Metrics are calculated on the full sequence of the selected subject.")
        print("Please turn on the use_sliding_window to obtain a meaning Pearson metric.")
    else:
        pearson = np.corrcoef(pred_hr_all, gt_hr_all)[0, 1]
        pearson_str = f"{pearson:.3f}"

    print("========== rPPG Evaluation ==========")
    print(f"Windows used : {len(gt_hr_all)}")
    print(f"MAE          : {mae:.3f} bpm")
    print(f"RMSE         : {rmse:.3f} bpm")
    print(f"MAPE         : {mape:.3f} %")
    print(f"Pearson      : {pearson_str}")
    print(f"SNR          : {snr_mean:.3f} dB")
    print(f"MACC         : {macc_mean:.3f}")
    print("======================================")

    return {
        "MAE": mae, "RMSE": rmse, "MAPE": mape,
        "Pearson": pearson_str, "SNR": snr_mean, "MACC": macc_mean
    }