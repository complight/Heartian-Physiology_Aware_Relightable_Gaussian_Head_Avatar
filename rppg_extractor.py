import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy import signal
from scipy.sparse import spdiags
from scipy.signal import butter
import scipy

import mediapipe as mp
from metrics_rppg import evaluate_rppg
from rppg_toolbox import _detrend, _bandpass, _next_power_of_2
import os
import scipy.io as sio
import json

def extract_rppg_from_rendered_images(
    signal: np.ndarray,
    split_indices: list,
    image_dir: str,
    fps: int = 30,
    output_dir: str = None,
    source_path: str = None,
    split_name: str = "full",
    filtered_full: np.ndarray = None,   
    filtered_gt_full: np.ndarray = None, 
    full_split_indices: list = None,
    test_indices: list = None,
):
    output_dir = Path(output_dir) if output_dir else Path(f"rppg_eval_{split_name}")
    output_dir.mkdir(parents=True, exist_ok=True)


    region_G_log = signal

    np.savetxt(output_dir / "rppg_raw.txt", region_G_log)

    if split_name == "full":
        # FFT + full video evaluate
        detrended = _detrend(region_G_log, 100)
        filtered  = _bandpass(detrended, fps, low=0.75, high=3.0)
        np.savetxt(output_dir / "rppg_detrended.txt", detrended)
        np.savetxt(output_dir / "rppg_filtered.txt",  filtered)

        _plot_signal(region_G_log, output_dir / "rppg_raw.png", title=f"[{split_name}] Raw Learned rPPG Signal")
        _plot_signal(detrended, output_dir / "rppg_detrended.png", title=f"[{split_name}] Detrended Learned rPPG Signal")
        _plot_signal(filtered, output_dir / "rppg_filtered.png", title=f"[{split_name}] Filtered Learned rPPG Signal")

        fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)

        axes[0].plot(np.arange(250) / fps, region_G_log[-250:], linewidth=0.9)
        axes[0].set_title('Ours PPG — Last 250 Frames (Raw)', fontsize=11)
        axes[0].set_ylabel('Amplitude', fontsize=9)
        axes[0].grid(alpha=0.12)

        axes[1].plot(np.arange(250) / fps, detrended[-250:], linewidth=0.9)
        axes[1].set_title('Ours PPG — Last 250 Frames (Detrended)', fontsize=11)
        axes[1].set_ylabel('Amplitude', fontsize=9)
        axes[1].grid(alpha=0.12)

        axes[2].plot(np.arange(250) / fps, filtered[-250:], linewidth=1.3)
        axes[2].set_title('Ours PPG — Last 250 Frames (Filtered)', fontsize=11)
        axes[2].set_ylabel('Amplitude', fontsize=9)
        axes[2].set_xlabel('Time (s)', fontsize=9)
        axes[2].grid(alpha=0.12)

        plt.tight_layout(pad=1.5)
        plt.savefig(output_dir / "ours_last250_raw_vs_filtered.png", bbox_inches='tight'); plt.close()


        result = {
            "raw_signal":      region_G_log,
            "detrended":       detrended,
            "filtered_signal": filtered,
            "hr_bpm":          None,
            "metrics":         None,
        }

        hr_bpm = _estimate_hr_fft(filtered, fps=fps, output_path=output_dir / "rppg_fft.png", split_name=split_name)
        result["hr_bpm"] = hr_bpm

        # evaluation metrics
        full_metrics, gt_filtered = _evaluate_against_gt_full(raw=region_G_log, detrended=detrended, pred_signal=filtered, source_path=source_path, fps=fps, output_dir=output_dir, split_name=split_name, test_indices=test_indices)
        result["metrics"] = full_metrics
        filtered_gt = gt_filtered

    else:
        # train/test: point-wise evaluation
        result = {"raw_signal": region_G_log, "metrics": None}
        filtered, filtered_gt = None, None
        if source_path is not None and Path(source_path).exists() and filtered_full is not None:
            metrics = _evaluate_against_gt_pointwise(
                filtered_full=filtered_full,
                filtered_gt_full=filtered_gt_full,
                full_split_indices=full_split_indices,
                cur_split_indices=split_indices,
                output_dir=output_dir,
                split_name=split_name,
            )
            result["metrics"] = metrics

    print(f"\n[{split_name}] rPPG extraction done → {output_dir}")
    return result, filtered, filtered_gt  # returned filtered results for train/test evaluation

def _evaluate_against_gt_full(raw, detrended, pred_signal, source_path, fps, output_dir, split_name, test_indices):
    
    if "UBFC-rPPG" in source_path:
        gt_path=os.path.join(source_path, "ground_truth.txt")
        with open(gt_path, 'r') as f:
            lines = f.readlines()
        gt_ppg_full = np.array([float(x) for x in lines[0].split()])
    elif "MMPD" in source_path:
        f = sio.loadmat(os.path.join(source_path, os.path.basename(source_path) + '.mat'), variable_names=['GT_ppg'])
        gt_ppg_full = f['GT_ppg'].reshape(-1)
        skipped_path = os.path.join(source_path, 'skipped_indices.npy')
        if os.path.exists(skipped_path):
            skipped_indices = np.load(skipped_path)
            print(f"🔧 Skip MMPD artifact data index: {skipped_indices}")
            if len(skipped_indices) > 0:
                gt_ppg_full = np.delete(gt_ppg_full, skipped_indices)

    elif "PURE" in source_path:
        gt_path = os.path.join(source_path, os.path.basename(source_path) + '.json')
        with open(gt_path, 'r') as f:
            labels = json.load(f)
        waves = np.array([label["Value"]["waveform"] for label in labels["/FullPackage"]])
        target_length = len(raw)
        gt_ppg_full = np.interp(np.linspace(1, waves.shape[0], target_length), np.linspace(1, waves.shape[0], waves.shape[0]), waves)

    gt_filtered = _bandpass(_detrend(gt_ppg_full, 100), fps)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(np.arange(250) / fps, gt_ppg_full[-250:], linewidth=0.9)
    axes[0].set_title('GT PPG — Last 250 Frames (Raw)', fontsize=11)
    axes[0].set_ylabel('Amplitude', fontsize=9)
    # axes[0].grid(alpha=0.12)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')

    axes[1].plot(np.arange(250) / fps, _detrend(gt_ppg_full, 100)[-250:], linewidth=0.9)
    axes[1].set_title('GT PPG — Last 250 Frames (Detrended)', fontsize=11)
    axes[1].set_ylabel('Amplitude', fontsize=9)
    # axes[1].grid(alpha=0.12)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')

    axes[2].plot(np.arange(250) / fps, gt_filtered[-250:], linewidth=2)
    axes[2].set_title('GT PPG — Last 250 Frames (Filtered)', fontsize=11)
    axes[2].set_ylabel('Amplitude', fontsize=9)
    axes[2].set_xlabel('Time (s)', fontsize=9)
    # axes[2].grid(alpha=0.12)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')

    plt.tight_layout(pad=1.5)
    plt.savefig(output_dir / "gt_last250_raw_vs_filtered.png", bbox_inches='tight'); plt.close()

    def _norm(s):
        return (s - s.mean()) / (s.std() + 1e-6)

    pred_n = _norm(pred_signal)
    gt_n   = _norm(gt_filtered)

    plt.figure(figsize=(14, 4))
    plt.plot(pred_n, label="Ours (filtered)", alpha=0.8)
    plt.plot(gt_n,   label="GT (filtered)",   alpha=0.8)
    plt.legend(); plt.grid(True)
    plt.title(f"[{split_name}] Ours vs GT")
    plt.tight_layout()
    plt.savefig(output_dir / "ours_vs_gt.png"); plt.close()

    # last 250 frames detail plot
    n = min(250, len(pred_n), len(gt_n))
    plt.figure(figsize=(14, 4))
    plt.plot(pred_n[-n:], label="Ours (filtered)", alpha=0.8)
    plt.plot(gt_n[-n:],   label="GT (filtered)",   alpha=0.8)
    plt.legend(); plt.grid(True)
    plt.title(f"[{split_name}] Ours vs GT — Last {n} Frames")
    plt.tight_layout()
    plt.savefig(output_dir / "ours_vs_gt_last250.png"); plt.close()

    # GT vs Ours: raw / detrended / filtered
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    total_frames = len(raw)
    last_n_start = total_frames - n
    test_in_window = [t for t in test_indices if t >= last_n_start]
    test_x = [(t - last_n_start) / fps for t in test_in_window]

    axes[0].plot(np.arange(n) / fps, _norm(gt_ppg_full[-n:]),  color='darkorange', linewidth=1.2, label='GT',   alpha=0.65)
    axes[0].plot(np.arange(n) / fps, _norm(raw[-n:]),           color='steelblue',  linewidth=2, label='Ours', alpha=0.85)
    raw_norm = _norm(raw[-n:])
    test_idx_local = [t - last_n_start for t in test_in_window]
    # axes[0].scatter(test_x, raw_norm[test_idx_local], color='lightcoral', marker='x', s=20, linewidths=0.8, zorder=5, label='Test frames')
    axes[0].set_title('Raw — Last 250 Frames', fontsize=11)
    axes[0].set_ylabel('Amplitude (norm)', fontsize=9)
    axes[0].legend(fontsize=8)
    # axes[0].grid(alpha=0.3)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')

    axes[1].plot(np.arange(n) / fps, _norm(_detrend(gt_ppg_full, 100)[-n:]), color='darkorange', linewidth=0.9, label='GT',   alpha=0.85)
    axes[1].plot(np.arange(n) / fps, _norm(detrended[-n:]),              color='steelblue',  linewidth=0.9, label='Ours', alpha=0.85)
    detrended_norm = _norm(detrended[-n:])
    axes[1].scatter(test_x, detrended_norm[test_idx_local], color='lightcoral', marker='x', s=20, linewidths=0.8, zorder=5, label='Test frames')
    axes[1].set_title('Detrended — Last 250 Frames', fontsize=11)
    axes[1].set_ylabel('Amplitude (norm)', fontsize=9)
    axes[1].legend(fontsize=8)
    # axes[1].grid(alpha=0.3)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')

    axes[2].plot(np.arange(n) / fps, gt_n[-n:],   color='darkorange', linewidth=1.2, label='GT',   alpha=0.65)
    axes[2].plot(np.arange(n) / fps, pred_n[-n:],  color='steelblue',  linewidth=2, label='Ours', alpha=0.85)
    # axes[2].scatter(test_x, pred_n[-n:][test_idx_local], color='lightcoral', marker='x', s=20, linewidths=0.8, zorder=5, label='Test frames')
    axes[2].set_title('Filtered — Last 250 Frames', fontsize=11)
    axes[2].set_ylabel('Amplitude (norm)', fontsize=9)
    axes[2].set_xlabel('Time (s)', fontsize=9)
    axes[2].legend(fontsize=8)
    # axes[2].grid(alpha=0.3)

    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')

    plt.suptitle(f"[{'Full'}] GT vs Ours — Raw / Detrended / Filtered", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "gt_ours_all_stages.png", bbox_inches='tight'); plt.close()

    full_metrics = evaluate_rppg(gt_signal=gt_filtered, pred_signal=pred_signal, fps=fps, win_sec=10, overlap=0.5, band=(0.75, 3.0), use_sliding_window=False)
    return full_metrics, gt_filtered

def _evaluate_against_gt_pointwise(filtered_full, filtered_gt_full, full_split_indices, cur_split_indices, output_dir, split_name):
    # Map from time frame index to the current split's indices
    t2idx = {t: i for i, t in enumerate(full_split_indices)}
    indices = np.array([t2idx[t] for t in sorted(cur_split_indices) if t in t2idx])

    pred_points = filtered_full[indices]
    gt_points   = filtered_gt_full[indices]

    def _norm(s):
        return (s - s.mean()) / (s.std() + 1e-6)

    pred_n = _norm(pred_points)
    gt_n   = _norm(gt_points)

    mae     = np.mean(np.abs(pred_n - gt_n))
    rmse    = np.sqrt(np.mean((pred_n - gt_n) ** 2))
    pearson = np.corrcoef(pred_n, gt_n)[0, 1]

    print(f"[{split_name}] Points evaluated : {len(indices)}")
    print(f"[{split_name}] MAE              : {mae:.6f}")
    print(f"[{split_name}] RMSE             : {rmse:.6f}")
    print(f"[{split_name}] Pearson          : {pearson:.4f}")

    np.savetxt(output_dir / "pred_points_filtered.txt", pred_n)
    np.savetxt(output_dir / "gt_points_filtered.txt",   gt_n)

    fig, axes = plt.subplots(2, 1, figsize=(16, 6))
    axes[0].plot(np.arange(len(filtered_full)), filtered_full,
                 color='steelblue', alpha=0.3, linewidth=0.8, label='Full pred')
    axes[0].scatter(indices, pred_n, s=4, color='steelblue', label=f'{split_name} pred')
    axes[0].scatter(indices, gt_n,   s=4, color='tomato',    label=f'{split_name} GT')
    axes[0].set_ylabel("Signal"); axes[0].legend(fontsize=8); axes[0].grid(True)
    axes[0].set_title(f"[{split_name}] Point-wise  Pearson={pearson:.3f}  MAE={mae:.5f}  RMSE={rmse:.5f}")

    axes[1].scatter(np.arange(len(indices)), pred_n - gt_n, s=4, color='purple', alpha=0.6)
    axes[1].axhline(0, color='gray', linewidth=0.8)
    axes[1].set_xlabel("Point index"); axes[1].set_ylabel("Pred - GT"); axes[1].grid(True)
    axes[1].set_title(f"[{split_name}] Residuals")

    plt.tight_layout()
    plt.savefig(output_dir / "pointwise_comparison.png"); plt.close()

    return {"mae": mae, "rmse": rmse, "pearson": pearson}


# ── Signal plots helpers ───

def _estimate_hr_fft(filtered_signal, fps, output_path=None, split_name=""):

    N = _next_power_of_2(len(filtered_signal))
    freqs, pxx = signal.periodogram(filtered_signal, fs=fps, nfft=N, detrend=False)

    valid = (freqs > 0.75) & (freqs < 3.0)
    peak_freq = freqs[valid][np.argmax(pxx[valid])]
    hr_bpm = peak_freq * 60

    print(f"[{split_name}] Estimated HR: {hr_bpm:.2f} bpm ({peak_freq:.3f} Hz)")

    if output_path:
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, pxx)
        plt.axvline(peak_freq, color='r', linestyle='--', label=f"Peak: {hr_bpm:.1f} bpm")
        plt.xlabel("Frequency (Hz)"); plt.ylabel("Power")
        plt.xlim([0, 4]); plt.legend(); plt.grid(True)
        plt.title(f"[{split_name}] FFT — HR: {hr_bpm:.2f} bpm")
        plt.tight_layout()
        plt.savefig(output_path); plt.close()

    return hr_bpm

def _plot_signal(sig, output_path, title=""):
    plt.figure(figsize=(16, 4))
    plt.plot(np.arange(len(sig)), sig)
    plt.xlabel("Timestep"); plt.ylabel("Signal")
    plt.title(title); plt.grid(True); plt.tight_layout()
    plt.savefig(output_path); plt.close()