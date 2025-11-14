import os
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from pyprintf import sprintf
from scipy.signal import resample_poly, butter, sosfiltfilt, savgol_filter, welch


# ---------- Plot helpers (PPG) ----------
def paint_ppg_spectrum_freq_domain(f, Pxx, hr_band=(0.8, 3.0), f_hr=None):
    """
    Plot PPG frequency spectrum. If f_hr is provided, use it directly.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(f, Pxx, lw=2, label="PPG spectrum")
    plt.axvspan(hr_band[0], hr_band[1], color='orange', alpha=0.3, label="HR band")

    if f_hr is None:
        m = (f >= hr_band[0]) & (f <= hr_band[1])
        if np.any(m):
            f_hr = f[m][np.argmax(Pxx[m])]

    if f_hr is not None:
        plt.axvline(f_hr, color='red', linestyle='--', label=f'Peak: {f_hr:.2f} Hz')
        xt = list(plt.xticks()[0]) + [f_hr]
        plt.xticks(xt, [*(f"{x:.0f}" for x in xt[:-1]), f"{f_hr:.2f}*"])

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [a.u./Hz]")
    plt.title("PPG Frequency Spectrum")
    plt.xlim(0, 10)  # PPG 的有效带宽通常 < 10 Hz
    plt.legend()
    plt.tight_layout()
    plt.show()


def paint_ppg_time_domain(ppg, fs, file_path, ch, sqi=None, smooth=True,
                          window_length=21, polyorder=3, max_time=15, win_label=None):
    """
    Plot PPG in time domain. Show only the first `max_time` seconds and optional SQI in title.
    """
    ppg = np.asarray(ppg, dtype=float)
    t = np.arange(len(ppg)) / float(fs) if len(ppg) > 0 else np.array([0.0])

    y = ppg
    label = "PPG"

    if max_time is not None and len(t) > 0:
        mask = t <= max_time
        t = t[mask]
        y = y[mask]

    plt.figure(figsize=(10, 4))
    plt.plot(t, y, lw=2, label=label)
    plt.xlabel("Time [s]")
    plt.ylabel("PPG [a.u.]")

    title = "PPG Time Series"
    if sqi is not None:
        if win_label is not None:
            title += f" (SQI = {sqi:.3f}) [{win_label}]"
        else:
            title += f" (SQI = {sqi:.3f})"
            title += sprintf(", whose channel is %s", ch)

    plt.title(title)
    if len(t) > 0:
        plt.xlim(t[0], t[-1])

    plt.legend()
    plt.tight_layout()

    father_folder = os.path.basename(os.path.dirname(file_path))
    save_dir = os.path.join(os.path.dirname(" "), "./data/ppg_reports", father_folder)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"PPG_Time_Domain_channel{ch}.png")
    plt.savefig(save_path, dpi=800)
    plt.show()

    print(f"Saved to: {save_path}")


def preprocess_ppg(ppg, fs_in,
                   target_fs=50,
                   bp_band=(0.5, 8.0),   # 生理合理带：去除基线漂移(<0.5 Hz)和高频毛刺(>8 Hz)
                   detrend_medwin=0.5,   # 中值滤波窗口（秒），None 关闭
                   butter_order=4):
    """
    返回: ppg_filt (100 Hz), fs_out=100.0
    步骤:
      1) 统一采样率到 target_fs（上/下采样，resample_poly 内置抗混叠 FIR）
      2) Butterworth 带通 (bp_band)
      3) 可选中值滑窗去趋势（抵抗缓慢基线与部分运动伪迹）
    """
    x = np.asarray(ppg, dtype=float)
    fs_in = float(fs_in)
    fs_out = float(target_fs)

    # 1) 统一采样率
    if abs(fs_in - fs_out) < 1e-9:
        x_rs = x
    else:
        frac = Fraction(fs_out / fs_in).limit_denominator(1000)
        up, down = frac.numerator, frac.denominator
        x_rs = resample_poly(x, up, down)

    # 2) 带通滤波
    nyq = 0.5 * fs_out
    lo = max(0.001, bp_band[0] / nyq)
    hi = min(0.999, bp_band[1] / nyq)

    if hi <= lo:  # 极端保护
        return x_rs.astype(float), fs_out

    sos = butter(butter_order, [lo, hi], btype='bandpass', output='sos')
    x_bp = sosfiltfilt(sos, x_rs)

    return x_bp.astype(float), fs_out


# ---------- Core: compute SQI for PPG ----------
def compute_ppg_sqi(ppg, file_path, fs_in, ch,
                    hr_band=(0.8, 7.0),
                    total_band=(0.0, 10.0),
                    use_harmonic=False, harmonic_bw=0.3,
                    bp_band=(0.5, 7.0),
                    detrend_medwin=0.5,
                    do_plot=True):
    """
    先基于人体信号预处理，再计算 SQI。
    """
    # 预处理：统一到 100 Hz + 带通 + 去趋势
    ppg_filt, fs = preprocess_ppg(
        ppg, fs_in, target_fs=50,
        bp_band=bp_band,
        detrend_medwin=detrend_medwin
    )

    # Welch PSD
    x = ppg_filt - np.mean(ppg_filt)
    nperseg = min(1024, len(x)) if len(x) >= 16 else len(x)

    if nperseg < 8:
        if do_plot:
            paint_ppg_time_domain(ppg_filt, fs, sqi=0.0, tag="short")
        return 0.0, 0.0, (np.array([0.0]), np.array([0.0]))

    f, Pxx = welch(x, fs, nperseg=nperseg)

    # HR 主带找峰
    m_hr = (f >= hr_band[0]) & (f <= hr_band[1])
    if not np.any(m_hr):
        if do_plot:
            paint_ppg_time_domain(ppg_filt, fs, sqi=0.0, tag="no HR")
            paint_ppg_spectrum_freq_domain(f, Pxx, hr_band=hr_band, f_hr=None)
        return 0.0, 0.0, (f, Pxx)

    f_hr = f[m_hr][np.argmax(Pxx[m_hr])]

    # 带功率
    def band_power(f1, f2):
        m = (f >= f1) & (f <= f2)
        return np.trapezoid(Pxx[m], f[m]) if np.any(m) else 0.0

    P_main = band_power(max(hr_band[0], f_hr - 0.2), min(hr_band[1], f_hr + 0.2))
    P_total = band_power(total_band[0], total_band[1])
    P = P_main

    if use_harmonic:
        f2 = 2.0 * f_hr
        P += band_power(max(total_band[0], f2 - harmonic_bw),
                        min(total_band[1], f2 + harmonic_bw))

    sqi = float(np.clip(P / P_total if P_total > 0 else 0.0, 0.0, 1.0))

    # 画图（时域只显示前 15 s，标题带 SQI）
    if do_plot:
        paint_ppg_spectrum_freq_domain(f, Pxx, hr_band=hr_band, f_hr=f_hr)
        paint_ppg_time_domain(ppg_filt, fs, file_path=file_path, ch=ch, sqi=sqi, max_time=15)
        #paint_ppg_time_domain(ppg, fs, file_path=file_path, ch=ch, sqi=sqi, max_time=15)

    return sqi, f_hr  # , (f, Pxx)
