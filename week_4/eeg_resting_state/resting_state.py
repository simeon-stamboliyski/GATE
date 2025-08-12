import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

trapz = getattr(np, "trapezoid", np.trapz)

def detect_rest_epochs(
    eeg,                      
    fs,                      
    channel_names,            
    window_sec=2.0,
    step_sec=0.5,
    eyes_closed=True,
    consecutive_windows_required=5,
    freq_range=(1, 80),
    thresholds=None,
):
    """
    Detect resting-state segments in EEG based on alpha dominance, ratios,
    spectral entropy, hemispheric symmetry, high-frequency noise, and stability.
    """
    if thresholds is None:
        thresholds = {
            'alpha_ratio_min':   1.4 if eyes_closed else 0.9,
            'alpha_fraction_min': 0.35 if eyes_closed else 0.25,
            'spectral_entropy_max': 0.72 if eyes_closed else 0.78,
            'symmetry_db_max': 3.0,
            'hf_fraction_max': 0.15,
        }

    eeg = np.asarray(eeg)
    assert eeg.ndim == 2, "eeg must be (n_channels, n_samples)"
    n_ch, n_samp = eeg.shape
    assert len(channel_names) == n_ch, "channel_names length must match eeg channels"

    bands = {
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'hf':    (30, 80),
        'total': (freq_range[0], freq_range[1]),
    }

    name_upper = [c.upper() for c in channel_names]
    roi_keywords = ['O1','O2','OZ','POZ','PO3','PO4','PO7','PO8','P3','P4','PZ','P7','P8']
    roi_idx = [i for i, nm in enumerate(name_upper) if any(k in nm for k in roi_keywords)]
    if len(roi_idx) == 0:
        roi_idx = list(range(n_ch))

    pairs = []
    def find_idx(tag):
        try:
            return name_upper.index(tag)
        except ValueError:
            return None
    for a,b in [('O1','O2'), ('P3','P4'), ('PO3','PO4'), ('P7','P8')]:
        ia, ib = find_idx(a), find_idx(b)
        if ia is not None and ib is not None:
            pairs.append((ia, ib))

    win = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    if win <= 1 or step < 1 or win > n_samp:
        raise ValueError("Unreasonable window/step vs data length")
    starts = np.arange(0, n_samp - win + 1, step)
    n_win = len(starts)

    def bandpower_from_psd(freqs, pxx, lo, hi):
        idx = (freqs >= lo) & (freqs < hi)
        if not np.any(idx):
            return 0.0
        return float(trapz(pxx[idx], freqs[idx]))

    def spectral_entropy(freqs, pxx, lo, hi, eps=1e-12):
        idx = (freqs >= lo) & (freqs < hi)
        p = pxx[idx].astype(float)
        if p.sum() <= 0:
            return 1.0
        p /= (p.sum() + eps)
        h = -np.sum(p * np.log2(p + eps))
        h_max = np.log2(p.size + eps)
        return float(h / (h_max + eps))

    alpha_ratio = np.zeros(n_win)
    alpha_fraction = np.zeros(n_win)
    spec_entropy = np.zeros(n_win)
    hf_fraction = np.zeros(n_win)
    symmetry_db = np.zeros(n_win) if len(pairs) > 0 else np.full(n_win, np.nan)

    for wi, s0 in enumerate(starts):
        seg = eeg[:, s0:s0+win]
        pxx_accum = None
        freqs_ref = None
        for ch in roi_idx:
            f, pxx = welch(seg[ch], fs=fs, nperseg=min(win, 512), detrend='constant')
            if freqs_ref is None:
                freqs_ref = f
                pxx_accum = pxx
            else:
                pxx_accum += pxx
        pxx_roi = pxx_accum / max(1, len(roi_idx))

        p_theta = bandpower_from_psd(freqs_ref, pxx_roi, *bands['theta'])
        p_alpha = bandpower_from_psd(freqs_ref, pxx_roi, *bands['alpha'])
        p_beta  = bandpower_from_psd(freqs_ref, pxx_roi, *bands['beta'])
        p_hf    = bandpower_from_psd(freqs_ref, pxx_roi, *bands['hf'])
        p_tot   = bandpower_from_psd(freqs_ref, pxx_roi, *bands['total'])

        denom_ratio = (p_theta + p_beta)
        alpha_ratio[wi] = p_alpha / denom_ratio if denom_ratio > 0 else 0.0
        denom_fraction = (p_theta + p_alpha + p_beta)
        alpha_fraction[wi] = p_alpha / denom_fraction if denom_fraction > 0 else 0.0
        hf_fraction[wi] = p_hf / p_tot if p_tot > 0 else 0.0
        spec_entropy[wi] = spectral_entropy(freqs_ref, pxx_roi, *bands['total'])

        if len(pairs) > 0:
            diffs_db = []
            for ia, ib in pairs:
                fa, pxx_a = welch(seg[ia], fs=fs, nperseg=min(win, 512), detrend='constant')
                fb, pxx_b = welch(seg[ib], fs=fs, nperseg=min(win, 512), detrend='constant')
                p_a_alpha = bandpower_from_psd(fa, pxx_a, *bands['alpha'])
                p_b_alpha = bandpower_from_psd(fb, pxx_b, *bands['alpha'])
                p_a = max(p_a_alpha, 1e-16)
                p_b = max(p_b_alpha, 1e-16)
                diffs_db.append(abs(10*np.log10(p_a) - 10*np.log10(p_b)))
            symmetry_db[wi] = float(np.mean(diffs_db)) if diffs_db else np.nan

    passes = {
        'alpha_ratio':       alpha_ratio >= thresholds['alpha_ratio_min'],
        'alpha_fraction':    alpha_fraction >= thresholds['alpha_fraction_min'],
        'spectral_entropy':  spec_entropy <= thresholds['spectral_entropy_max'],
        'hf_fraction':       hf_fraction <= thresholds['hf_fraction_max'],
    }
    if not np.all(np.isnan(symmetry_db)):
        passes['symmetry_db'] = symmetry_db <= thresholds['symmetry_db_max']

    all_pass = np.ones(n_win, dtype=bool)
    for v in passes.values():
        all_pass &= v

    stable = np.zeros_like(all_pass)
    run = 0
    for i, val in enumerate(all_pass):
        run = run + 1 if val else 0
        if run >= consecutive_windows_required:
            stable[i-consecutive_windows_required+1:i+1] = True

    window_times = np.column_stack([starts / fs, (starts + win) / fs])

    intervals = []
    if np.any(stable):
        i = 0
        while i < n_win:
            if not stable[i]:
                i += 1
                continue
            start_t = window_times[i, 0]
            j = i
            while j + 1 < n_win and stable[j+1] and (starts[j+1] - starts[j]) == step:
                j += 1
            end_t = window_times[j, 1]
            intervals.append((float(start_t), float(end_t)))
            i = j + 1

    results = {
        'window_times': window_times,
        'features': {
            'alpha_ratio': alpha_ratio,
            'alpha_fraction': alpha_fraction,
            'spectral_entropy': spec_entropy,
            'symmetry_db': symmetry_db,
            'hf_fraction': hf_fraction,
        },
        'passes': passes,
        'rest_mask': stable,
        'rest_intervals': intervals,
    }
    return results



if __name__ == "__main__":
    fs = 250
    duration = 20.0
    n_channels = 8
    t = np.arange(0, duration, 1/fs)


    eeg = np.random.randn(n_channels, t.size) * 0.2   


    beta = np.sin(2*np.pi*18*t)
    active_mask = t < 5.0
    eeg[6, active_mask] += beta[active_mask] * 1.5
    eeg[7, active_mask] += beta[active_mask] * 1.5


    alpha = np.sin(2*np.pi*10*t)
    rest_mask_time = t >= 5.0
    eeg[6, rest_mask_time] += alpha[rest_mask_time] * 2.0  
    eeg[7, rest_mask_time] += alpha[rest_mask_time] * 2.0  

    channel_names = ["F3","F4","C3","C4","P3","P4","O1","O2"]

    results = detect_rest_epochs(
        eeg, fs, channel_names,
        window_sec=2.0,
        step_sec=0.5,
        eyes_closed=True,
        consecutive_windows_required=5
    )

    print("Rest intervals (s):", results["rest_intervals"])
    print("Num windows:", len(results["rest_mask"]), 
          "  Num rest windows:", int(results["rest_mask"].sum()))

  
    times = results["window_times"][:, 0]                 
    alpha_ratio = results["features"]["alpha_ratio"]
    is_rest = results["rest_mask"]


    alpha_ratio_threshold = 1.4

    plt.figure(figsize=(10, 4))
    plt.plot(times, alpha_ratio, label="Alpha ratio")
    plt.axhline(alpha_ratio_threshold, linestyle="--", label="Alpha ratio threshold")


    plt.plot(times[is_rest], alpha_ratio[is_rest], "o", label="Rest windows")


    for (s, e) in results["rest_intervals"]:
        plt.axvspan(s, e, alpha=0.15, label="Rest interval" if s == results["rest_intervals"][0][0] else None)

    plt.title("Rest detection demo (0–5 s active → 5–20 s rest)")
    plt.xlabel("Time (s)")
    plt.ylabel("Alpha ratio")
    plt.legend()
    plt.tight_layout()
    plt.show()