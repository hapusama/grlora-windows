import numpy as np


def build_chirp(sf: int, bw: float, fs: float, up: bool = True):
    """Build a LoRa reference chirp for given SF/BW/fs."""
    ns = int(fs * (1 << sf) / bw)
    t = np.arange(ns, dtype=np.float64) / fs
    Ts = ns / fs
    phase = 2.0 * np.pi * (-bw / 2.0 * t + (bw / (2.0 * Ts)) * t ** 2)
    chirp = np.exp(1j * phase)
    return chirp if up else np.exp(-1j * phase)


def _fft_corr_valid(sig: np.ndarray, templ: np.ndarray) -> np.ndarray:
    """Cross-correlate sig with templ using FFT (mode='valid')."""
    n = len(sig) + len(templ) - 1
    S = np.fft.fft(sig, n)
    T = np.fft.fft(templ, n)
    c = np.fft.ifft(S * np.conj(T))
    return c[len(templ) - 1:len(sig)]


def detect_packets(
    data: np.ndarray,
    sf: int,
    bw: float,
    fs: float,
    preamble_symbols: int = 8,
    energy_ratio_thr: float = 2.0,
    upchirp_thr: float = 0.3,
) -> list[dict]:
    """
    Detect LoRa packet positions using sliding-window correlation.
    Returns a list of dicts: [{"packet_start": int, "sfd_pos": int, "score": float}, ...]
    """
    ns = int(fs * (1 << sf) / bw)
    up = build_chirp(sf, bw, fs, up=True)
    down = build_chirp(sf, bw, fs, up=False)

    # ------------------------------------------------------------------
    # 1. Energy coarse peaks (100 ms window, 50 ms hop)
    # ------------------------------------------------------------------
    win = int(0.1 * fs)
    hop = int(0.05 * fs)
    powers = np.empty((len(data) - win) // hop + 1, dtype=np.float64)
    times = np.empty_like(powers, dtype=np.int64)
    for i, idx in enumerate(range(0, len(data) - win, hop)):
        powers[i] = np.mean(np.abs(data[idx:idx + win]) ** 2)
        times[i] = idx

    mean_power = np.mean(powers)
    candidates = []
    last = -1
    for i in np.argsort(powers)[::-1]:
        if powers[i] < mean_power * energy_ratio_thr:
            break
        t = int(times[i])
        if last < 0 or abs(t - last) > int(0.5 * fs):
            candidates.append(t)
            last = t
        if len(candidates) >= 30:
            break

    # ------------------------------------------------------------------
    # 2. Refine each coarse candidate with upchirp correlation
    # ------------------------------------------------------------------
    detected = []
    for coarse in candidates:
        start = max(0, coarse - int(0.3 * fs))
        end = min(len(data), coarse + int(0.5 * fs))
        seg = data[start:end]
        if len(seg) < ns:
            continue

        corr = _fft_corr_valid(seg, up)
        power = np.abs(corr) ** 2
        if power.size == 0:
            continue
        peak_rel = int(np.argmax(power))
        peak_abs = start + peak_rel

        # score using exact match
        x = np.array(data[peak_abs:peak_abs + ns], copy=True)
        if len(x) < ns:
            continue
        score = float(
            np.abs(np.sum(x.conj() * up))
            / (np.sqrt(np.sum(np.abs(x) ** 2) * np.sum(np.abs(up) ** 2)) + 1e-12)
        )
        if score < upchirp_thr:
            continue

        # ------------------------------------------------------------------
        # 3. Validate SFD with downchirp
        # ------------------------------------------------------------------
        expected_sfd = peak_abs + preamble_symbols * ns
        margin = int(ns * 0.25)
        sfd_start = max(0, expected_sfd - margin)
        sfd_end = min(len(data), expected_sfd + ns + margin)
        sfd_seg = data[sfd_start:sfd_end]
        if len(sfd_seg) < ns:
            continue
        sfd_corr = _fft_corr_valid(sfd_seg, down)
        sfd_peak = int(np.argmax(np.abs(sfd_corr) ** 2))
        sfd_pos = sfd_start + sfd_peak

        detected.append({
            "packet_start": peak_abs,
            "sfd_pos": sfd_pos,
            "score": score,
        })

    # Sort by packet_start and remove exact duplicates
    detected.sort(key=lambda d: d["packet_start"])
    unique = []
    for d in detected:
        if not unique or abs(d["packet_start"] - unique[-1]["packet_start"]) > 100:
            unique.append(d)
    return unique, ns
