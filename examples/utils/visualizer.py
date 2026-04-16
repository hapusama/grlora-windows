import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _get_viridis_lut():
    """Return a 256x3 uint8 viridis-like colormap (simplified)."""
    t = np.linspace(0, 1, 256)
    r = np.clip(0.267 + 0.105 * t + 0.63 * t**2 - 0.213 * t**3, 0, 1)
    g = np.clip(0.004 + 0.898 * t + 0.05 * t**2, 0, 1)
    b = np.clip(0.329 + 0.644 * t - 0.867 * t**2 + 0.538 * t**3, 0, 1)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


def _stft_magnitude(data, nperseg, noverlap):
    """Simple STFT magnitude."""
    hop = nperseg - noverlap
    n_frames = (len(data) - noverlap) // hop
    window = np.hanning(nperseg)
    frames = np.lib.stride_tricks.as_strided(
        data,
        shape=(n_frames, nperseg),
        strides=(hop * data.strides[0], data.strides[0]),
    )
    fft = np.fft.fft(frames * window, axis=1)
    mag = np.abs(fft) ** 2
    mag = np.fft.fftshift(mag, axes=1)
    return mag


def plot_packet_spectrogram(
    data: np.ndarray,
    packet_start: int,
    sfd_pos: int,
    sf: int,
    bw: float,
    fs: float,
    preamble_symbols: int = 8,
    save_path: str = "packet.png",
    title: str = "",
):
    """Draw a spectrogram of the preamble + SFD region using PIL (no matplotlib)."""
    ns = int(fs * (1 << sf) / bw)
    n_show = preamble_symbols + 4
    seg = data[packet_start:packet_start + n_show * ns + ns // 4]
    if len(seg) == 0:
        return

    nperseg = min(256, ns // 4)
    noverlap = nperseg // 2
    mag = _stft_magnitude(seg, nperseg, noverlap)
    db = 10 * np.log10(mag + 1e-20)
    db_min, db_max = np.percentile(db, 1), np.percentile(db, 99)
    if db_max <= db_min:
        db_min, db_max = db.max() - 60, db.max()
    norm = np.clip((db - db_min) / (db_max - db_min), 0, 1)

    lut = _get_viridis_lut()
    img_arr = (norm * 255).astype(np.uint8)
    rgb = lut[img_arr]

    margin_top = 60
    margin_left = 80
    margin_bottom = 40
    margin_right = 20
    H, W = rgb.shape[:2]
    canvas = np.zeros(
        (H + margin_top + margin_bottom, W + margin_left + margin_right, 3),
        dtype=np.uint8,
    )
    canvas[margin_top:margin_top + H, margin_left:margin_left + W, :] = rgb

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_title = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        font_title = font

    draw.text((margin_left, 10), f"{title} - SF{sf} BW{bw/1e3:.0f}kHz", fill=(255, 255, 255), font=font_title)

    sym_dur_ms = ns / fs * 1000.0
    total_ms = W / fs * (nperseg - noverlap) * 1000.0
    px_per_ms = W / total_ms
    for i in range(1, preamble_symbols + 3):
        x_ms = i * sym_dur_ms
        x_px = margin_left + int(x_ms * px_per_ms)
        if x_px >= margin_left + W:
            break
        color = (255, 0, 0) if i == preamble_symbols else (255, 165, 0) if i == preamble_symbols + 2 else (200, 200, 200)
        draw.line([(x_px, margin_top), (x_px, margin_top + H)], fill=color, width=2 if i >= preamble_symbols else 1)

    draw.text((margin_left + 10, margin_top + 10), f"Preamble ({preamble_symbols} upchirps)", fill=(0, 255, 0), font=font)
    preamble_end_px = margin_left + int(preamble_symbols * sym_dur_ms * px_per_ms)
    sfd_end_px = margin_left + int((preamble_symbols + 2.25) * sym_dur_ms * px_per_ms)
    draw.text(((preamble_end_px + sfd_end_px) // 2, margin_top + 10), "SFD", fill=(255, 0, 0), font=font)

    draw.text((margin_left + W // 2 - 20, margin_top + H + 10), "Time →", fill=(255, 255, 255), font=font)
    draw.text((10, margin_top + H // 2), "Freq", fill=(255, 255, 255), font=font)

    img.save(save_path)
    print(f"  Saved figure: {save_path}")


def plot_full_packet_spectrogram(
    data: np.ndarray,
    packet_start: int,
    sf: int,
    bw: float,
    fs: float,
    preamble_symbols: int = 8,
    header_syms: int = 8,
    payload_syms: int = 32,
    save_path: str = "packet_full.png",
    title: str = "",
):
    """
    Draw a spectrogram of the ENTIRE packet (Preamble + Sync + SFD + Header + Payload).
    Does NOT dechirp — shows raw time-frequency content.
    """
    ns = int(fs * (1 << sf) / bw)

    # Total symbols to visualize
    total_syms = preamble_symbols + 2 + 2.25 + header_syms + payload_syms
    total_samples = int(total_syms * ns)
    seg = data[packet_start:packet_start + total_samples]
    if len(seg) == 0:
        return

    # STFT parameters: use ~1/4 symbol as window to resolve chirp slope
    nperseg = min(2048, max(256, ns // 4))
    noverlap = nperseg // 2
    hop = nperseg - noverlap

    mag = _stft_magnitude(seg, nperseg, noverlap)
    db = 10 * np.log10(mag + 1e-20)
    db_min, db_max = np.percentile(db, 1), np.percentile(db, 99)
    if db_max <= db_min:
        db_min, db_max = db.max() - 70, db.max()
    norm = np.clip((db - db_min) / (db_max - db_min), 0, 1)

    lut = _get_viridis_lut()
    img_arr = (norm * 255).astype(np.uint8)
    rgb = lut[img_arr]  # (H, W, 3)

    # Canvas margins
    margin_top = 70
    margin_left = 90
    margin_bottom = 50
    margin_right = 30
    H, W = rgb.shape[:2]
    canvas = np.zeros(
        (H + margin_top + margin_bottom, W + margin_left + margin_right, 3),
        dtype=np.uint8,
    )
    canvas[margin_top:margin_top + H, margin_left:margin_left + W, :] = rgb

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_title = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_title = font

    # Title
    draw.text(
        (margin_left, 10),
        f"{title} - SF{sf} BW{bw/1e3:.0f}kHz Fs{fs/1e3:.0f}kHz",
        fill=(255, 255, 255),
        font=font_title,
    )

    # Helper to convert symbol offset -> pixel x
    # time per STFT frame = hop / fs
    px_per_sample = W / len(seg) if len(seg) > 0 else 0
    # Actually it's more accurate: W columns correspond to (len(seg)-noverlap)//hop frames
    # but linear mapping over len(seg) is close enough for annotation
    def sym_x(sym_idx):
        samp = sym_idx * ns
        return margin_left + int(samp * px_per_sample)

    boundaries = {
        "Preamble start": 0,
        "Sync Word": preamble_symbols,
        "SFD": preamble_symbols + 2,
        "Header": preamble_symbols + 4.25,
        "Payload": preamble_symbols + 4.25 + header_syms,
    }

    colors = {
        "Preamble start": (0, 255, 0),
        "Sync Word": (255, 255, 0),
        "SFD": (255, 0, 0),
        "Header": (0, 255, 255),
        "Payload": (255, 165, 0),
    }

    # Draw vertical boundary lines
    for label, sym_idx in boundaries.items():
        x = sym_x(sym_idx)
        if margin_left <= x < margin_left + W:
            draw.line([(x, margin_top), (x, margin_top + H)], fill=colors[label], width=2)

    # Label positions (avoid overlap)
    label_y = margin_top + 10
    label_positions = [
        ("Preamble", 0, preamble_symbols, (0, 255, 0)),
        ("Sync", preamble_symbols, preamble_symbols + 2, (255, 255, 0)),
        ("SFD", preamble_symbols + 2, preamble_symbols + 4.25, (255, 0, 0)),
        ("Header", preamble_symbols + 4.25, preamble_symbols + 4.25 + header_syms, (0, 255, 255)),
        ("Payload", preamble_symbols + 4.25 + header_syms, total_syms, (255, 165, 0)),
    ]

    for text, s0, s1, col in label_positions:
        x0 = sym_x(s0)
        x1 = sym_x(s1)
        xm = (x0 + x1) // 2
        if xm < margin_left or xm > margin_left + W:
            continue
        # Draw small text label centered in the region
        draw.text((xm - 20, label_y), text, fill=col, font=font)

    # Axes
    draw.text((margin_left + W // 2 - 20, margin_top + H + 10), "Time →", fill=(255, 255, 255), font=font)
    draw.text((10, margin_top + H // 2), "Freq", fill=(255, 255, 255), font=font)

    img.save(save_path)
    print(f"  Saved full-packet figure: {save_path}")
