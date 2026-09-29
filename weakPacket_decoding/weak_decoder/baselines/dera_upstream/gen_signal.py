#!/usr/bin/env python3
import time
import math
import cupy as xp
import cupyx.scipy.fft as xfft
import cupyx
import numpy as np
from scipy.special import fresnel


def get_default_codes(Config):
    """Returns the default code sequence for the LoRa frame."""
    codes_base = [3401, 128, 4012, 567, 2190, 843, 3310, 15, 3988, 1254, 276, 1899, 3055, 421, 2234, 98, 3671, 1442, 2890, 512, 3888, 112, 2567, 190, 3100, 776, 4090, 134, 2211, 65]
    codes_base = [c % Config.n_classes for c in codes_base]
    payload_codes = (codes_base * (Config.payload_len // len(codes_base) + 1))[:Config.payload_len]
    codes = [0]*8 + [8, 16] + [0, 0, 0] + payload_codes
    return codes

# --- Signal Generation ---
def gen_signal2(cfo, to, phase_start, total_len, codes, Config, return_phase=False):
    """
    Generate complete LoRa signal using piecewise closed-form functions.
    """
    dtype = xp.float64 if return_phase else xp.complex64
    sig = xp.zeros(int(total_len + to), dtype=dtype)
    
    # Constants
    epsilon = cfo / Config.sig_freq
    T_nominal_full = (2 ** Config.sf) / Config.bw * Config.fs
    T_prime = T_nominal_full * (1 - epsilon) # samples per symbol
    T_prime_sec = T_prime / Config.fs
    
    BW_prime = Config.bw * (1 + epsilon)
    # Tx_sec for beta calculation
    Tx_sec = (Config.nsamp / Config.fs) * (1 - epsilon)
    
    beta_up = Config.bw * (1 + epsilon) / Tx_sec
    beta_down = -beta_up
    
    dphi_per_symbol = 2 * xp.pi * cfo * T_prime_sec
    
    segments = []
    def add_seg_explicit(t_start, t_end, f_start, beta_val, phi_start):
        segments.append({
            't_start': t_start, 't_end': t_end,
            'A': xp.pi * beta_val, 'B': 2 * xp.pi * f_start, 'C': phi_start
        })

    # --- Preamble (0..7) + Sync (8..9) ---
    for i in range(Config.preamble_len + Config.code_len):
        sym_start = to + i * T_prime
        sym_phase = phase_start + i * dphi_per_symbol
        code_val = codes[i]
        c_rat = code_val / Config.n_classes
        
        # Segment 1
        dur1 = T_prime * (1.0 - c_rat)
        f_start1 = BW_prime * (-0.5 + c_rat) + cfo
        add_seg_explicit(sym_start, sym_start + dur1, f_start1, beta_up, sym_phase)
        
        if c_rat > 0:
            # Segment 2
            T1_sec = dur1 / Config.fs
            dphi1 = 2 * xp.pi * (f_start1 * T1_sec + 0.5 * beta_up * T1_sec**2)
            
            dur2 = T_prime * c_rat
            f_start2 = BW_prime * (-0.5 + 0.0) + cfo
            add_seg_explicit(sym_start + dur1, sym_start + dur1 + dur2, f_start2, beta_up, sym_phase + dphi1)

    # --- SFD (10..12) ---
    base_idx = Config.preamble_len + Config.code_len
    sfd_base_time = to + base_idx * T_prime
    sfd_base_phase = phase_start + base_idx * dphi_per_symbol
    
    f_down_start = BW_prime * 0.5 + cfo
    # Downchirp 1 & 2
    add_seg_explicit(sfd_base_time, sfd_base_time + T_prime, f_down_start, beta_down, sfd_base_phase)
    add_seg_explicit(sfd_base_time + T_prime, sfd_base_time + 2*T_prime, f_down_start, beta_down, sfd_base_phase + dphi_per_symbol)
    
    # Downchirp 0.25
    t_start_q = sfd_base_time + 2*T_prime
    phi_start_q = sfd_base_phase + 2*dphi_per_symbol
    add_seg_explicit(t_start_q, t_start_q + 0.25*T_prime, f_down_start, beta_down, phi_start_q)
    
    # --- Payload (13..) ---
    T_quarter_sec = (0.25 * T_prime) / Config.fs
    dphi_quarter = 2 * xp.pi * (f_down_start * T_quarter_sec + 0.5 * beta_down * T_quarter_sec**2)
    
    base_idx = Config.preamble_len + Config.code_len + 3
    payload_base_time = to + (base_idx - 0.75) * T_prime
    payload_base_phase = phase_start + (base_idx - 1) * dphi_per_symbol + dphi_quarter
    
    n_payload = len(codes) - base_idx
    for i in range(n_payload):
        idx = base_idx + i
        sym_start = payload_base_time + i * T_prime
        sym_phase = payload_base_phase + i * dphi_per_symbol
        code_val = codes[idx]
        c_rat = code_val / Config.n_classes
        
        dur1 = T_prime * (1.0 - c_rat)
        f_start1 = BW_prime * (-0.5 + c_rat) + cfo
        add_seg_explicit(sym_start, sym_start + dur1, f_start1, beta_up, sym_phase)
        
        if c_rat > 0:
            T1_sec = dur1 / Config.fs
            dphi1 = 2 * xp.pi * (f_start1 * T1_sec + 0.5 * beta_up * T1_sec**2)
            
            dur2 = T_prime * c_rat
            f_start2 = BW_prime * (-0.5 + 0.0) + cfo
            add_seg_explicit(sym_start + dur1, sym_start + dur1 + dur2, f_start2, beta_up, sym_phase + dphi1)
            
    # --- Generation ---
    for seg in segments:
        t_start = seg['t_start']
        t_end = seg['t_end']
        start_idx = int(xp.ceil(t_start))
        end_idx = int(xp.ceil(t_end))
        
        if start_idx >= total_len: break
        if end_idx <= start_idx: continue
            
        len_write = min(end_idx - start_idx, total_len - start_idx)
        if len_write <= 0: continue
            
        t_idx = xp.arange(len_write, dtype=xp.float64)
        t_rel_sec = (start_idx - t_start + t_idx) / Config.fs
        
        A, B, C = seg['A'], seg['B'], seg['C']
        phase = A * t_rel_sec**2 + B * t_rel_sec + C
        
        if return_phase:
             sig[start_idx:start_idx + len_write] = phase.astype(xp.float64)
        else:
             sig[start_idx:start_idx + len_write] += xp.exp(1j * phase).astype(xp.complex64)

    return sig

# --- Grid Search Logic ---
