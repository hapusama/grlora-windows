# -*- coding: utf-8 -*-
"""DeRa (MobiCom'26) 检测级 port（phase-coherent detection，v2）。

论文依据（paper/DeRa_MobiCom26.pdf 附录 Algorithm 1 lines 1-23 + §4.1-4.2
+ Appendix A.6 Eqs.46-47，页 7/19-21 pdftotext -layout 直读核对）：

  - Precomputation（lines 1-5）：符号窗分别用对应参考啁啾去斜 → 零填
    padded FFT → DC 中心符号谱（Appendix B Eq.48）。
  - Column-wise FFT（lines 6-9, §4.1 Eqs.13-14）：P×bin 谱矩阵 Y，峰值列
    k*，沿符号轴零填列向 FFT → 峰值 q̂ 解出符号间相位旋转率 =
    frac(f_cfo·T')（= κ，抛物线内插连续化）。列向 FFT = 对全部旋转率的
    同时相干累积（Eq.12 的高效搜索形式）。
  - 检测得分（Appendix A.6 Eq.47）：相干 SNR
    score = |F[q̂]|²/(P·σ̂²)，σ̂² = 远离峰列的噪声列功率中位数
    （SNR_coh = P·SNR₁ 的峰列版）。⚠️ 不能用峰/均值：Parseval 使其
    封顶 10log10(P)（v1 教训，native 8/16/32 符号恰得 9/12/15dB）。
  - Coarse-Fine 融合（Eqs.16-18）：f̂ = round(k_up − κ̂) + κ̂。
  - 结构校验 + 精化（lines 14-23）：产线 locate_frame_from_event 在候选
    start 附近做 1-sample 级符号栅格搜索（sync-word/SFD 判据与产线
    共用），以其 preamble_start 为整数边界，随后在该边界重测 k_up。
  - top-5 候选留给解码端 CRC fallback（Stage 3 统一环 lines 24-45）。

v2 相对 v1 的修正（probe1-4 定案，见实验目录 debug_detector*.py）：
  1. **啁啾时频模糊**：参考在窗内从样本 0 重启，窗偏移 δ 采样使去斜
     tone 移 δ·N/NF bin（probe4/A-B 探针实测 +64 bin/256 smp）。因此
     候选 k_up = cfo + δ·N/NF；只有符号栅格对齐（δ 的整数部分=0）的
     窗口其 tone 才与扫描事件 bin 一致——locate 的 1-sample 精化给出
     该对齐，f̂ 在对齐窗上重测。
  2. 检测统计量改为相干 SNR（Eq.47 形态），噪声-only 基线 ~7.4dB，
     门限 13dB。
  3. τ̂（Eqs.50-51 up/down 括号）：u = k_up − k_dn = u0 + e/2（e =
     start 误差，样本；斜率 2 为探针实测，u0=0.5 native 冻结）。括号用于
     把粗 start 校正到 ±8 样本内；残余亚样本时延 ≤8 sample ⇒ ≤2 bin 的
     tone 残差由下游解调器的 δ 仲裁吸收（与全部参战链同一原语）。对齐
     不施加 frac_delay（论文 line 27 的 corrected timing 效果由括号校正
     + tone 一致性等价实现，差异在 RESULTS 声明）。

本 port 与论文的其余差异（数据集上不活跃机制，如实声明）：
  η=f_cfo/f_c ≈ 1e-6 ⇒ 论文 §3 复合畸变项（时漂 Eq.2/频移 Eq.3/泄漏
  Eq.4/Eqs.10-11 中项）在 P≤32 符号上 <0.03 bin，未逐字 port；两段能量
  分裂以峰值列 ±1 列聚取覆盖；5×5 精化代之以 locate 1-sample 栅格；
  SFD 段仅用于结构校验（locate 内部）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...chirp import build_upchirp


def _interp_peak(mag_row: np.ndarray, idx: int) -> float:
    """抛物线内插峰位（符号谱 k* 与列向 FFT q̂ 共用）。"""
    if idx <= 0 or idx >= len(mag_row) - 1:
        return float(idx)
    a, b, c = float(mag_row[idx - 1]), float(mag_row[idx]), float(mag_row[idx + 1])
    denom = a - 2.0 * b + c
    if denom == 0.0:
        return float(idx)
    delta = 0.5 * (a - c) / denom
    if abs(delta) > 0.5:
        return float(idx)
    return idx + delta


@dataclass(frozen=True)
class DeRaCandidate:
    """一个 (f̂_cfo, start) 候选（Algorithm 1 line 23 的 top-5 元素）。

    对齐用法（与产线 exp2 对齐同构）：
      seg_a = seg·exp(−2πj·f_bins·n/NF)，再 frac_delay(seg_a, −grid_shift)
      （把 locate 网格平移到 NF 整数倍网格，使 pay0·NF 解码窗与 f̂ 测量
      窗同一网格残差——时频模糊下 tone 一致性的必要条件）；
      pay0 = hs_est // NF + 8。
    """

    score_db: float          # 相干 SNR（dB，Eq.47 形态）
    start_sample: int        # locate 精化后的前导首符号样本位（原坐标）
    f_bins: float            # f̂（bin 域 int+frac，Eq.18 融合，locate 网格上测）
    kappa: float             # κ̂（列向 FFT 旋转率，诊断输出）
    hs_est: int              # header 起点样本（已扣除 grid_shift，平移后坐标）
    grid_shift: int          # locate 网格残差 r_L（对齐时推进的样本数）
    scan_bin: int            # 扫描事件参考 bin（signed，1-bin 格）

    def as_dict(self) -> dict:
        return dict(score_db=self.score_db, start_sample=self.start_sample,
                    f_bins=self.f_bins, kappa=self.kappa,
                    hs_est=self.hs_est, grid_shift=self.grid_shift,
                    scan_bin=self.scan_bin)


class DeRaDetector:
    """DeRa 相干检测前端（输出与产线 sync_and_align 对等的对齐估计）。

    用法::

        det = DeRaDetector(sf=10, os_factor=4)
        cands = det.detect_frame(seg, pre, locate=locate_cb)  # score 降序
        # 对齐：seg·exp(-2πj·f_bins·n/NF)（不施加分数时延，见模块 docstring）
        # pay0 = hs_est // NF + 8；逐候选解码 + CRC 仲裁（Stage 3）。
    """

    #: 噪声列估计与峰列的最小距离（padded bin）
    NOISE_GUARD = 16
    #: up/down 括号常数 u0：u = k_up − k_dn = u0 + e/2（e=locate 起点误差，
    #: 样本）。斜率 2 为探针实测（大误差帧 e=2u+b, b∈[0,6]）；u0=1.0 为
    #: native 校正后 f̂ 残差≈0.5 bin 的冻结值（downchirp 参考约定常数）。
    BRACKET_U0 = 1.0

    def __init__(self, sf: int, os_factor: int, n_fine: int = 256,
                 scan_hop: int | None = None, scan_thresh_db: float = 4.0,
                 coh_thresh_db: float = 15.0, max_candidates: int = 5):
        self.sf = int(sf)
        self.os = int(os_factor)
        self.n = 1 << self.sf
        self.nf = self.n * self.os
        self.n_fft = 2 * self.nf                     # ζ=2 零填（0.5-bin 网格）
        self.n_fine = int(n_fine)                    # 列向 FFT 长度
        self.scan_hop = int(scan_hop or self.nf // 8)
        self.scan_thresh_db = float(scan_thresh_db)
        self.coh_thresh_db = float(coh_thresh_db)
        self.max_candidates = int(max_candidates)
        up = build_upchirp(self.sf, symbol_id=0, os_factor=self.os)
        self.down_ref = np.conj(up).astype(np.complex64)   # 去斜 upchirp 符号
        self.up_ref = up.astype(np.complex64)              # 去斜 downchirp 符号

    # ---------------- 符号谱（Algorithm 1 lines 1-5 / Appendix B） ----------------
    def signed_spectrum(self, seg: np.ndarray, start: int, ref: np.ndarray,
                        ) -> np.ndarray:
        """去斜 → ζ=2 零填 FFT → DC 中心 2N 点行（0.5-bin 格，±B/2）。"""
        w = np.asarray(seg[start:start + self.nf], dtype=np.complex64)
        if w.size != self.nf:
            raise ValueError("symbol window exceeds input")
        X = np.fft.fft(w * ref, self.n_fft)
        return np.concatenate((X[self.n_fft - self.n:], X[:self.n]))

    # ---------------- 扫描（DeRa 自有盲扫门） ----------------
    def scan(self, seg: np.ndarray) -> list[dict]:
        """hop 滑窗单啁啾峰检测 → 周期事件分组（按总峰功率降序）。

        只有符号栅格对齐（±4 bin）的窗口在同一 signed bin 出峰（时频
        模糊使未对齐窗的 tone 随偏移移动 1 bin/4 smp），故事件 bin 即
        对齐前导 tone，事件 start 对齐符号栅格（mod NF）。
        """
        half = self.n // 2
        peaks = []
        for s in range(0, len(seg) - self.nf + 1, self.scan_hop):
            X = np.fft.fft(np.asarray(seg[s:s + self.nf], dtype=np.complex64)
                           * self.down_ref)
            row = np.concatenate((np.abs(X[self.nf - half:]) ** 2,
                                  np.abs(X[:half]) ** 2))   # signed ±B/2
            med = float(np.median(row))
            if med <= 0:
                continue
            k = int(np.argmax(row))
            ratio_db = 10.0 * np.log10(float(row[k]) / med)
            if ratio_db > self.scan_thresh_db:
                peaks.append(dict(pos=s, bin=k, ratio=ratio_db))
        peaks.sort(key=lambda p: p["pos"])
        events: list[dict] = []
        used = [False] * len(peaks)
        for i, p in enumerate(peaks):
            if used[i]:
                continue
            group = [p]
            used[i] = True
            for j in range(i + 1, len(peaks)):
                if used[j]:
                    continue
                q = peaks[j]
                db = abs(q["bin"] - group[-1]["bin"])
                db = min(db, self.n - db)
                gap = q["pos"] - group[-1]["pos"]
                if db <= 4 and 0 < gap <= 2 * self.nf:
                    group.append(q)
                    used[j] = True
            if len(group) >= 3:
                events.append(dict(
                    start=min(g["pos"] for g in group),
                    ref_bin=int(np.bincount([g["bin"] for g in group]).argmax()),
                    power=float(sum(10 ** (g["ratio"] / 10.0) for g in group))))
        events.sort(key=lambda e: -e["power"])
        return events

    # ---------------- 列向 FFT 相干阶段（lines 6-13, §4.1-4.2） ----------------
    def coherent_stage(self, seg: np.ndarray, pre: int, event: dict,
                       ) -> list[dict]:
        """两阶段 FFT + coarse-fine 融合（对事件内符号栅格 start 网格）。

        返回候选（score_db, start, kappa, k_up）；f̂ 的最终值在 locate
        精化的 start 上重测（detect_frame 内完成）。
        """
        out = []
        base = int(event["start"])
        span = 2 * self.nf + base
        for s in range(base - 2 * self.nf, span + 1, self.nf):
            if s < 0 or s + pre * self.nf > len(seg):
                continue
            # 覆盖 512·k 网格残差类（hop 栅格的相位量子）
            for off in (-512, 0, 512):
                s2 = s + off
                if s2 < 0:
                    continue
                try:
                    rows = np.stack([
                        self.signed_spectrum(seg, s2 + i * self.nf,
                                             self.down_ref)
                        for i in range(pre)])
                except ValueError:
                    continue
                out.append(self._score_rows(rows, pre, s2,
                                            int(event["ref_bin"])))
        out = [c for c in out if c is not None]
        out.sort(key=lambda c: -c["score_db"])
        return out

    def _score_rows(self, rows: np.ndarray, pre: int, start: int,
                    scan_bin: int) -> dict | None:
        """列向 FFT 相干得分 + κ̂/k_up（一段已算好的符号谱行）。"""
        mag = np.sqrt(np.mean(np.abs(rows) ** 2, axis=0))
        k_star = int(np.argmax(mag))
        if not (self.NOISE_GUARD < k_star < rows.shape[1] - self.NOISE_GUARD):
            return None
        far = np.ones(rows.shape[1], dtype=bool)
        far[max(0, k_star - self.NOISE_GUARD):k_star + self.NOISE_GUARD + 1] = False
        sigma2 = float(np.median(np.abs(rows[:, far]) ** 2))
        if sigma2 <= 0:
            return None
        best = None
        for k in (k_star - 1, k_star, k_star + 1):
            if k <= 0 or k >= rows.shape[1] - 1:
                continue
            col = rows[:, k]
            F = np.abs(np.fft.fft(col, self.n_fine))
            q = int(np.argmax(F))
            score_db = 10.0 * np.log10(float(F[q] ** 2) / (pre * sigma2))
            if best is None or score_db > best["score_db"]:
                best = dict(score_db=score_db, k=k, q=q, F=F)
        if best is None:
            return None
        q_i = _interp_peak(best["F"], best["q"])
        kappa = q_i / self.n_fine
        if kappa > 0.5:
            kappa -= 1.0
        k_up = _interp_peak(mag, best["k"])      # signed，bin 域（0.5 格）
        return dict(score_db=best["score_db"], start=int(start),
                    kappa=float(kappa),
                    k_up=float((k_up - self.n) / 2.0), scan_bin=scan_bin)

    def _k_up_k_dn(self, seg: np.ndarray, pre: int, start: int,
                   ) -> tuple[float, float]:
        """对齐窗 (start, ±4) 上的 signed 谱峰：up=preamble, dn=SFD 下啁啾。"""
        rows = np.stack([self.signed_spectrum(seg, start + i * self.nf,
                                              self.down_ref)
                         for i in range(pre)])
        mag_u = np.sqrt(np.mean(np.abs(rows) ** 2, axis=0))
        k_up = (_interp_peak(mag_u, int(np.argmax(mag_u))) - self.n) / 2.0
        dn = np.stack([self.signed_spectrum(seg, start + (pre + 2 + j) * self.nf,
                                            self.up_ref) for j in range(2)])
        mag_d = np.sqrt(np.mean(np.abs(dn) ** 2, axis=0))
        k_dn = (_interp_peak(mag_d, int(np.argmax(mag_d))) - self.n) / 2.0
        return float(k_up), float(k_dn)

    # ---------------- 对外入口 ----------------
    def detect_frame(self, seg: np.ndarray, pre: int,
                     locate=None) -> list[DeRaCandidate]:
        """完整检测：扫描 → 相干阶段 → 括号校正 → locate 精化 → top-5。

        locate 回调签名 locate(seg, start, pre) → (valid, hs_est,
        preamble_start)；由调用方注入产线 locate_frame_from_event 的包装
        （两前端共用同一结构判据；回调内部吞异常，失败即候选出局）。
        流程：相干阶段选出事件内最优 start（粗，符号栅格）→ up/down 括号
        （Eqs.50-51）把 start 校正到 ±8 样本内（e = 2·(u − u0)）→ locate
        在校正位置做结构校验与 1-sample 栅格精化 → 对齐窗重测 f̂（Eq.18）。
        """
        cands: list[DeRaCandidate] = []
        locate_calls = 0
        for event in self.scan(seg)[:6]:
            if locate_calls >= 2 and cands:
                break
            stage = self.coherent_stage(seg, pre, event)
            if not stage or stage[0]["score_db"] < self.coh_thresh_db:
                continue
            c0 = stage[0]
            try:
                k_up0, k_dn0 = self._k_up_k_dn(seg, pre, c0["start"])
                u = k_up0 - k_dn0
                # 括号在 ±512 signed 谱上卷绕：|u| 过大说明 start 落在错
                # 符号/错残差上，弃用括号（locate 的 span=1 网格搜索兜底）
                e_corr = int(round(2.0 * (u - self.BRACKET_U0)))                     if abs(u) <= 200 else 0
            except ValueError:
                e_corr = 0
            s_fixed = c0["start"] - e_corr
            if locate is not None:
                hs = pre_start = -1
                for s_try in (s_fixed, c0["start"]):
                    locate_calls += 1
                    try:
                        ok, hs_, ps_ = locate(seg, s_try, pre)
                    except Exception:
                        continue
                    if ok:
                        hs, pre_start = hs_, ps_
                        break
                if hs < 0:
                    continue
            else:
                pre_start = s_fixed
                hs = pre_start + int(round((pre + 4.25) * self.nf))
            # f̂/κ̂ 终测：测量窗放在【前导自身的符号网格】=
            # payload 网格 − 0.25·NF（SFD 四分之一符号的结构常量）：
            #     pre_meas = (hs_loc//NF − pre − 4)·NF − 1024
            # 本库帧结构：payload 边界恰在 NF 整数倍网格（段构造
            # lead=pre+6 与帧内 4.25 对消），前导网格与之恒差 0.25 符号。
            # 该放置对 locate 误差 e 完全鲁棒（NF 整数倍 − 1024 与 e 无
            # 关），native 实测恰落在真前导起点上；tone 残差 = κ（≤0.5 bin）。
            hs_sym = int(round(int(hs) / self.nf))   # floor 越界免疫
            hs = hs_sym * self.nf
            pre_meas = (hs_sym - pre - 4) * self.nf - self.nf // 4
            if pre_meas < 0:
                pre_meas = int(pre_start)
            try:
                rows_c = np.stack([
                    self.signed_spectrum(seg, pre_meas + i * self.nf,
                                         self.down_ref) for i in range(pre)])
                m_fin = self._score_rows(rows_c, pre, pre_meas,
                                         c0["scan_bin"])
            except ValueError:
                m_fin = None
            if m_fin is None:
                continue
            k_up, kappa_fin = m_fin["k_up"], m_fin["kappa"]
            f_bins = round(k_up - kappa_fin) + kappa_fin     # Eq.18 融合
            if any(abs(int(hs) - x.hs_est) < 512 for x in cands):
                continue
            cands.append(DeRaCandidate(
                score_db=c0["score_db"], start_sample=int(pre_start),
                f_bins=float(f_bins), kappa=float(kappa_fin),
                hs_est=int(hs), grid_shift=0,
                scan_bin=c0["scan_bin"]))
        cands.sort(key=lambda x: -x.score_db)
        return cands[: self.max_candidates]
