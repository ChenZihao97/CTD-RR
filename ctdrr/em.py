"""
Core EM for CTD-RR.

Each mark y_i from annotator j on segment t is drawn from one of K_t Gaussian
components, one per latent boundary:

    p(y_i | boundary k) = N(y_i; X*_k, 1 / lambda_{j,t})

E-step -- soft correspondence of marks to boundaries:

    r_ik  proportional to  N(y_i; X*_k, 1 / lambda_{j,t})

Truth update -- reliability-weighted mean of the marks assigned to k:

    X*_k = sum_j lambda_{j,t} sum_i r_ik y_i / sum_j lambda_{j,t} sum_i r_ik

The precision update lives in reliability.py.  This module also holds the
K initialisation, the pruning of unsupported components and the merging of
degenerate ones.
"""

import json
import numpy as np
import os
from scipy.optimize import linear_sum_assignment


# =====================================================================
# 1. Helpers
# =====================================================================

def time_to_sec(time_str):
    """'MM:SS' -> seconds.  Numbers pass through unchanged.

    The MedVid annotations store 'MM:SS' strings; the Kinetics corpora store
    float seconds, and their sub-second boundaries would be destroyed by the
    integer parse, so numeric input is returned as-is.
    """
    if time_str is None or time_str == '':
        return None
    if isinstance(time_str, (int, float)):
        return float(time_str)
    if isinstance(time_str, str) and ':' not in time_str:
        return float(time_str)
    parts = str(time_str).split(':')
    return int(parts[0]) * 60 + int(parts[1])


def sec_to_time(sec):
    """Converts seconds to 'MM:SS' string."""
    if sec is None or np.isnan(sec):
        return None
    sec_rounded = int(round(sec))
    return f"{sec_rounded // 60:02d}:{sec_rounded % 60:02d}"


def load_json_data(filepath):
    """Loads JSON and sorts by sample_id."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return sorted(data, key=lambda x: x['sample_id'])


def log_gaussian(y, mu, precision):
    """Log N(y; mu, 1/precision). Log-space for numerical stability."""
    if precision <= 1e-15:
        return -700.0
    return 0.5 * np.log(precision / (2.0 * np.pi)) - 0.5 * precision * (y - mu) ** 2


# =====================================================================
# 2. Data Extraction
# =====================================================================

def extract_annotations(annotator_data_list):
    """
    Extract timestamps and segment boundaries from arbitrary number of annotators.

    Parameters
    ----------
    annotator_data_list : list of list-of-dicts
        Each entry is one annotator's JSON data (already sorted by sample_id).
        All entries must have the same length N (number of video segments).

    Returns
    -------
    timestamps : dict {(j, t): list of float}
    seg_starts : list of float, length N
    seg_ends   : list of float, length N
    N          : int, number of video segments
    J          : int, number of annotators
    """
    J = len(annotator_data_list)
    assert J >= 2, "Need at least 2 annotators for truth discovery."

    N = len(annotator_data_list[0])
    for j in range(1, J):
        assert len(annotator_data_list[j]) == N, (
            f"Annotator {j} has {len(annotator_data_list[j])} videos, "
            f"expected {N}"
        )

    timestamps = {}
    seg_starts = []
    seg_ends = []

    # Use annotator 0 as the reference for segment metadata.
    ref_data = annotator_data_list[0]

    for t in range(N):
        seg_start_str = ref_data[t].get('segment_start')
        seg_end_str = ref_data[t].get('segment_end')
        video_length = ref_data[t].get('video_length', None)

        if seg_start_str and isinstance(seg_start_str, str) and ':' in seg_start_str:
            seg_start = time_to_sec(seg_start_str)
        else:
            seg_start = ref_data[t].get('segment_start_second', 0)

        if seg_end_str and isinstance(seg_end_str, str) and ':' in seg_end_str:
            seg_end = time_to_sec(seg_end_str)
        elif video_length:
            seg_end = float(video_length)
        else:
            seg_end = ref_data[t].get('segment_end_second', 3600)

        seg_starts.append(float(seg_start))
        seg_ends.append(float(seg_end))

        for j in range(J):
            ts = []
            for s in annotator_data_list[j][t].get('steps_list', []):
                val = time_to_sec(s['step_caption_boundary'])
                if val is not None:
                    ts.append(float(val))
            ts.sort()
            timestamps[(j, t)] = ts

    return timestamps, seg_starts, seg_ends, N, J


# =====================================================================
# 3. Initialization
# =====================================================================

def determine_K(timestamps, t, J):
    """
    Step count K_t = max of non-zero annotator counts.

    Why max?
      Start with the most steps any annotator found. If a step is NOT
      real, it receives low responsibility and gets pruned. Over-estimation
      is recoverable; under-estimation loses steps permanently.
    """
    counts = [len(timestamps[(j, t)]) for j in range(J)]
    non_zero = [c for c in counts if c > 0]

    if len(non_zero) == 0:
        return 0

    return max(non_zero)


def initialize(timestamps, N, J):
    """Initialize K_t, X*_k, and lambda_jt."""
    K = []
    active_videos = []
    X_star = {}

    for t in range(N):
        K_t = determine_K(timestamps, t, J)
        K.append(K_t)

        if K_t == 0:
            continue

        active_videos.append(t)

        # Pool all timestamps, sorted
        all_ts = []
        for j in range(J):
            all_ts.extend(timestamps[(j, t)])
        all_ts.sort()

        if len(all_ts) == 0:
            K[t] = 0
            active_videos.pop()
            continue

        # Initialize X*_k as evenly-spaced quantiles
        for k in range(K_t):
            quantile_pos = (k + 1) / (K_t + 1)
            idx = int(quantile_pos * len(all_ts))
            idx = min(idx, len(all_ts) - 1)
            X_star[(t, k)] = all_ts[idx]

    # Initial precision: std ~ 3s -> lambda = 1/9
    precision = np.full((J, N), 1.0 / 9.0)

    return K, active_videos, X_star, precision


# =====================================================================
# 4. E-Step
# =====================================================================

def e_step(timestamps, X_star, precision, K, active_videos,
           seg_starts, seg_ends, J):
    """
    E-Step: Compute responsibilities (soft alignment) and update truth.
    """
    responsibilities = {}
    max_shift = 0.0

    for t in active_videos:
        K_t = K[t]

        for j in range(J):
            ts = timestamps[(j, t)]
            n_jt = len(ts)

            if n_jt == 0:
                responsibilities[(j, t)] = np.zeros((0, K_t))
                continue

            r_matrix = np.zeros((n_jt, K_t))

            for i in range(n_jt):
                log_probs = np.zeros(K_t)
                for k in range(K_t):
                    log_probs[k] = log_gaussian(ts[i], X_star[(t, k)],
                                                precision[j, t])

                log_max = np.max(log_probs)
                probs = np.exp(log_probs - log_max)
                prob_sum = probs.sum()

                if prob_sum > 0:
                    r_matrix[i, :] = probs / prob_sum
                else:
                    r_matrix[i, :] = 1.0 / K_t

            responsibilities[(j, t)] = r_matrix

        # Update truth estimates
        for k in range(K_t):
            numerator = 0.0
            denominator = 0.0

            for j in range(J):
                ts = timestamps[(j, t)]
                n_jt = len(ts)
                if n_jt == 0:
                    continue

                r = responsibilities[(j, t)]
                lam = precision[j, t]

                for i in range(n_jt):
                    w = lam * r[i, k]
                    numerator += w * ts[i]
                    denominator += w

            if denominator > 1e-10:
                new_val = np.clip(numerator / denominator,
                                  seg_starts[t], seg_ends[t])
            else:
                new_val = X_star[(t, k)]

            shift = abs(new_val - X_star[(t, k)])
            if shift > max_shift:
                max_shift = shift

            X_star[(t, k)] = new_val

    return responsibilities, max_shift


# =====================================================================
# 5. M-Step
# =====================================================================

def m_step(timestamps, X_star, responsibilities, K, active_videos,
           precision, J, N, window_radius=5, precision_cap=4.0):
    """
    M-Step: Update annotator precision with Gaussian window smoothing
    and precision cap.
    """
    # Raw sufficient statistics
    N_eff_raw = np.zeros((J, N))
    S_raw = np.zeros((J, N))

    for t in active_videos:
        K_t = K[t]
        for j in range(J):
            ts = timestamps[(j, t)]
            n_jt = len(ts)
            if n_jt == 0:
                continue

            r = responsibilities[(j, t)]
            n_eff = 0.0
            s_err = 0.0

            for k in range(K_t):
                x_k = X_star[(t, k)]
                for i in range(n_jt):
                    r_ik = r[i, k]
                    n_eff += r_ik
                    s_err += r_ik * (ts[i] - x_k) ** 2

            N_eff_raw[j, t] = n_eff
            S_raw[j, t] = s_err

    # Gaussian window smoothing
    sigma_w = window_radius / 2.0

    for j in range(J):
        N_eff_smooth = np.zeros(N)
        S_smooth = np.zeros(N)

        for t in range(N):
            n_num = 0.0
            s_num = 0.0
            w_sum = 0.0

            start = max(0, t - window_radius)
            end = min(N - 1, t + window_radius)

            for t2 in range(start, end + 1):
                if K[t2] == 0:
                    continue

                w = np.exp(-((t2 - t) ** 2) / (2.0 * sigma_w ** 2))
                n_num += w * N_eff_raw[j, t2]
                s_num += w * S_raw[j, t2]
                w_sum += w

            if w_sum > 0:
                N_eff_smooth[t] = n_num / w_sum
                S_smooth[t] = s_num / w_sum

        # Precision with cap
        for t in range(N):
            n_eff = N_eff_smooth[t]
            s_err = S_smooth[t]

            if n_eff > 1e-10 and s_err > 1e-10:
                raw_precision = n_eff / s_err
                precision[j, t] = min(raw_precision, precision_cap)
            elif n_eff > 1e-10 and s_err <= 1e-10:
                precision[j, t] = precision_cap
            else:
                precision[j, t] = 1.0 / 9.0  # fallback prior

    return precision


# =====================================================================
# 6. Post-Processing: Pruning + Degenerate Component Merging
# =====================================================================

def prune_and_merge_degenerate(responsibilities, X_star, K, active_videos,
                               J, pruning_threshold=1.0,
                               degeneracy_threshold=0.5):
    """
    Stage 1: Prune steps with total responsibility R_k < pruning_threshold.
    Stage 2: Merge components within degeneracy_threshold (responsibility-weighted).
    """
    final_truths = {}
    total_before = 0
    total_pruned = 0
    total_merged = 0

    for t in active_videos:
        K_t = K[t]
        total_before += K_t

        kept = []
        for k in range(K_t):
            R_k = 0.0
            for j in range(J):
                r = responsibilities[(j, t)]
                if r.shape[0] > 0:
                    R_k += r[:, k].sum()

            if R_k >= pruning_threshold:
                kept.append((X_star[(t, k)], R_k))
            else:
                total_pruned += 1

        kept.sort(key=lambda x: x[0])

        if len(kept) > 1:
            merged = [kept[0]]
            for i in range(1, len(kept)):
                pos_curr, R_curr = kept[i]
                pos_prev, R_prev = merged[-1]

                if pos_curr - pos_prev < degeneracy_threshold:
                    R_total = R_prev + R_curr
                    pos_merged = (R_prev * pos_prev + R_curr * pos_curr) / R_total
                    merged[-1] = (pos_merged, R_total)
                    total_merged += 1
                else:
                    merged.append(kept[i])

            kept = merged

        final_truths[t] = [pos for pos, _ in kept]

    return final_truths, total_before, total_pruned, total_merged


# =====================================================================
# 7. Core Algorithm (callable, supports arbitrary J >= 2)
# =====================================================================

def run_truth_discovery_core(annotator_data_list,
                             window_radius=5,
                             max_iters=20,
                             convergence_threshold=2.0,
                             pruning_threshold=1.0,
                             precision_cap=4.0,
                             verbose=True):
    """
    Core truth discovery algorithm. Works with any J >= 2 annotators.

    Returns
    -------
    final_truths : dict {t: list of float} -- truth positions per video
    precision    : np.ndarray (J, N)       -- learned per-annotator-per-video precision
    timestamps   : dict {(j,t): list of float} -- input timestamps (for reuse)
    seg_starts, seg_ends : list of float
    active_videos : list of int
    N, J : int
    """
    timestamps, seg_starts, seg_ends, N, J = extract_annotations(
        annotator_data_list
    )

    K, active_videos, X_star, precision = initialize(timestamps, N, J)

    if verbose:
        n_empty = N - len(active_videos)
        total_slots = sum(K[t] for t in active_videos)
        K_values = [K[t] for t in active_videos] if active_videos else [0]
        print(f"  J={J}, N={N}, active={len(active_videos)}, empty={n_empty}")
        print(f"  Initial slots: {total_slots}, "
              f"K_t mean={np.mean(K_values):.2f}, max={np.max(K_values)}")

    # EM Loop
    responsibilities = None
    for iteration in range(max_iters):
        responsibilities, max_shift = e_step(
            timestamps, X_star, precision, K, active_videos,
            seg_starts, seg_ends, J
        )

        precision = m_step(
            timestamps, X_star, responsibilities, K, active_videos,
            precision, J, N,
            window_radius=window_radius, precision_cap=precision_cap
        )

        if verbose:
            mean_prec = [np.mean([precision[j, t] for t in active_videos])
                         for j in range(J)]
            prec_str = "  ".join(f"L_{j}={mp:.3f}"
                                 for j, mp in enumerate(mean_prec))
            print(f"  iter {iteration+1:3d}  shift={max_shift:7.4f}s  {prec_str}")

        if iteration > 0 and max_shift < convergence_threshold:
            if verbose:
                print(f"  Converged at iteration {iteration+1}")
            break

    # Post-processing
    final_truths, _, _, _ = prune_and_merge_degenerate(
        responsibilities, X_star, K, active_videos, J,
        pruning_threshold=pruning_threshold
    )

    return {
        'final_truths': final_truths,
        'precision': precision,
        'timestamps': timestamps,
        'seg_starts': seg_starts,
        'seg_ends': seg_ends,
        'active_videos': active_videos,
        'N': N,
        'J': J,
        'K': K,
        'responsibilities': responsibilities,
    }


# =====================================================================
# 8. Evaluation Metrics
# =====================================================================
