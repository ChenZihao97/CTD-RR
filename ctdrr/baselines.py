"""
Clustering and sequence-combination baselines used in the paper.

  HDBSCAN      Campello et al., 2013 - density clustering of pooled marks
  KDE + peaks  kernel density estimate of the pooled marks, peak picking
  BSC          Simpson & Gurevych, 2019 - Bayesian sequence combination

Copied unchanged from the working tree except for imports, so the released
code reproduces the reported numbers exactly.
"""

import numpy as np
from collections import Counter

from scipy.special import digamma, logsumexp
from scipy.signal import find_peaks
from sklearn.cluster import HDBSCAN

from .em import (
    extract_annotations,
    initialize,
    e_step,
    m_step,
    prune_and_merge_degenerate,
)

def _pool(timestamps, t, J):
    """All annotators' timestamps for video t, pooled and sorted."""
    out = []
    for j in range(J):
        out.extend(timestamps[(j, t)])
    out.sort()
    return out


def _clip_sort(vals, lo, hi):
    """Clip to the segment and return sorted floats."""
    return sorted(float(np.clip(v, lo, hi)) for v in vals)


def _median_K(timestamps, t, J):
    """Median of the non-zero per-annotator step counts (the B1/B2 rule)."""
    counts = [len(timestamps[(j, t)]) for j in range(J)]
    nz = [c for c in counts if c > 0]
    if not nz:
        return 0
    return int(round(float(np.median(nz))))


def _majority_K(timestamps, t, J):
    """Majority-vote step count, falling back to the median (the B3 rule)."""
    counts = [len(timestamps[(j, t)]) for j in range(J)]
    nz = [c for c in counts if c > 0]
    if not nz:
        return 0
    most_common, freq = Counter(nz).most_common(1)[0]
    if freq >= 2:
        return most_common
    return int(np.median(nz))


def _resolve_K(timestamps, t, J, rule):
    if rule == 'median':
        return _median_K(timestamps, t, J)
    if rule == 'majority':
        return _majority_K(timestamps, t, J)
    if rule == 'mean':
        counts = [len(timestamps[(j, t)]) for j in range(J)]
        nz = [c for c in counts if c > 0]
        return int(round(float(np.mean(nz)))) if nz else 0
    if rule == 'max':
        counts = [len(timestamps[(j, t)]) for j in range(J)]
        nz = [c for c in counts if c > 0]
        return max(nz) if nz else 0
    raise ValueError(f"unknown K rule: {rule}")


def _n_supporting(timestamps, t, J, center, radius):
    """How many distinct annotators placed a mark within `radius` of center."""
    n = 0
    for j in range(J):
        if any(abs(y - center) <= radius for y in timestamps[(j, t)]):
            n += 1
    return n


# =====================================================================
# B6. HDBSCAN
# =====================================================================

def baseline_hdbscan(annotator_data_list,
                     min_cluster_size=2,
                     min_samples=1,
                     cluster_selection_epsilon=0.0,
                     cluster_selection_method='eom',
                     center='median',
                     min_annotators=1,
                     **kwargs):
    """
    B6: pool the annotators' timestamps for each segment and run HDBSCAN on the
    resulting 1-D point cloud; each retained cluster contributes one boundary.

    Points labelled noise (-1) are discarded, which is the mechanism by which
    HDBSCAN suppresses isolated single-annotator marks. `min_annotators` adds
    the same >=2-annotator support filter used by the Mean Shift baseline (B4)
    so the two density methods are compared on equal terms; set it to 1 to
    disable.

    Degenerate segments (fewer points than `min_cluster_size`) fall back to a
    single boundary at the pooled median, since HDBSCAN is undefined there.
    """
    timestamps, seg_starts, seg_ends, N, J = extract_annotations(
        annotator_data_list
    )
    final_truths = {}

    for t in range(N):
        pool = _pool(timestamps, t, J)
        if not pool:
            continue

        lo, hi = seg_starts[t], seg_ends[t]

        if len(pool) < max(2, min_cluster_size):
            final_truths[t] = _clip_sort([float(np.median(pool))], lo, hi)
            continue

        X = np.asarray(pool, dtype=float).reshape(-1, 1)
        try:
            model = HDBSCAN(
                min_cluster_size=int(min_cluster_size),
                min_samples=int(min_samples),
                cluster_selection_epsilon=float(cluster_selection_epsilon),
                cluster_selection_method=cluster_selection_method,
                allow_single_cluster=True,
            )
            labels = model.fit_predict(X)
        except Exception:
            final_truths[t] = _clip_sort([float(np.median(pool))], lo, hi)
            continue

        centers = []
        for lab in sorted(set(labels)):
            if lab < 0:
                continue
            members = X[labels == lab, 0]
            c = float(np.median(members) if center == 'median'
                      else np.mean(members))
            if min_annotators > 1:
                # Support radius: the cluster's own spread, floored at 1s.
                radius = max(1.0, float(members.max() - members.min()))
                if _n_supporting(timestamps, t, J, c, radius) < min_annotators:
                    continue
            centers.append(c)

        if not centers:
            # Everything was labelled noise: fall back to the pooled median
            # rather than emitting nothing, which the metric harness would
            # score as a total miss for the whole segment.
            centers = [float(np.median(pool))]

        final_truths[t] = _clip_sort(centers, lo, hi)

    return final_truths


# =====================================================================
# B7. Dirichlet-Process GMM (infinite Gaussian mixture)
# =====================================================================

def baseline_kde_peaks(annotator_data_list,
                       bandwidth=2.0,
                       grid_step=0.1,
                       prominence_frac=0.05,
                       height_frac=0.0,
                       min_annotators=1,
                       **kwargs):
    """
    B8: build a Gaussian kernel density estimate over each segment's pooled
    timestamps and take its local maxima as boundaries.

    This is the continuous analogue of histogram voting: `bandwidth` (in
    seconds) sets the temporal resolution at which nearby marks are treated as
    the same event, and `prominence_frac` / `height_frac` (both as a fraction
    of the segment's peak density) filter out shallow bumps produced by single
    stray marks.

    A fixed bandwidth is used rather than Scott/Silverman because the sample
    size per segment is tiny and highly variable, which makes the plug-in rules
    swing wildly; the bandwidth is instead tuned on the same protocol as every
    other method here.
    """
    timestamps, seg_starts, seg_ends, N, J = extract_annotations(
        annotator_data_list
    )
    final_truths = {}
    h = float(bandwidth)

    for t in range(N):
        pool = _pool(timestamps, t, J)
        if not pool:
            continue

        lo, hi = seg_starts[t], seg_ends[t]
        if hi <= lo:
            final_truths[t] = _clip_sort([float(np.median(pool))], lo, hi)
            continue

        # Pad the grid by 3 bandwidths so peaks near the segment edges are not
        # clipped off by the boundary of the evaluation grid itself.
        g_lo, g_hi = lo - 3 * h, hi + 3 * h
        n_grid = int(np.ceil((g_hi - g_lo) / grid_step)) + 1
        n_grid = min(max(n_grid, 8), 200000)
        grid = np.linspace(g_lo, g_hi, n_grid)

        pts = np.asarray(pool, dtype=float)
        z = (grid[:, None] - pts[None, :]) / h
        dens = np.exp(-0.5 * z ** 2).sum(axis=1) / (h * np.sqrt(2 * np.pi))

        peak_max = float(dens.max())
        if peak_max <= 0:
            final_truths[t] = _clip_sort([float(np.median(pool))], lo, hi)
            continue

        idx, _ = find_peaks(
            dens,
            prominence=(prominence_frac * peak_max
                        if prominence_frac > 0 else None),
            height=(height_frac * peak_max if height_frac > 0 else None),
        )

        centers = [float(grid[i]) for i in idx if lo - 1e-6 <= grid[i] <= hi + 1e-6]

        if min_annotators > 1:
            centers = [c for c in centers
                       if _n_supporting(timestamps, t, J, c, h) >= min_annotators]

        if not centers:
            # Unimodal density with no interior maximum (e.g. all marks within
            # one bandwidth): use the global argmax.
            centers = [float(grid[int(np.argmax(dens))])]

        final_truths[t] = _clip_sort(centers, lo, hi)

    return final_truths


# =====================================================================
# B9. DTW Barycenter Averaging
# =====================================================================

def baseline_bsc(annotator_data_list,
                 bin_seconds=1.0,
                 n_vb_iter=15,
                 alpha_diag=10.0,
                 alpha_off=1.0,
                 beta_self=10.0,
                 beta_switch=1.0,
                 posterior_threshold=0.5,
                 merge_runs=True,
                 seq_model=True,
                 **kwargs):
    """
    B10: Bayesian Sequence Combination (Simpson & Gurevych, TACL 2019).

    The segment is discretised into `bin_seconds`-wide bins and each annotator
    is treated as producing a binary tag sequence (1 = boundary in this bin).
    The model has

      * a Markov prior over the true tag sequence, with a Dirichlet prior on
        the transition matrix — this is what distinguishes BSC from Dawid-Skene
        (B5), which treats bins as i.i.d. and so cannot express "boundaries are
        rare and never adjacent";
      * per-annotator confusion parameters. With `seq_model=True` (the "seq"
        variant of the paper) an annotator's tag also depends on their own
        previous tag, capturing the fact that a rater who just marked a
        boundary is unlikely to mark the very next bin.

    Inference is mean-field variational Bayes: forward-backward over the true
    tag sequence given expected log parameters, then Dirichlet posterior
    updates from the expected counts. Annotator parameters are shared across
    all segments (as in the paper) while the tag sequences are per-segment,
    so a single global VB run produces the whole corpus's boundaries.

    Implementation note: the forward-backward pass is vectorised across all
    segments at once, with each segment's own length enforced by an explicit
    mask so that zero-padding cannot leak into the backward messages.
    """
    timestamps, seg_starts, seg_ends, N, J = extract_annotations(
        annotator_data_list
    )
    bw = float(bin_seconds)

    # ---------------- Build the padded observation tensor ----------------
    lengths = np.zeros(N, dtype=int)
    for t in range(N):
        dur = seg_ends[t] - seg_starts[t]
        lengths[t] = max(1, int(np.ceil(dur / bw))) if dur > 0 else 1

    docs = [t for t in range(N) if any(timestamps[(j, t)] for j in range(J))]
    if not docs:
        return {}

    D = len(docs)
    L = int(lengths[docs].max())

    # c[j, d, b] in {0, 1}: annotator j's tag for bin b of document d
    c = np.zeros((J, D, L), dtype=np.int64)
    valid = np.zeros((D, L), dtype=bool)
    for d, t in enumerate(docs):
        n_b = lengths[t]
        valid[d, :n_b] = True
        for j in range(J):
            for y in timestamps[(j, t)]:
                b = int((y - seg_starts[t]) // bw)
                b = max(0, min(b, n_b - 1))
                c[j, d, b] = 1

    # c_prev[j, d, b]: annotator j's tag in the previous bin (0 at b = 0)
    c_prev = np.zeros_like(c)
    c_prev[:, :, 1:] = c[:, :, :-1]
    if not seq_model:
        c_prev[:] = 0  # collapses the seq model to a plain confusion matrix

    # ---------------- Dirichlet priors ----------------
    # alpha0[j, prev, z, obs] — annotator confusion
    alpha0 = np.full((J, 2, 2, 2), float(alpha_off))
    for prev in range(2):
        for z in range(2):
            alpha0[:, prev, z, z] = float(alpha_diag)
    # beta0[z_prev, z] — transitions of the true tag sequence. Both rows favour
    # tag 0: boundaries are rare, and two adjacent boundary bins rarer still.
    beta0 = np.array([[float(beta_self), float(beta_switch)],
                      [float(beta_self), float(beta_switch)]])

    alpha = alpha0.copy()
    beta = beta0.copy()
    pi0 = np.array([float(beta_self), float(beta_switch)])

    def _elog_dirichlet(a, axis=-1):
        return digamma(a) - digamma(a.sum(axis=axis, keepdims=True))

    lens_d = lengths[docs]
    q_z = None

    for _ in range(int(n_vb_iter)):
        E_log_pi = _elog_dirichlet(alpha)      # (J, 2, 2, 2)
        E_log_T = _elog_dirichlet(beta)        # (2, 2)
        E_log_p0 = _elog_dirichlet(pi0)        # (2,)

        # ---- Expected log-likelihood of each (doc, bin, true tag) ----
        ll = np.zeros((D, L, 2))
        for j in range(J):
            # Gather E_log_pi[j][prev, :, obs] for every (doc, bin). Mixing two
            # advanced indices around a slice puts the gathered axis first, so
            # the result is (D*L, 2) over the true tag z.
            gathered = E_log_pi[j][c_prev[j].ravel(), :, c[j].ravel()]
            ll += gathered.reshape(D, L, 2)
        ll[~valid] = 0.0

        # ---- Forward pass ----
        log_alpha = np.zeros((D, L, 2))
        log_alpha[:, 0, :] = E_log_p0[None, :] + ll[:, 0, :]
        for b in range(1, L):
            prev = log_alpha[:, b - 1, :]
            log_alpha[:, b, :] = logsumexp(
                prev[:, :, None] + E_log_T[None, :, :], axis=1
            ) + ll[:, b, :]

        # ---- Backward pass (mask-corrected at each document's own end) ----
        log_beta = np.zeros((D, L, 2))
        is_last = (np.arange(L)[None, :] >= (lens_d[:, None] - 1))
        for b in range(L - 2, -1, -1):
            nxt = ll[:, b + 1, :] + log_beta[:, b + 1, :]
            comp = logsumexp(E_log_T[None, :, :] + nxt[:, None, :], axis=2)
            log_beta[:, b, :] = np.where(is_last[:, b][:, None], 0.0, comp)

        # ---- Marginals ----
        log_q = log_alpha + log_beta
        log_q -= logsumexp(log_q, axis=2, keepdims=True)
        q_z = np.exp(log_q)
        q_z[~valid] = 0.0

        # ---- Pairwise marginals for the transition counts ----
        xi = np.zeros((2, 2))
        for b in range(1, L):
            m = valid[:, b]
            if not m.any():
                continue
            lx = (log_alpha[m, b - 1, :][:, :, None]
                  + E_log_T[None, :, :]
                  + (ll[m, b, :] + log_beta[m, b, :])[:, None, :])
            lx -= logsumexp(lx.reshape(lx.shape[0], -1), axis=1)[:, None, None]
            xi += np.exp(lx).sum(axis=0)

        # ---- M-step: Dirichlet posteriors from expected counts ----
        alpha = alpha0.copy()
        for j in range(J):
            for prev in range(2):
                for obs in range(2):
                    sel = valid & (c_prev[j] == prev) & (c[j] == obs)
                    if sel.any():
                        alpha[j, prev, :, obs] += q_z[sel].sum(axis=0)
        beta = beta0 + xi
        pi0 = np.array([float(beta_self), float(beta_switch)]) + \
            q_z[:, 0, :].sum(axis=0)

    # ---------------- Decode boundaries ----------------
    final_truths = {}
    for d, t in enumerate(docs):
        n_b = lengths[t]
        p1 = q_z[d, :n_b, 1]
        on = p1 > float(posterior_threshold)
        centers = []
        if merge_runs:
            # Collapse each contiguous run of "boundary" bins to a single
            # position, weighted by the posterior within the run.
            b = 0
            while b < n_b:
                if not on[b]:
                    b += 1
                    continue
                e = b
                while e + 1 < n_b and on[e + 1]:
                    e += 1
                idx = np.arange(b, e + 1)
                w = p1[idx]
                pos = float((idx + 0.5).dot(w) / w.sum()) * bw + seg_starts[t]
                centers.append(pos)
                b = e + 1
        else:
            for b in range(n_b):
                if on[b]:
                    centers.append(seg_starts[t] + (b + 0.5) * bw)

        if centers:
            final_truths[t] = _clip_sort(centers, seg_starts[t], seg_ends[t])

    return final_truths


# =====================================================================
# B11. Wasserstein barycenter of per-annotator timestamp distributions
# =====================================================================



# =====================================================================
# Registry
# =====================================================================

CLUSTERING_BASELINES = {
    'HDBSCAN': {'fn': baseline_hdbscan,
                'grid': {'min_cluster_size': [2, 3, 4],
                         'min_samples': [1, 2, 3],
                         'cluster_selection_epsilon': [0.0, 1.0, 2.0, 3.0, 5.0],
                         'cluster_selection_method': ['eom', 'leaf'],
                         'center': ['median', 'mean'],
                         'min_annotators': [1, 2]}},
    'KDE_peaks': {'fn': baseline_kde_peaks,
                  'grid': {'bandwidth': [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
                           'prominence_frac': [0.0, 0.02, 0.05, 0.10, 0.20],
                           'height_frac': [0.0, 0.05, 0.10],
                           'min_annotators': [1, 2]}},
    'BSC': {'fn': baseline_bsc,
            'grid': {'bin_seconds': [1.0, 2.0, 4.0],
                     'alpha_diag': [10.0, 50.0],
                     'posterior_threshold': [0.2, 0.3, 0.5],
                     'seq_model': [True, False],
                     'beta_self': [0.5, 2.0],
                     'beta_switch': [1.0, 5.0, 20.0],
                     'alpha_off': [2.0, 5.0],
                     'merge_runs': [True, False],
                     'n_vb_iter': [10]}},
}
