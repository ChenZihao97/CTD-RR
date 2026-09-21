"""
baselines_td.py
===============
Five *numerical* truth-discovery methods from the truth-discovery literature,
adapted to temporal boundary annotation and exposed through the same interface
as the clustering baselines:

    method(annotator_data_list, **kwargs) -> {video_index: [sorted seconds]}

  B12. CRH     Li et al., SIGMOD 2014  "Resolving Conflicts in Heterogeneous
                                       Data by Truth Discovery"
  B13. GTM     Zhao & Han, QDB@VLDB 2012  "A Probabilistic Model for Estimating
                                       Real-valued Truth from Conflicting
                                       Sources"
  B14. CATD    Li et al., VLDB 2015    "A Confidence-Aware Approach for Truth
                                       Discovery on Long-Tail Data"
  B15. KDEm    Wan et al., KDD 2016    "From Truth Discovery to Trustworthy
                                       Opinion Discovery"
  B16. EvolvT  Zhi et al., ICDM 2018   "Dynamic Truth Discovery on Numerical
                                       Data"  -- the paper NAMES its method
                                       EvolvT ("we propose a model named
                                       EvolvT"), despite the title.
  B17. DynaTD  Li et al., KDD 2015      "On the Discovery of Evolving Truth"
                                       -- and THIS paper names its method
                                       DynaTD ("denoted as Dynamic Truth
                                       Discovery, DynaTD for short").
                                       The names are swapped relative to what
                                       the two titles suggest; they are used
                                       here as the AUTHORS defined them.


Why an adaptation is needed at all
----------------------------------
Every one of these methods assumes the canonical truth-discovery input: an
*entity* about which each *source* supplies exactly ONE number.  Our input is a
video segment for which each annotator supplies a variable-length *set* of
timestamps, and the number of true boundaries is unknown.  Two things therefore
have to be supplied on top of each published method:

  (1) a rule for the number of boundaries K_t, and
  (2) an alignment of each annotator's marks to those K_t slots.

To keep the comparison about the methods rather than about the scaffolding,
CRH / GTM / CATD / DynaTD all sit on the SAME alignment layer
(`_aligned_claims`, below) with the SAME K-rules already used by B1-B11.  Each
(segment, slot) pair becomes one entity, each annotator is one source, and an
annotator that contributed no mark to a slot is *missing data* for that entity
- which all four methods handle natively.  What then differs between them is
exactly what the papers differ on: how a source's reliability is estimated and
how it enters the truth.

KDEm is the exception, and deliberately so.  KDEm was designed to output a
*set* of "opinions" per entity rather than a single truth, so it needs no
K-rule and no alignment: one segment is one entity, every annotator mark is a
claim, and the discovered modes are the boundaries.

CRH, GTM, CATD and KDEm estimate one reliability number per source for the
whole corpus.  DynaTD adds temporal dynamics on the truth -- a Kalman state
that evolves -- while its source quality stays constant over the timeline.

Sources for the formulas are given per method.  Where the widely-circulated
reference code (github.com/MengtingWan/KDEm, by the KDEm author, which ships
CRH.py / CATD.py / GTM.py as baselines) disagrees with the published paper, the
paper is followed and the discrepancy is recorded in the docstring.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from .em import extract_annotations
from .baselines import _pool, _clip_sort, _resolve_K


EPS = 1e-12


# =====================================================================
# Shared scaffolding: K selection, alignment, per-segment scale
# =====================================================================

def _segment_scale(timestamps, t, J, seg_starts, seg_ends, mode):
    """
    Per-segment scale used to make errors comparable across segments.

    CRH normalises its distance by a per-entity spread, and CATD 3.2.4 raises
    the same issue ("we can normalize the claims of the same entity so that the
    scale on all the entities falls into the same range").  Taking that at the
    *entity* level here would be degenerate - an entity is one boundary slot
    with at most J=3 claims, whose sample sd is frequently 0 - so the scale is
    estimated once per segment from all of its pooled marks, and shared by the
    slots inside it.  `span` uses the segment duration instead, which is
    defined even for segments where every annotator agrees exactly.
    """
    if mode == 'none':
        return 1.0
    if mode == 'span':
        return max(float(seg_ends[t] - seg_starts[t]), 1.0)
    pool = _pool(timestamps, t, J)
    if len(pool) < 2:
        return max(float(seg_ends[t] - seg_starts[t]), 1.0)
    sd = float(np.std(pool))
    return sd if sd > 1e-6 else max(float(seg_ends[t] - seg_starts[t]), 1.0)


def _init_truths(timestamps, t, J, K_t):
    """Evenly spaced quantiles of the pooled marks - the same initialisation
    CTD-RR uses, so no method gains or loses from a different starting point."""
    pool = _pool(timestamps, t, J)
    if not pool or K_t <= 0:
        return []
    out = []
    for k in range(K_t):
        q = (k + 1) / (K_t + 1)
        idx = min(int(q * len(pool)), len(pool) - 1)
        out.append(float(pool[idx]))
    return sorted(out)


def _assign(marks, truths):
    """
    Match one annotator's marks to the current truth slots, minimising total
    absolute error (Hungarian on a rectangular cost matrix, so the shorter side
    is matched fully and the surplus on the longer side is left unassigned).

    Returns {slot_index: mark_value}.  Slots with no mark are missing data.
    """
    if not marks or not truths:
        return {}
    cost = np.abs(np.asarray(marks, dtype=float)[:, None]
                  - np.asarray(truths, dtype=float)[None, :])
    rows, cols = linear_sum_assignment(cost)
    return {int(c): float(marks[int(r)]) for r, c in zip(rows, cols)}


class _Claims:
    """
    The corpus reshaped into the canonical truth-discovery layout.

    entities   list of (t, k)                     - one boundary slot each
    ent_of     {(t,k): entity index}
    src        list[np.ndarray]  per entity, the source ids that claimed it
    val        list[np.ndarray]  per entity, the claimed values (same order)
    scale      np.ndarray        per entity, the segment scale it inherits
    truth      np.ndarray        per entity, current truth estimate
    """

    def __init__(self, timestamps, seg_starts, seg_ends, N, J,
                 K_rule, scale_mode):
        self.J = J
        self.N = N
        self.timestamps = timestamps
        self.seg_starts = seg_starts
        self.seg_ends = seg_ends
        self.K = {}
        self.slots = {}          # t -> list of truth values
        self.entities = []
        self.ent_of = {}

        for t in range(N):
            K_t = _resolve_K(timestamps, t, J, K_rule)
            if K_t <= 0:
                continue
            init = _init_truths(timestamps, t, J, K_t)
            if not init:
                continue
            self.K[t] = len(init)
            self.slots[t] = list(init)
            for k in range(len(init)):
                self.ent_of[(t, k)] = len(self.entities)
                self.entities.append((t, k))

        self.n_ent = len(self.entities)
        self.truth = np.array([self.slots[t][k] for t, k in self.entities],
                              dtype=float)
        self.scale = np.array(
            [_segment_scale(timestamps, t, J, seg_starts, seg_ends, scale_mode)
             for t, _ in self.entities], dtype=float)
        self.src = [np.empty(0, dtype=int)] * self.n_ent
        self.val = [np.empty(0, dtype=float)] * self.n_ent
        self.realign()

    def realign(self):
        """Re-match every annotator's marks against the current truths."""
        src = [[] for _ in range(self.n_ent)]
        val = [[] for _ in range(self.n_ent)]
        for t, truths in self.slots.items():
            for j in range(self.J):
                for k, v in _assign(self.timestamps[(j, t)], truths).items():
                    e = self.ent_of[(t, k)]
                    src[e].append(j)
                    val[e].append(v)
        self.src = [np.array(s, dtype=int) for s in src]
        self.val = [np.array(v, dtype=float) for v in val]

    def push_truth(self):
        """Write `self.truth` back into the per-segment slot lists (sorted)."""
        for t in self.slots:
            vals = [self.truth[self.ent_of[(t, k)]] for k in range(self.K[t])]
            self.slots[t] = sorted(float(v) for v in vals)

    def source_counts(self):
        """|N_s| - the number of entities each source claims."""
        c = np.zeros(self.J)
        for s in self.src:
            if s.size:
                np.add.at(c, s, 1.0)
        return c

    def sq_errors(self, normalise=True):
        """Per-source sum of (normalised) squared errors against the truth."""
        out = np.zeros(self.J)
        for e in range(self.n_ent):
            s = self.src[e]
            if not s.size:
                continue
            d = (self.val[e] - self.truth[e]) ** 2
            if normalise:
                d = d / max(self.scale[e] ** 2, EPS)
            np.add.at(out, s, d)
        return out

    def weighted_truth(self, w):
        """x*_n = sum_s w_s v_s^n / sum_s w_s, over the sources claiming n."""
        out = np.array(self.truth, copy=True)
        for e in range(self.n_ent):
            s = self.src[e]
            if not s.size:
                continue
            ws = w[s]
            tot = float(ws.sum())
            if tot > EPS:
                out[e] = float(np.dot(ws, self.val[e]) / tot)
        return out

    def emit(self, prune_unclaimed=True, merge_within=0.0):
        """{segment index: sorted boundary seconds}, clipped to the segment."""
        final = {}
        for t in self.slots:
            vals = []
            for k in range(self.K[t]):
                e = self.ent_of[(t, k)]
                if prune_unclaimed and not self.src[e].size:
                    continue
                vals.append(float(self.truth[e]))
            vals.sort()
            if merge_within > 0 and len(vals) > 1:
                merged = [vals[0]]
                for v in vals[1:]:
                    if v - merged[-1] < merge_within:
                        merged[-1] = 0.5 * (merged[-1] + v)
                    else:
                        merged.append(v)
                vals = merged
            if vals:
                final[t] = _clip_sort(vals, self.seg_starts[t],
                                      self.seg_ends[t])
        return final


def _build(annotator_data_list, K_rule, scale_mode):
    timestamps, seg_starts, seg_ends, N, J = extract_annotations(
        annotator_data_list)
    return _Claims(timestamps, seg_starts, seg_ends, N, J, K_rule, scale_mode)


# =====================================================================
# B12. CRH  (Li, Gao, Meng, Li, Su, Zhao, Fan, Han - SIGMOD 2014)
# =====================================================================

def baseline_crh(annotator_data_list,
                 K_rule='median',
                 scale_mode='segment_std',
                 max_itr=30,
                 tol=1e-3,
                 realign_every=1,
                 merge_within=0.0,
                 **kwargs):
    """
    CRH solves

        min_{X*, W}  sum_s w_s sum_{o in O_s} d(v_s^o, x*_o)
        s.t.         sum_s exp(-w_s) = 1

    by block coordinate descent.  For continuous data the loss is the
    normalised squared deviation, and the two closed-form updates are

        w_s   = -log( delta_s / sum_s' delta_s' ),
                delta_s = sum_o (v_s^o - x*_o)^2 / std_o
        x*_o  = sum_s w_s v_s^o / sum_s w_s

    The -log form is what enforces the exp(-w) constraint: a source whose share
    of the total loss is small gets a large positive weight, and the weights are
    scale-free.

    Note on `std_o`: the author's reference implementation divides the squared
    deviation by the *standard deviation* of the claims for that entity, not by
    the variance.  That is followed here (`scale_mode` chooses what plays the
    role of std_o; see `_segment_scale` for why it is estimated per segment
    rather than per slot).
    """
    C = _build(annotator_data_list, K_rule, scale_mode)
    if C.n_ent == 0:
        return {}

    w = np.ones(C.J)
    prev = np.array(C.truth, copy=True)

    for itr in range(max_itr):
        C.truth = C.weighted_truth(w)

        # CRH's normalisation is by std_o, i.e. a single power of the scale.
        delta = np.zeros(C.J)
        for e in range(C.n_ent):
            s = C.src[e]
            if not s.size:
                continue
            np.add.at(delta, s,
                      (C.val[e] - C.truth[e]) ** 2 / max(C.scale[e], EPS))
        tot = float(delta.sum())
        if tot > 0:
            pos = delta > 0
            w = np.ones(C.J)
            w[pos] = -np.log(delta[pos] / tot)
            # exp(-w) must stay a probability vector; a source with zero loss
            # would otherwise take infinite weight.
            w = np.clip(w, 1e-6, 50.0)

        denom = np.linalg.norm(prev)
        err = np.linalg.norm(C.truth - prev) / denom if denom > 0 else 99.0
        prev = np.array(C.truth, copy=True)
        if err < tol and itr > 0:
            break
        if realign_every and (itr + 1) % realign_every == 0:
            C.push_truth()
            C.realign()

    C.push_truth()
    return C.emit(merge_within=merge_within)


# =====================================================================
# B13. GTM  (Zhao & Han - QDB @ VLDB 2012)
# =====================================================================

def baseline_gtm(annotator_data_list,
                 K_rule='median',
                 scale_mode='segment_std',
                 alpha=10.0,
                 beta=10.0,
                 mu0=0.0,
                 sigma0=1.0,
                 max_itr=30,
                 tol=1e-3,
                 realign_every=1,
                 merge_within=0.0,
                 **kwargs):
    """
    GTM is the Bayesian counterpart of CRH:

        mu_o     ~ N(mu_0, sigma_0^2)
        sigma_s^2 ~ Inv-Gamma(alpha, beta)
        v_s^o    ~ N(mu_o, sigma_s^2)

    with EM updates

        mu_o        = ( mu_0/sigma_0^2 + sum_s v_s^o/sigma_s^2 )
                      / ( 1/sigma_0^2  + sum_s 1/sigma_s^2 )
        sigma_s^2   = ( 2 beta + sum_o (v_s^o - mu_o)^2 )
                      / ( 2(alpha+1) + |O_s| )

    Unlike CRH's normalised loss, GTM's prior is stated on the *raw* scale, so
    the model only makes sense once observations are on a common scale.  The
    paper normalises each entity's claims; here the values are z-scored per
    segment before the EM and mapped back afterwards, which is the same idea at
    the level at which a spread is actually estimable (see `_segment_scale`).
    The Inv-Gamma prior is what keeps a source that happens to agree exactly on
    its few slots from acquiring an unbounded precision.
    """
    C = _build(annotator_data_list, K_rule, scale_mode)
    if C.n_ent == 0:
        return {}

    # Work in normalised units: v~ = (v - centre_t)/scale_t.
    centre = np.zeros(C.n_ent)
    for e, (t, _) in enumerate(C.entities):
        pool = _pool(C.timestamps, t, C.J)
        centre[e] = float(np.mean(pool)) if pool else 0.0

    def to_norm(e, v):
        return (v - centre[e]) / max(C.scale[e], EPS)

    def from_norm(e, v):
        return v * max(C.scale[e], EPS) + centre[e]

    sigma = np.ones(C.J)
    prev = np.array(C.truth, copy=True)

    for itr in range(max_itr):
        # ---- E-step: posterior mean of each entity's truth ----------
        for e in range(C.n_ent):
            s = C.src[e]
            if not s.size:
                continue
            prec = 1.0 / np.maximum(sigma[s] ** 2, EPS)
            num = mu0 / max(sigma0 ** 2, EPS) + float(
                np.dot(prec, to_norm(e, C.val[e])))
            den = 1.0 / max(sigma0 ** 2, EPS) + float(prec.sum())
            if den > EPS:
                C.truth[e] = from_norm(e, num / den)

        # ---- M-step: per-source variance with the Inv-Gamma prior ---
        num = np.full(C.J, 2.0 * beta)
        cnt = np.zeros(C.J)
        for e in range(C.n_ent):
            s = C.src[e]
            if not s.size:
                continue
            d = (to_norm(e, C.val[e]) - to_norm(e, C.truth[e])) ** 2
            np.add.at(num, s, d)
            np.add.at(cnt, s, 1.0)
        sigma = np.sqrt(np.maximum(num / (2.0 * (alpha + 1.0) + cnt), EPS))

        denom = np.linalg.norm(prev)
        err = np.linalg.norm(C.truth - prev) / denom if denom > 0 else 99.0
        prev = np.array(C.truth, copy=True)
        if err < tol and itr > 0:
            break
        if realign_every and (itr + 1) % realign_every == 0:
            C.push_truth()
            C.realign()

    C.push_truth()
    return C.emit(merge_within=merge_within)


# =====================================================================
# B14. CATD  (Li, Li, Gao, Su, Zhao, Fan, Han - VLDB 2015)
# =====================================================================

def baseline_catd(annotator_data_list,
                  K_rule='median',
                  scale_mode='segment_std',
                  alpha_sig=0.05,
                  max_itr=30,
                  tol=1e-3,
                  realign_every=1,
                  merge_within=0.0,
                  **kwargs):
    """
    CATD weights a source by the *upper confidence bound* on its error
    variance rather than by the point estimate, so that a source with few
    claims cannot earn a high weight from a lucky small sample.  Paper Eq. (7):

        w_s  ~  1 / u_s^2  =  chi2_{(alpha/2, |N_s|)}
                              / sum_{n in N_s} (x_s^n - x*_n)^2

    and the truth is the weighted average, Eq. (1).

    The chi-square convention matters and is easy to get backwards.  Eq. (5)
    gives the confidence interval for sigma_s^2 as
        ( SSE / chi2_{(1-alpha/2,|N_s|)} ,  SSE / chi2_{(alpha/2,|N_s|)} ),
    so chi2_{(alpha/2, n)} is the LOWER 2.5% quantile - `chi2.ppf(0.025, n)`,
    not an upper tail point.  Verified against the paper's own worked example
    (Table 5: 200 claims, sigma^2 = 0.1 -> CI (0.0830, 0.1229); 20/162.73 =
    0.1229 and chi2.ppf(0.025, 200) = 162.73).

    The consequence is the long-tail behaviour CATD is named for: with
    SSE ~ n sigma_s^2, the weight behaves like chi2.ppf(0.025,n)/(n sigma_s^2),
    which is ~0.001/sigma_s^2 at n=1 and rises towards 1/sigma_s^2 as n grows.
    Sources with few claims are pulled towards zero weight.

    The author's public reference implementation uses `chi2.cdf(0.025, n)`
    here, which is a tail *probability* rather than a quantile and decreases in
    n - the opposite direction to the paper.  The paper is followed.
    """
    C = _build(annotator_data_list, K_rule, scale_mode)
    if C.n_ent == 0:
        return {}

    w = np.ones(C.J)
    prev = np.array(C.truth, copy=True)

    for itr in range(max_itr):
        C.truth = C.weighted_truth(w)

        sse = C.sq_errors(normalise=True)
        cnt = C.source_counts()
        w = np.zeros(C.J)
        for j in range(C.J):
            if cnt[j] <= 0:
                continue
            q = float(chi2.ppf(alpha_sig / 2.0, max(cnt[j], 1.0)))
            w[j] = q / max(sse[j], EPS)
        if not np.any(w > 0):
            w = np.ones(C.J)
        else:
            # Rescale for numerical comfort only; the truth update is
            # invariant to a common positive factor on the weights.
            w = w / max(w.max(), EPS)

        denom = np.linalg.norm(prev)
        err = np.linalg.norm(C.truth - prev) / denom if denom > 0 else 99.0
        prev = np.array(C.truth, copy=True)
        if err < tol and itr > 0:
            break
        if realign_every and (itr + 1) % realign_every == 0:
            C.push_truth()
            C.realign()

    C.push_truth()
    return C.emit(merge_within=merge_within)


# =====================================================================
# B15. KDEm  (Wan, Li, Chen, Gao, Kaplan, Zhao, Han - KDD 2016)
# =====================================================================

def _kdem_K(x, method):
    """KDEm's kernels, in the author's parameterisation.  Note the Gaussian is
    exp(-x^2)/sqrt(2 pi), not exp(-x^2/2): the bandwidth is therefore sqrt(2)
    times a textbook Gaussian bandwidth.  Kept as published."""
    m = method.lower()
    if m == 'uniform':
        return (np.abs(x) <= 1) / 2.0
    if m in ('epanechnikov', 'ep'):
        return 0.75 * (1 - x ** 2) * (np.abs(x) <= 1)
    if m in ('biweight', 'bi'):
        return (15.0 / 16.0) * (1 - x ** 2) ** 2 * (np.abs(x) <= 1)
    if m in ('triweight', 'tri'):
        return (35.0 / 32.0) * (1 - x ** 2) ** 3 * (np.abs(x) <= 1)
    if m == 'laplace':
        return np.exp(-np.abs(x))
    return np.exp(-x ** 2) / np.sqrt(2 * np.pi)


def _mad(x):
    return float(np.median(np.abs(x - np.median(x)))) + 1e-10 * float(np.std(x))


def _denclue(x, w, h, method, tol=1e-8, max_itr=300):
    """Weighted mean-shift: every claim is walked uphill on the weighted KDE
    until it lands on a mode."""
    if np.var(x) <= 0:
        return np.array(x, copy=True)
    if w.sum() <= 0:
        w = w + 1e-5
    cur = np.array(x, dtype=float) + 1e-12
    for _ in range(max_itr):
        z = (cur[:, None] - x[None, :]) / h
        kk = _kdem_K(z, method)
        num = kk @ (w * x)
        den = kk @ w
        nxt = np.where(den > 0, num / np.maximum(den, EPS), cur)
        n0 = np.linalg.norm(cur)
        err = np.linalg.norm(cur - nxt) / n0 if n0 > 0 else 0.0
        cur = nxt
        if err <= tol:
            break
    return cur


def baseline_kdem(annotator_data_list,
                  kernel='gaussian',
                  bandwidth=-1.0,
                  bandwidth_mult=1.0,
                  max_itr=30,
                  tol=1e-5,
                  cut=0.0,
                  merge_tol=0.5,
                  min_annotators=1,
                  **kwargs):
    """
    KDEm is the only one of the five that natively returns a *set* of truths
    per entity, so it needs neither a K-rule nor an alignment: one segment is
    one entity and every annotator mark is a claim.

    Two alternating updates, exactly as published:

      loss   the claim's squared distance to the weighted mean embedding in the
             kernel's RKHS,
                 ||phi(x_a) - sum_b w_b phi(x_b)||^2
               = K(x_a,x_a) - 2 (K w)_a + w' K w
      c_s    source reliability, CRH-style from the mean loss,
                 c_s = -log( (L_s / |N_s|) / sum_s' L_s' )
      w      within an entity, claim weights proportional to c_s.

    Opinions are then the modes of the weighted KDE, found by DENCLUE
    mean-shift, merged, and scored by their normalised density (`cut` drops
    modes below a share of the total density - KDEm's own confidence filter).

    One extension is needed and is flagged as such: in KDEm each source makes
    ONE claim per entity, whereas an annotator here contributes several marks
    to one segment.  Nothing in the update rules assumes uniqueness - the loss
    is per claim and the reliability aggregates over a source's claims - so the
    source index simply repeats within an entity.  `|N_s|` counts claims.

    `merge_tol` is in *seconds*.  The reference implementation merges converged
    points by relative difference, which on absolute timestamps would make the
    merge radius grow with position in the video; an absolute radius is used
    instead.
    """
    timestamps, seg_starts, seg_ends, N, J = extract_annotations(
        annotator_data_list)

    ents = []            # (t, values, source ids)
    for t in range(N):
        vals, srcs = [], []
        for j in range(J):
            for v in timestamps[(j, t)]:
                vals.append(float(v))
                srcs.append(j)
        if vals:
            ents.append((t, np.array(vals), np.array(srcs, dtype=int)))
    if not ents:
        return {}

    # Fixed per-entity kernel matrices (claims never move, only their weights).
    kmats, bws = [], []
    for _, x, _ in ents:
        h = _mad(x) if bandwidth < 0 else float(bandwidth)
        h = max(h * bandwidth_mult, 1e-6)
        bws.append(h)
        kmats.append(_kdem_K((x[:, None] - x[None, :]) / h, kernel))

    w_M = [np.ones(len(x)) / len(x) for _, x, _ in ents]
    cnt = np.zeros(J)
    for _, _, s in ents:
        np.add.at(cnt, s, 1.0)
    cnt = np.maximum(cnt, 1.0)

    def losses():
        out = []
        for i, km in enumerate(kmats):
            w = w_M[i]
            t2 = km @ w
            t3 = float(w @ t2)
            out.append(np.maximum(np.diag(km) - 2 * t2 + t3, 0.0))
        return out

    def update_c(L):
        raw = np.zeros(J)
        for i, (_, _, s) in enumerate(ents):
            np.add.at(raw, s, L[i] / len(s))
        tot = float(raw.sum())
        c = np.ones(J)
        if tot > 0:
            pos = raw > 0
            c[pos] = -np.log((raw[pos] / cnt[pos]) / tot)
        return c, tot

    L = losses()
    c, J_obj = update_c(L)
    for _ in range(max_itr):
        J_old = J_obj
        for i, (_, _, s) in enumerate(ents):
            wi = np.where(L[i] > 0, c[s], 0.0)
            tot = float(wi.sum())
            w_M[i] = wi / tot if tot > 0 else np.ones(len(s)) / len(s)
        L = losses()
        c, J_obj = update_c(L)
        if J_old > 0 and abs((J_obj - J_old) / J_old) <= tol:
            break

    final = {}
    for i, (t, x, s) in enumerate(ents):
        h = bws[i]
        w = w_M[i]
        if w.sum() <= 0:
            w = np.ones(len(x)) / len(x)
        conv = _denclue(x, w, h, kernel)

        order = np.argsort(conv)
        centers, members = [], []
        for idx in order:
            v = float(conv[idx])
            if centers and v - centers[-1] <= merge_tol:
                members[-1].append(int(idx))
            else:
                centers.append(v)
                members.append([int(idx)])

        dens = []
        for cpos in centers:
            d = float(np.dot(w, _kdem_K((cpos - x) / h, kernel))
                      / (h * max(w.sum(), EPS)))
            dens.append(d)
        dens = np.array(dens)
        keep = list(range(len(centers)))
        if cut > 0 and dens.sum() > 0:
            share = dens / dens.sum()
            keep = [i2 for i2 in keep if share[i2] > cut]
        if min_annotators > 1:
            keep = [i2 for i2 in keep
                    if len(set(s[members[i2]])) >= min_annotators]
        if not keep:
            keep = [int(np.argmax(dens))] if len(dens) else []

        # Re-centre each surviving mode on its members' weighted mean, so the
        # reported boundary is a real timestamp aggregate rather than the
        # arbitrary point mean-shift happened to stop at.
        out = []
        for i2 in keep:
            mem = members[i2]
            ww = w[mem]
            out.append(float(np.dot(ww, x[mem]) / max(ww.sum(), EPS))
                       if ww.sum() > 0 else centers[i2])
        if out:
            final[t] = _clip_sort(out, seg_starts[t], seg_ends[t])
    return final


# =====================================================================
# B16. EvolvT  (Zhi, Yang, Zhu, Li, Wang, Han - ICDM 2018)
#      The ICDM 2018 paper names its own method EvolvT, not DynaTD.
# =====================================================================

def baseline_evolvt_kalman(annotator_data_list,
                    K_rule='median',
                    alpha=2.0,
                    beta=1.0,
                    learn_A=True,
                    max_itr=15,
                    tol=1e-4,
                    realign_every=1,
                    merge_within=0.0,
                    init_sigma=0.1,
                    **kwargs):
    """
    DynaTD models the truth as a latent state that evolves along a timeline and
    is observed, noisily, by several sources:

        mu_{t+1} = A mu_t + omega_t ,  omega_t ~ N(0, Gamma)      (Eq. 1)
        mu_1     ~ N(pi_1, V_1)                                   (Eq. 2)
        v_t      = C mu_t + eps_t   ,  eps_t   ~ N(0, Pi)         (Eq. 4-5)

    with source quality on the diagonal of Sigma.  Inference is EM with a
    Kalman filter and RTS smoother in the E-step (Eq. 7-8) and closed-form
    updates in the M-step: sigma_i^2 by Eq. (9) under an Inv-Gamma(alpha,beta)
    prior, pi_1/V_1 by Eq. (13), A by Eq. (14) and Gamma by Eq. (15).

    WHAT PLAYS THE ROLE OF TIME.  DynaTD needs a sequence of observations of
    the same entity.  Our corpus has no such axis at the segment level -
    segments are independent and their order is an artefact of the annotation
    session - but *inside* a segment the boundaries x_1 < x_2 < ... < x_K are
    exactly that: an ordered sequence, observed by every annotator, with
    annotators missing individual elements.  So one segment is one object block
    and the boundary index is the timeline.  A then encodes how boundary
    position advances from one step to the next and Gamma how regular that
    advance is - a learned prior on step spacing - while sigma_i^2 is the
    annotator's timing noise, pooled over the whole corpus as in the paper.

    Timestamps are normalised to [0,1] within the segment before the EM and
    mapped back afterwards, so that pi_1, A and Gamma are comparable across
    segments of different durations.  The paper's Section III-D asks for the
    same ("the normalization of observations is also necessary").

    Sources are taken as independent, so Sigma is diagonal and the blocked
    filter of Section III-B3 reduces to a scalar filter per segment.  The paper
    sanctions this case explicitly ("If we assume sources are not correlated to
    each other, sigma_{i,i'} can also set to 0"); the inverse-Wishart update of
    Eq. (10) for dependent sources is not used here.

    DynaTD's dynamics are on the
    TRUTH, and its source quality sigma_i^2 is a single number per source for
    the whole timeline.  CTD-RR puts the dynamics on the SOURCE instead.
    """
    C = _build(annotator_data_list, K_rule, scale_mode='none')
    if C.n_ent == 0:
        return {}
    J = C.J

    segs = sorted(C.slots)
    lo = {t: float(C.seg_starts[t]) for t in segs}
    span = {t: max(float(C.seg_ends[t] - C.seg_starts[t]), 1e-6) for t in segs}

    sigma2 = np.full(J, float(init_sigma) ** 2)
    A = 1.0
    Gam = 0.05
    pi1, V1 = 0.3, 0.1

    prev = np.array(C.truth, copy=True)

    for itr in range(max_itr):
        # Observations in normalised units, per segment, per slot.
        obs = {}
        for t in segs:
            Kt = C.K[t]
            seq = []
            for k in range(Kt):
                e = C.ent_of[(t, k)]
                s, v = C.src[e], C.val[e]
                seq.append((s, (v - lo[t]) / span[t]))
            obs[t] = seq

        # ---- E-step: scalar Kalman filter + RTS smoother per segment ----
        smoothed = {}
        stats = {'S11': 0.0, 'S10': 0.0, 'S00': 0.0, 'n_trans': 0,
                 'm1': [], 'v1': []}
        for t in segs:
            seq = obs[t]
            T = len(seq)
            mp, Vp = np.zeros(T), np.zeros(T)     # predicted
            mf, Vf = np.zeros(T), np.zeros(T)     # filtered
            for k in range(T):
                if k == 0:
                    mp[k], Vp[k] = pi1, V1
                else:
                    mp[k] = A * mf[k - 1]
                    Vp[k] = A * Vf[k - 1] * A + Gam
                s, v = seq[k]
                if s.size:
                    prec = 1.0 / np.maximum(sigma2[s], EPS)
                    # Scalar Gaussian update with several observations.
                    tot = float(prec.sum())
                    mo = float(np.dot(prec, v) / tot)
                    Ro = 1.0 / tot
                    Kg = Vp[k] / (Vp[k] + Ro)
                    mf[k] = mp[k] + Kg * (mo - mp[k])
                    Vf[k] = (1.0 - Kg) * Vp[k]
                else:
                    mf[k], Vf[k] = mp[k], Vp[k]

            ms, Vs = np.array(mf), np.array(Vf)
            Vlag = np.zeros(T)
            for k in range(T - 2, -1, -1):
                Jk = A * Vf[k] / max(Vp[k + 1], EPS)
                ms[k] = mf[k] + Jk * (ms[k + 1] - mp[k + 1])
                Vs[k] = Vf[k] + Jk * (Vs[k + 1] - Vp[k + 1]) * Jk
                Vlag[k + 1] = Jk * Vs[k + 1]
            smoothed[t] = (ms, Vs)

            stats['m1'].append(ms[0])
            stats['v1'].append(Vs[0])
            for k in range(1, T):
                stats['S11'] += Vs[k] + ms[k] ** 2
                stats['S10'] += Vlag[k] + ms[k] * ms[k - 1]
                stats['S00'] += Vs[k - 1] + ms[k - 1] ** 2
                stats['n_trans'] += 1

        # ---- M-step -----------------------------------------------------
        # Eq. (9): sigma_i^2 with the Inv-Gamma(alpha, beta) prior.
        num = np.full(J, 2.0 * beta)
        cnt = np.zeros(J)
        for t in segs:
            ms, Vs = smoothed[t]
            for k, (s, v) in enumerate(obs[t]):
                if not s.size:
                    continue
                d = v ** 2 - 2 * v * ms[k] + ms[k] ** 2 + Vs[k]
                np.add.at(num, s, d)
                np.add.at(cnt, s, 1.0)
        sigma2 = np.maximum(num / (2.0 * (alpha + 1.0) + cnt), 1e-8)

        # Eq. (14) and (15).
        if learn_A and stats['S00'] > EPS and stats['n_trans'] > 0:
            A = float(np.clip(stats['S10'] / stats['S00'], 0.1, 3.0))
        if stats['n_trans'] > 0:
            Gam = float(max(
                (stats['S11'] - A * stats['S10']) / stats['n_trans'], 1e-6))
        # Eq. (13), pooled over segments (each segment is its own block, so the
        # initial-state parameters are shared rather than per block).
        if stats['m1']:
            pi1 = float(np.mean(stats['m1']))
            V1 = float(max(np.mean(stats['v1']) + np.var(stats['m1']), 1e-6))

        # Write the smoothed states back as truths, in seconds.
        for t in segs:
            ms, _ = smoothed[t]
            for k in range(C.K[t]):
                C.truth[C.ent_of[(t, k)]] = ms[k] * span[t] + lo[t]

        denom = np.linalg.norm(prev)
        err = np.linalg.norm(C.truth - prev) / denom if denom > 0 else 99.0
        prev = np.array(C.truth, copy=True)
        if err < tol and itr > 0:
            break
        if realign_every and (itr + 1) % realign_every == 0:
            C.push_truth()
            C.realign()

    C.push_truth()
    return C.emit(merge_within=merge_within)


# =====================================================================
# Registry
# =====================================================================
# All five estimate source reliability over the WHOLE corpus, exactly as the
# papers do, so none of them can be tuned on a subset of segments the way the
# per-segment baselines (B6-B9, B11) can: `per_video` is False throughout.

ADVANCED_BASELINES = {
    "B12_CRH": {
        "fn": baseline_crh,
        "per_video": False,
        "default": dict(K_rule='median', scale_mode='segment_std',
                        max_itr=30, realign_every=1),
        "grid": {
            "K_rule": ['median', 'majority', 'max'],
            "scale_mode": ['segment_std', 'span', 'none'],
            "realign_every": [1, 0],
            "merge_within": [0.0, 0.5],
        },
    },
    "B13_GTM": {
        "fn": baseline_gtm,
        "per_video": False,
        "default": dict(K_rule='median', scale_mode='segment_std',
                        alpha=10.0, beta=10.0),
        "grid": {
            "K_rule": ['median', 'majority', 'max'],
            "alpha": [1.0, 10.0, 50.0],
            "beta": [1.0, 10.0, 50.0],
            "sigma0": [1.0, 3.0],
            "realign_every": [1, 0],
        },
    },
    "B14_CATD": {
        "fn": baseline_catd,
        "per_video": False,
        "default": dict(K_rule='median', scale_mode='segment_std',
                        alpha_sig=0.05),
        "grid": {
            "K_rule": ['median', 'majority', 'max'],
            "scale_mode": ['segment_std', 'span', 'none'],
            "alpha_sig": [0.01, 0.05, 0.10],
            "realign_every": [1, 0],
        },
    },
    "B15_KDEm": {
        "fn": baseline_kdem,
        "per_video": True,
        "default": dict(kernel='gaussian', bandwidth=-1.0, bandwidth_mult=1.0,
                        cut=0.0, merge_tol=0.5),
        "grid": {
            "kernel": ['gaussian', 'ep', 'triweight'],
            "bandwidth": [-1.0, 1.0, 2.0, 3.0],
            "bandwidth_mult": [1.0, 2.0],
            "cut": [0.0, 0.05, 0.10],
            "merge_tol": [0.5, 1.0],
            "min_annotators": [1, 2],
        },
    },
    "B16_EvolvT": {
        "fn": baseline_evolvt_kalman,
        "per_video": False,
        "default": dict(K_rule='median', alpha=2.0, beta=1.0, learn_A=True),
        "grid": {
            "K_rule": ['median', 'majority', 'max'],
            "alpha": [1.0, 2.0, 10.0],
            "beta": [0.1, 1.0, 10.0],
            "learn_A": [True, False],
            "init_sigma": [0.05, 0.1],
        },
    },
}


# =====================================================================
# B17 / B18. DynaTD  (Li, Li, Gao, Su, Zhao, Demirbas, Fan, Han - KDD 2015)
#            "On the Discovery of Evolving Truth"
# =====================================================================
#
# DynaTD (KDD 2015) is the direct predecessor of EvolvT (ICDM 2018), and the
# method EvolvT compares itself against - reference [10] there, with DynaTD's
# equations restated as Eq. 16-17.  NOTE THE NAMING: the ICDM 2018 paper titled
# "Dynamic Truth Discovery..." names its method EvolvT, and the KDD 2015 paper
# titled "...Evolving Truth" names its method DynaTD.  The names are the
# opposite way round from what the titles suggest; both are used here exactly
# as their own authors defined them.
#
# The model, from the paper (loss at timestamp t):
#
#   l_t = theta * sum_s sum_o w_s (v_{o,ts} - x*_{o,t})^2
#         - sum_s c_{ts} log(w_s)
#         + theta * lambda * sum_o (x*_{o,t-1} - x*_{o,t})^2
#
# giving two closed-form updates:
#
#   source weight, with an exponential decay gamma on past errors
#     w_s = [ 2 alpha - 2 + sum_t gamma^{T-t} c_t^s ]
#           / [ 2 beta + theta sum_t sum_o gamma^{T-t} (v_{o,ts} - x*_{o,t})^2 ]
#
#   truth, treating the previous timestamp's truth as a pseudo-source of
#   weight lambda
#     x*_{o,t} = [ sum_s w_s v_{o,ts} + lambda x*_{o,t-1} ]
#                / [ sum_s w_s + lambda ]
#
# Two things distinguish it from DynaTD, and both are the point of having it:
# its balancing parameter lambda is FIXED where DynaTD's Kalman gain is
# dynamic, and its source weight carries a RECENCY DECAY gamma where DynaTD's
# sigma_i^2 is constant over the timeline.
#
# That second property is why EvolvT gets two mappings here, not one.


def _evolvt_weights(alpha, beta, theta, cnt, sse):
    """Eq. (9)/(12): w_s from decayed claim counts and decayed squared error."""
    return np.maximum(2.0 * alpha - 2.0 + cnt, EPS) / \
        np.maximum(2.0 * beta + theta * sse, EPS)


def baseline_dynatd_kdd15(annotator_data_list,
                    K_rule='median',
                    alpha=2.0,
                    beta=1.0,
                    theta=1.0,
                    lam=1.0,
                    gamma=1.0,
                    max_itr=10,
                    tol=1e-3,
                    realign_every=1,
                    merge_within=0.0,
                    **kwargs):
    """
    B17: EvolvT on the SAME mapping used for DynaTD, so the two are directly
    comparable — one segment is one object block and the boundary index is the
    timeline, with timestamps normalised to [0,1] inside the segment.

    On that mapping `lambda` pulls boundary k towards boundary k-1.  EvolvT
    assumes the truth is roughly constant along the timeline (it has no
    transition term; DynaTD's A is exactly what EvolvT lacks), and a boundary
    sequence is strictly increasing, so the smoothing is a compression.  That
    is a faithful consequence of the model rather than a defect of the port,
    and the tuning is free to drive `lam` to 0, which is the informative
    outcome to report: it says the same thing as DynaTD selecting `learn_A =
    False` and a huge alpha, namely that the within-segment timeline is not
    where these methods' temporal machinery pays off.

    Source weights are global per annotator and estimated over the whole
    corpus, as in the paper.
    """
    C = _build(annotator_data_list, K_rule, scale_mode='none')
    if C.n_ent == 0:
        return {}
    J = C.J

    segs = sorted(C.slots)
    lo = {t: float(C.seg_starts[t]) for t in segs}
    span = {t: max(float(C.seg_ends[t] - C.seg_starts[t]), 1e-6) for t in segs}

    w = np.ones(J)
    prev = np.array(C.truth, copy=True)

    for itr in range(max_itr):
        # ---- truth pass, forward along the boundary index ----------
        for t in segs:
            Kt = C.K[t]
            prev_star = None
            for k in range(Kt):
                e = C.ent_of[(t, k)]
                s, v = C.src[e], C.val[e]
                if not s.size:
                    continue
                vn = (v - lo[t]) / span[t]
                ws = w[s]
                num = float(np.dot(ws, vn))
                den = float(ws.sum())
                if prev_star is not None:
                    num += lam * prev_star
                    den += lam
                if den > EPS:
                    x = num / den
                    prev_star = x
                    C.truth[e] = x * span[t] + lo[t]

        # ---- source weights, with the gamma recency decay ----------
        cnt = np.zeros(J)
        sse = np.zeros(J)
        for t in segs:
            Kt = C.K[t]
            for k in range(Kt):
                e = C.ent_of[(t, k)]
                s, v = C.src[e], C.val[e]
                if not s.size:
                    continue
                # decay is over the timeline, i.e. distance from the last
                # boundary of this segment
                g = gamma ** (Kt - 1 - k)
                d = ((v - C.truth[e]) / span[t]) ** 2
                np.add.at(cnt, s, g)
                np.add.at(sse, s, g * d)
        w = _evolvt_weights(alpha, beta, theta, cnt, sse)
        w = w / max(w.max(), EPS)

        denom = np.linalg.norm(prev)
        err = np.linalg.norm(C.truth - prev) / denom if denom > 0 else 99.0
        prev = np.array(C.truth, copy=True)
        if err < tol and itr > 0:
            break
        if realign_every and (itr + 1) % realign_every == 0:
            C.push_truth()
            C.realign()

    C.push_truth()
    return C.emit(merge_within=merge_within)


def baseline_dynatd_decay(annotator_data_list,
                          K_rule='median',
                          alpha=2.0,
                          beta=1.0,
                          theta=1.0,
                          gamma=0.9,
                          n_passes=2,
                          merge_within=0.0,
                          **kwargs):
    """
    B18: EvolvT's ONLINE form (EvolvT(T*)) run along the CORPUS axis, which is
    the mapping that actually tests our contribution.

    Here a timestamp is a segment, in annotation order, and the objects at that
    timestamp are that segment's boundary slots.  Objects do not persist from
    one segment to the next, so the lambda term has nothing to link and is
    dropped (lambda = 0); what survives, and what matters, is the **gamma
    recency decay on the source weights**:

        w_j(T) = [ 2 alpha - 2 + sum_{t<=T} gamma^{T-t} c_{j,t} ]
                 / [ 2 beta + theta sum_{t<=T} gamma^{T-t} SSE_{j,t} ]

    so the weight annotator j carries at segment T is dominated by how well it
    agreed on *recent* segments.  It is the one comparator that, like
    CTD-RR, lets source reliability vary along the corpus: a causal
    exponential filter rather than a two-sided Gaussian window.

    Algorithm 1 of the paper: at each timestamp, aggregate with the current
    weights, then fold that timestamp's counts and errors into the decayed
    accumulators.  `n_passes` > 1 repeats the sweep with the converged weights
    as the starting point, which the batch setting allows and the streaming
    setting does not.

    gamma = 1 removes the decay and reduces this to CRH-style global weights
    computed in one pass, which is the natural ablation to read it against.
    """
    C = _build(annotator_data_list, K_rule, scale_mode='none')
    if C.n_ent == 0:
        return {}
    J = C.J

    segs = sorted(C.slots)
    lo = {t: float(C.seg_starts[t]) for t in segs}
    span = {t: max(float(C.seg_ends[t] - C.seg_starts[t]), 1e-6) for t in segs}

    cnt = np.zeros(J)
    sse = np.zeros(J)

    for p in range(max(1, n_passes)):
        for t in segs:
            w = _evolvt_weights(alpha, beta, theta, cnt, sse)
            # aggregate this segment with the weights as they stand
            err_t = np.zeros(J)
            cnt_t = np.zeros(J)
            for k in range(C.K[t]):
                e = C.ent_of[(t, k)]
                s, v = C.src[e], C.val[e]
                if not s.size:
                    continue
                ws = w[s]
                tot = float(ws.sum())
                if tot > EPS:
                    C.truth[e] = float(np.dot(ws, v) / tot)
                d = ((v - C.truth[e]) / span[t]) ** 2
                np.add.at(err_t, s, d)
                np.add.at(cnt_t, s, 1.0)
            # discount everything seen so far, then fold in this timestamp
            cnt = gamma * cnt + cnt_t
            sse = gamma * sse + err_t
        C.push_truth()
        C.realign()

    C.push_truth()
    return C.emit(merge_within=merge_within)


ADVANCED_BASELINES["B17_DynaTD"] = {
    "fn": baseline_dynatd_kdd15,
    "per_video": False,
    "default": dict(K_rule='median', alpha=2.0, beta=1.0, theta=1.0,
                    lam=1.0, gamma=1.0),
    "grid": {
        "K_rule": ['median', 'majority', 'max'],
        "lam": [0.0, 0.1, 1.0, 10.0],
        "gamma": [0.7, 0.9, 1.0],
        "theta": [0.1, 1.0, 10.0],
        "beta": [0.1, 1.0],
    },
}

ADVANCED_BASELINES["B18_DynaTD_decay"] = {
    "fn": baseline_dynatd_decay,
    "per_video": False,
    "default": dict(K_rule='median', alpha=2.0, beta=1.0, theta=1.0,
                    gamma=0.9, n_passes=2),
    "grid": {
        "K_rule": ['median', 'majority', 'max'],
        "gamma": [0.5, 0.8, 0.9, 0.95, 0.99, 1.0],
        "theta": [0.1, 1.0, 10.0],
        "beta": [0.1, 1.0],
        "n_passes": [1, 2],
    },
}
