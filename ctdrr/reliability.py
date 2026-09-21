"""
Reliability estimation for CTD-RR.

The M-step needs a precision lambda_{j,t} for every (annotator, segment) pair,
estimated from the few residuals that pair contributes.  `estimator` selects
where the extra strength for that estimate is borrowed from:

    'window'        a Gaussian window over segment index, for fixed annotators
                    labelling the corpus in order
    'shrink_log'    shrinkage toward the corpus centre in log-variance space,
                    for anonymous panels with neither identity nor order
    'shrink'        the same shrinkage in linear variance space
    'shrink_mix'    shrinkage toward a two-component mixture centre
    'shrink_f1'     shrinkage toward a centre regressed on the corpus's own
                    annotator consistency score
    'video_shared'  one precision per segment, shared by its annotators
    'nowin'         per-pair maximum likelihood, no regularization
    'global'        one precision per annotator slot, pooled over the corpus
    'uniform'       no reliability weighting
    'eb', 'eb_video', 'eb_window'
                    empirical-Bayes Gamma prior variants

All of them have the same form,

    lambda = (pseudo-count + own count) / (pseudo-error + own error),

and differ only in where the pseudo-counts come from.  `ctd_run` is the entry
point; it takes the estimator name and its hyperparameters.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, digamma, polygamma

from .em import (
    extract_annotations, initialize, e_step, prune_and_merge_degenerate,
)
from .baselines import _resolve_K

EPS = 1e-12
FALLBACK_PRECISION = 1.0 / 9.0     # the base method's sigma ~ 3 s prior


# =====================================================================
# Sufficient statistics
# =====================================================================

def raw_stats(timestamps, X_star, responsibilities, K, active_videos, J, N):
    """
    N_eff[j,t] = total responsibility mass annotator j puts on segment t,
    S[j,t]     = the matching responsibility-weighted squared error.

    Identical to the accumulation at the top of the base method's `m_step`;
    factored out so every estimator below sees exactly the same input and the
    comparison is about the estimator alone.
    """
    N_eff = np.zeros((J, N))
    S = np.zeros((J, N))
    for t in active_videos:
        K_t = K[t]
        for j in range(J):
            ts = timestamps[(j, t)]
            if not ts:
                continue
            r = responsibilities[(j, t)]
            arr = np.asarray(ts, dtype=float)
            xs = np.array([X_star[(t, k)] for k in range(K_t)], dtype=float)
            diff2 = (arr[:, None] - xs[None, :]) ** 2
            N_eff[j, t] = float(r.sum())
            S[j, t] = float((r * diff2).sum())
    return N_eff, S


# =====================================================================
# Empirical-Bayes fit of the Gamma prior
# =====================================================================

def _solve_trigamma(y, lo=1e-3, hi=1e6):
    """Invert trigamma(a) = y.  trigamma is strictly decreasing, so bisect."""
    if not np.isfinite(y) or y <= 0:
        return hi
    if polygamma(1, lo) < y:
        return lo
    if polygamma(1, hi) > y:
        return hi
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if polygamma(1, mid) > y:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))



def _robust_logvar(x, lo_q=1.0, hi_q=99.0):
    """
    Variance of log lambda-hat, estimated from the interquartile range.

    lambda-hat = n/S diverges whenever an annotator's marks land exactly on the
    current truth (S -> 0).  On MedVid, whose timestamps are integer seconds,
    that is common rather than exceptional - two annotators naming the same
    second put the truth exactly on both of them - and it makes a plain
    `np.var` useless: the raw log-variance is 851 against a chi-square
    prediction of 1.8, and winsorising at the 1st/99th percentiles still leaves
    132.  The interquartile scale ignores the whole tail instead of trying to
    bound it, so the degenerate pairs cannot set the scale however many of them
    there are, and it is what the excess-dispersion test below needs: a
    measure of the BULK spread, comparable across corpora.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float('nan')
    q25, q75 = np.percentile(x, [25.0, 75.0])
    return float(((q75 - q25) / 1.349) ** 2)


def fit_gamma_prior_moment(n_eff, S, min_n=1e-3, a_max=1e4):
    """
    Fit (a, b) by moments, targeting the part of the spread in lambda-hat that
    sampling noise cannot explain.

    This is the estimator actually used, because the marginal-likelihood fit
    below is degenerate on this data - see `fit_gamma_prior` for why.  Two
    moments determine (a, b):

      SPREAD.  If lambda ~ Gamma(a,b) and S|lambda ~ Gamma(n/2, lambda/2), then
      lambda-hat = n/S satisfies
          Var(log lambda-hat) = trigamma(n/2) + trigamma(a),
      the first term being pure sampling noise.  So the EXCESS of the observed
      log-variance over mean(trigamma(n/2)) identifies trigamma(a) directly,
      and a = trigamma^{-1}(excess).  A source population with no real
      heterogeneity gives excess <= 0 and a -> a_max, i.e. full shrinkage; the
      more the reliabilities genuinely differ, the smaller a and the weaker the
      pull towards the population.

      LOCATION.  E[S]/E[n] estimates E[1/lambda] = b/(a-1), so
          b = (a - 1) * sum(n) / sum(S).
      The pooled ratio is used rather than the mean of the per-pair ratios,
      which is badly biased upward at n ~ 4 (E[n/S] = n lambda /(n-2)).

    Both moments are computed on the same statistics the estimate will use, so
    the prior is refitted at every M-step as the truths move.
    """
    n = np.asarray(n_eff, dtype=float).ravel()
    s = np.asarray(S, dtype=float).ravel()
    m = (n > min_n) & np.isfinite(s) & (s > 0)
    if m.sum() < 10:
        return 2.0, 1.0
    n, s = n[m], s[m]

    lam = n / s
    obs = _robust_logvar(np.log(lam))
    pred = float(np.mean(polygamma(1, n / 2.0)))
    excess = obs - pred
    a = float(min(_solve_trigamma(excess), a_max)) if excess > 0 else a_max

    pooled = float(n.sum() / max(s.sum(), EPS))     # estimates (a-1)/b
    b = max((a - 1.0), EPS) / max(pooled, EPS)
    return a, b


def fit_within_video_shape(n_eff, S, min_n=1e-3, a_max=1e4):
    """
    Fit only the WITHIN-video heterogeneity shape `a`.

    The pooled fit above cannot tell "this video is hard, everyone was
    imprecise on it" apart from "this annotator is imprecise", and on GEBD the
    first component is much the larger of the two.  Shrinking towards a corpus
    mean therefore spends most of its pull flattening exactly the contrast that
    the weighted average needs - which annotator of THIS video to trust - and
    costs accuracy.

    So the log-variance decomposition is applied after centring each video:

        Var_within(log lambda-hat) = mean(trigamma(n/2)) + trigamma(a)

    with the video mean removed from both sides, and `a` is again
    trigamma^{-1} of the excess.  The location parameter is not fitted at all;
    each video supplies its own, so the prior becomes Gamma(a, a/mu_t) with
    mu_t the video's pooled precision.
    """
    n = np.asarray(n_eff, dtype=float)
    s = np.asarray(S, dtype=float)
    mask = (n > min_n) & np.isfinite(s) & (s > 0)
    if mask.sum() < 10:
        return 2.0

    lam = np.where(mask, n / np.maximum(s, EPS), np.nan)
    log_lam = np.log(lam)

    resid, samp = [], []
    J, N = n.shape
    for t in range(N):
        col = mask[:, t]
        if col.sum() < 2:            # a single annotator has no within-video
            continue                 # contrast to measure
        v = log_lam[col, t]
        resid.append(v - v.mean())
        # Centring J values costs one degree of freedom.
        samp.append(polygamma(1, n[col, t] / 2.0) * (col.sum() - 1)
                    / col.sum())
    if not resid:
        return 2.0
    resid = np.concatenate(resid)
    samp = np.concatenate(samp)
    # The residuals of a mean over m values have variance (m-1)/m of the
    # original, which the sampling term above already accounts for.
    excess = _robust_logvar(resid) - float(np.mean(samp))
    return float(min(_solve_trigamma(excess), a_max)) if excess > 0 else a_max


def video_level_precision(n_eff, S, a, prior_scale=1.0, min_mu=1e-4):
    """
    lambda_{j,t} = (tau a + n_{j,t}/2) / (tau a / mu_t + S_{j,t}/2),
    mu_t = sum_j n_{j,t} / sum_j S_{j,t}  (the video's own pooled precision).

    An annotator with plenty of marks keeps its own estimate; one with few is
    pulled towards how precise this video's annotators were in general, not
    towards how precise the corpus was.  Videos where nobody agreed are still
    down-weighted as a whole, because mu_t itself is low there.
    """
    J, N = n_eff.shape
    num = n_eff.sum(axis=0)
    den = S.sum(axis=0)
    mu = np.where(den > EPS, num / np.maximum(den, EPS), np.nan)
    if np.all(np.isnan(mu)):
        mu = np.full(N, 1.0)
    fallback = float(np.nanmedian(mu)) if np.any(~np.isnan(mu)) else 1.0
    mu = np.where(np.isnan(mu) | (mu <= min_mu), fallback, mu)

    a_s = a * prior_scale
    return (a_s + n_eff / 2.0) / np.maximum(a_s / mu[None, :] + S / 2.0, EPS)


def fit_gamma_prior(n_eff, S, a0=2.0, b0=1.0, min_n=1e-3):
    """
    Maximise the marginal likelihood of the residual statistics over (a, b).

    With lambda ~ Gamma(a,b) and S|lambda ~ Gamma(n/2, lambda/2), integrating
    lambda out gives, up to terms free of (a,b),

        log p(S | n, a, b) = a log b - lgamma(a) + lgamma(a + n/2)
                             - (a + n/2) log(b + S/2)

    Optimised over (log a, log b) so both stay positive.  Pairs with no
    responsibility mass carry no information and are dropped.

    Where the per-pair evidence is thin the likelihood is close to flat along
    a -> infinity at fixed a/b, which drives every precision to the population
    mean; the moment fit above is the better-determined alternative.
    """
    n = np.asarray(n_eff, dtype=float).ravel()
    s = np.asarray(S, dtype=float).ravel()
    m = (n > min_n) & np.isfinite(s) & (s >= 0)
    if m.sum() < 10:
        return float(a0), float(b0)
    n, s = n[m], s[m]
    half_n = n / 2.0
    half_s = s / 2.0

    def neg(p):
        a, b = np.exp(p[0]), np.exp(p[1])
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
            return 1e12
        val = (a * np.log(b) - gammaln(a) + gammaln(a + half_n)
               - (a + half_n) * np.log(b + half_s))
        return -float(val.sum())

    def grad(p):
        a, b = np.exp(p[0]), np.exp(p[1])
        da = (np.log(b) - digamma(a) + digamma(a + half_n)
              - np.log(b + half_s)).sum()
        db = (a / b - ((a + half_n) / (b + half_s))).sum()
        return np.array([-da * a, -db * b])

    # Method-of-moments start, which keeps the optimiser out of flat regions.
    lam_hat = n / np.maximum(s, EPS)
    lam_hat = lam_hat[np.isfinite(lam_hat) & (lam_hat > 0)]
    if lam_hat.size >= 2:
        mu = float(np.mean(lam_hat))
        var = float(np.var(lam_hat))
        if var > EPS and mu > EPS:
            a0 = float(np.clip(mu * mu / var, 1e-2, 1e4))
            b0 = float(np.clip(mu / var, 1e-4, 1e4))

    try:
        res = minimize(neg, np.log([a0, b0]), jac=grad, method='L-BFGS-B',
                       options={'maxiter': 100})
        a, b = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            return a, b
    except Exception:
        pass
    return float(a0), float(b0)


# =====================================================================
# Precision estimators
# =====================================================================

def _apply_window(N_eff, S, K, window_radius, J, N):
    sigma_w = max(window_radius / 2.0, EPS)
    idx = np.arange(N)
    live = np.array([1.0 if K[t] > 0 else 0.0 for t in range(N)])
    Nb = np.zeros((J, N))
    Sb = np.zeros((J, N))
    for t in range(N):
        lo = max(0, t - window_radius)
        hi = min(N - 1, t + window_radius)
        w = np.exp(-((idx[lo:hi + 1] - t) ** 2) / (2.0 * sigma_w ** 2))
        w = w * live[lo:hi + 1]
        ws = w.sum()
        if ws <= 0:
            continue
        Nb[:, t] = (N_eff[:, lo:hi + 1] * w).sum(axis=1) / ws
        Sb[:, t] = (S[:, lo:hi + 1] * w).sum(axis=1) / ws
    return Nb, Sb



# =====================================================================
# Hierarchical shrinkage for the anonymous-panel regime
# =====================================================================

def corpus_mean_variance(N_eff, S, min_n=1e-10):
    """
    sigma2_bar: the corpus-level mean variance, pooled over every
    (annotator-slot, video) pair that actually carries responsibility mass.

    Pairs with no mass are EXCLUDED rather than counted as zero — an absent
    annotator "contributes nothing to any sum", so it must not drag the corpus
    mean toward 0.  Pooling (sum of squared error over sum of mass) rather than
    averaging per-pair ML variances is deliberate: at ~4 marks per annotator
    per clip the per-pair variance is extremely noisy and its unweighted mean
    is dominated by the pairs with the least evidence.
    """
    m = N_eff > min_n
    if not np.any(m):
        return 1.0 / FALLBACK_PRECISION
    num = float(S[m].sum())
    den = float(N_eff[m].sum())
    if den <= min_n or num <= 0:
        return 1.0 / FALLBACK_PRECISION
    return num / den


def shrunk_precision(N_eff, S, m_pseudo, sigma2_bar, precision_cap=None):
    """
    Shrinkage in linear variance space, per (segment, annotator):

        sigma2_jv = ( sum_k r_jk (t_jk - mu_k)^2  +  m sigma2_bar )
                    / ( sum_k r_jk  +  m )

    and lambda_jv = 1 / sigma2_jv.

    This borrows strength from the CORPUS rather than from neighbouring
    segments, which is what makes it valid on GEBD/GEB+: it never keys on the
    slot index across videos, so it is indifferent to the fact that slot j is a
    different person in every clip.  `m` is a pseudo-count in the same units as
    the responsibility mass, so m = 5 means "shrink as if five extra marks of
    average difficulty had been seen".

    m = 0 reduces to the plain per-video ML precision n/S exactly, including
    the degenerate branches, so the ablation is a true nesting rather than an
    approximation.  `test_shrink_m0_equals_nowin` asserts this.

    Slots with no responsibility mass get no estimate and fall back to the
    prior constant, as they would under any estimator: they have no marks, so
    their precision is never read by the E-step.
    """
    n = np.asarray(N_eff, dtype=float)
    sse = np.asarray(S, dtype=float)
    has = n > 1e-10

    num = sse + m_pseudo * sigma2_bar
    den = n + m_pseudo
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma2 = np.where(den > 1e-10, num / np.maximum(den, EPS), np.nan)
        prec = np.where((sigma2 > 1e-10) & np.isfinite(sigma2),
                        1.0 / np.maximum(sigma2, EPS), np.nan)

    # zero residual with real mass -> infinite precision, take the cap
    cap = precision_cap if (precision_cap is not None
                            and np.isfinite(precision_cap)) else np.inf
    prec = np.where(has & (sigma2 <= 1e-10), cap, prec)
    prec = np.where(np.isnan(prec) | ~has, FALLBACK_PRECISION, prec)
    return np.minimum(prec, cap)



# ---------------------------------------------------------------------
# Shrinkage variants motivated by the sigma-hat^2 histogram
# ---------------------------------------------------------------------
# The per-(video, slot) sigma-hat^2 distribution on GEBD/GEB+ is (i) extremely
# heavy-tailed in linear space - GEB+ skew +11.7, excess kurtosis +217, pooled
# mean 13x the median - and (ii) BIMODAL in log space: a 2-component GMM beats
# 1 component by dBIC = -7363 (GEBD) / -1198 (GEB+), with modes near
# sigma ~ 0.085 s (marks pinned to the same frame) and sigma ~ 0.65 s (same
# event, different instant).  So the original shrinkage was pulling every pair
# toward a centre that (i) sits in the tail rather than the bulk and (ii) lies
# in the trough BETWEEN the two modes.  The variants below fix each half.

LOG_FLOOR_S2 = 1e-4      # 0.01 s: below one video frame; where S = 0 lands


def prior_centre(N_eff, S, how='pooled', min_n=1e-10):
    """
    sigma2_bar under four definitions.  The pooled mean sits in the tail of the
    heavy-tailed variance distribution; the other three sit in its bulk.
    """
    m = (N_eff > min_n) & (S > 1e-12)
    if not np.any(m):
        return 1.0 / FALLBACK_PRECISION
    s2 = S[m] / N_eff[m]
    if how == 'pooled':
        return corpus_mean_variance(N_eff, S)
    if how == 'median':
        return float(np.median(s2))
    if how == 'geomean':
        return float(np.exp(np.mean(np.log(np.maximum(s2, LOG_FLOOR_S2)))))
    if how == 'arith':
        return float(np.mean(s2))
    raise ValueError(how)


def shrunk_precision_log(N_eff, S, m_pseudo, log_centre, precision_cap=None):
    """
    Shrink in LOG variance space:

        log sigma2_jv = ( n_jv * log sigma-hat2_jv + m * log_centre_jv )
                        / ( n_jv + m )

    i.e. a geometric interpolation sigma2 = shat2^(n/(n+m)) * centre^(m/(n+m)).
    In log space the heavy tail is a few units wide instead of two orders of
    magnitude, so the same pseudo-count m means the same thing for a pair at
    0.005 s^2 and a pair at 5 s^2, and the pull is toward where the mass is.

    `log_centre` may be a scalar (one centre for the corpus) or a (J, N) array
    (a centre per pair, used by the two-component prior).  A pair with zero
    residual is floored at LOG_FLOOR_S2 before the log rather than sent to
    infinite precision.
    """
    n = np.asarray(N_eff, dtype=float)
    sse = np.asarray(S, dtype=float)
    has = n > 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        shat2 = np.where(has, sse / np.maximum(n, EPS), np.nan)
    shat2 = np.maximum(shat2, LOG_FLOOR_S2)
    lc = np.broadcast_to(np.asarray(log_centre, dtype=float), n.shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        logs2 = (n * np.log(shat2) + m_pseudo * lc) / np.maximum(n + m_pseudo,
                                                                  EPS)
        prec = np.exp(-logs2)
    cap = precision_cap if (precision_cap is not None
                            and np.isfinite(precision_cap)) else np.inf
    prec = np.where(has & np.isfinite(prec), prec, FALLBACK_PRECISION)
    return np.minimum(prec, cap)


def mixture_log_centres(N_eff, S, n_comp=2, min_n=1e-10, seed=0):
    """
    Two-component prior: fit a GMM to log sigma-hat^2 over all pairs with mass,
    then give EACH pair its responsibility-weighted component mean as its own
    centre.  A pair that looks precise is pulled toward the precise mode and a
    loose pair toward the loose mode; the trough between them, where the single
    pooled centre used to sit, is no longer anybody's target.

    Returns (J, N) array of log centres (corpus geometric mean where a pair has
    no mass) plus the fitted (weights, means, sds) for reporting.
    """
    from sklearn.mixture import GaussianMixture
    m = (N_eff > min_n) & (S > 1e-12)
    n = np.asarray(N_eff, dtype=float)
    out = np.full(n.shape, np.log(prior_centre(N_eff, S, 'geomean')))
    if m.sum() < 50:
        return out, None
    ls = np.log(np.maximum(S[m] / n[m], LOG_FLOOR_S2)).reshape(-1, 1)
    g = GaussianMixture(n_comp, random_state=seed, n_init=2).fit(ls)
    resp = g.predict_proba(ls)                       # (P, C)
    mu = g.means_.ravel()
    out[m] = resp @ mu
    return out, (g.weights_.tolist(), mu.tolist(),
                 np.sqrt(g.covariances_.ravel()).tolist())


def f1consis_log_centres(N_eff, S, f1, min_n=1e-10):
    """
    f1_consis as the prior centre: regress log sigma-hat^2 on each pair's
    consistency score across the corpus, and use the fitted value as that
    pair's centre.  The score is derived from inter-annotator agreement on the
    marks being aggregated, so this variant is reported as an ablation rather
    than used by the method.
    """
    m = (N_eff > min_n) & (S > 1e-12) & np.isfinite(f1)
    n = np.asarray(N_eff, dtype=float)
    base = np.log(prior_centre(N_eff, S, 'geomean'))
    out = np.full(n.shape, base)
    if m.sum() < 50:
        return out, None
    y = np.log(np.maximum(S[m] / n[m], LOG_FLOOR_S2))
    x = f1[m]
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = coef[0] + coef[1] * f1
    out = np.where(np.isfinite(f1), fit, base)
    r = float(np.corrcoef(x, y)[0, 1])
    return out, (float(coef[0]), float(coef[1]), r)


def video_shared_precision(N_eff, S, precision_cap=None):
    """
    Ablation: ONE precision per video, shared by every slot in it.

    sigma2_v = sum_j SSE_jv / sum_j n_jv, applied to all slots of video v.
    This keeps the "some videos are harder than others" part of the model and
    removes the "some annotators are better than others" part, so it isolates
    what per-annotator reliability is worth on top of per-video difficulty.
    """
    n = np.asarray(N_eff, dtype=float)
    sse = np.asarray(S, dtype=float)
    tot_n = n.sum(axis=0)
    tot_s = sse.sum(axis=0)
    cap = precision_cap if (precision_cap is not None
                            and np.isfinite(precision_cap)) else np.inf
    with np.errstate(divide='ignore', invalid='ignore'):
        sig_v = np.where(tot_n > 1e-10, tot_s / np.maximum(tot_n, EPS), np.nan)
        pv = np.where((sig_v > 1e-10) & np.isfinite(sig_v),
                      1.0 / np.maximum(sig_v, EPS), np.nan)
    pv = np.where((tot_n > 1e-10) & (sig_v <= 1e-10), cap, pv)
    pv = np.where(np.isnan(pv), FALLBACK_PRECISION, pv)
    pv = np.minimum(pv, cap)
    prec = np.repeat(pv[None, :], n.shape[0], axis=0)
    return np.where(n > 1e-10, prec, FALLBACK_PRECISION)


def f1_consis_init(annotator_data_list, J, N, base_precision=1.0,
                   power=1.0, lo=0.25, hi=4.0):
    """
    OPTIONAL initialisation only: seed lambda_jv from the dataset's own
    per-annotator consistency score, scaled so the corpus mean is unchanged.

        lambda_jv = base * clip( (f1_jv / mean f1) ** power , lo, hi )

    This is a starting point for EM, never supervision: `f1_consis` is derived
    from inter-annotator agreement on the same marks the method is aggregating,
    so using it in the objective would leak the answer.  Every M-step
    afterwards overwrites it from the data.  Returns None when the corpus
    carries no consistency scores.
    """
    vals = np.full((J, N), np.nan)
    for j in range(J):
        col = annotator_data_list[j]
        for t in range(min(N, len(col))):
            f = col[t].get('f1_consis')
            if f is not None and np.isfinite(f):
                vals[j, t] = float(f)
    if not np.any(np.isfinite(vals)):
        return None
    mean_f = float(np.nanmean(vals))
    if not np.isfinite(mean_f) or mean_f <= 0:
        return None
    ratio = np.where(np.isfinite(vals), vals / mean_f, 1.0)
    return base_precision * np.clip(ratio ** power, lo, hi)


def estimate_precision(N_eff, S, K, active, J, N, estimator,
                       window_radius=15, precision_cap=8.0,
                       prior=None, prior_scale=1.0, eb_fit='moment',
                       m_shrink=5.0, sigma2_bar=None,
                       shrink_centre='pooled', f1_matrix=None):
    """
    Returns (precision[J,N], fitted_prior or None).

    `prior_scale` (tau) multiplies the fitted pseudo-counts (a, b) together, so
    it scales how hard the prior pulls WITHOUT moving the population mean it
    pulls towards.  tau -> 0 recovers the unshrunk per-segment estimate,
    tau -> infinity a single constant precision, and tau = 1 is the empirical-
    Bayes fit.  It is the one free knob of the EB estimator, and plays the same
    role for it that `window_radius` plays for the published method: both say
    how much strength to borrow.
    """
    prec = np.full((J, N), FALLBACK_PRECISION)

    if estimator == 'uniform':
        return prec, None

    if estimator == 'shrink':
        sb = prior_centre(N_eff, S, shrink_centre) if sigma2_bar is None \
            else float(sigma2_bar)
        # returned in the same (numeric, numeric-or-None) shape every other
        # estimator uses, so downstream reporting can float() it blindly
        return shrunk_precision(N_eff, S, m_shrink, sb,
                                precision_cap=precision_cap), (sb, None)

    if estimator == 'shrink_log':
        sb = prior_centre(N_eff, S, shrink_centre) if sigma2_bar is None \
            else float(sigma2_bar)
        return shrunk_precision_log(N_eff, S, m_shrink, np.log(sb),
                                    precision_cap=precision_cap), (sb, None)

    if estimator == 'shrink_mix':
        lc, fit = mixture_log_centres(N_eff, S, n_comp=2)
        return shrunk_precision_log(N_eff, S, m_shrink, lc,
                                    precision_cap=precision_cap), \
            (float(np.exp(np.median(lc))), None)

    if estimator == 'shrink_f1':
        f1 = f1_matrix if f1_matrix is not None else np.full(N_eff.shape,
                                                               np.nan)
        lc, fit = f1consis_log_centres(N_eff, S, f1)
        return shrunk_precision_log(N_eff, S, m_shrink, lc,
                                    precision_cap=precision_cap), \
            (float(np.exp(np.median(lc))), None)

    if estimator == 'video_shared':
        return video_shared_precision(N_eff, S,
                                      precision_cap=precision_cap), None

    if estimator in ('window', 'eb_window'):
        Nb, Sb = _apply_window(N_eff, S, K, window_radius, J, N)
    elif estimator == 'global':
        tot_n = N_eff.sum(axis=1)
        tot_s = S.sum(axis=1)
        Nb = np.repeat(tot_n[:, None], N, axis=1)
        Sb = np.repeat(tot_s[:, None], N, axis=1)
    else:                                   # 'nowin', 'eb'
        Nb, Sb = N_eff, S

    if estimator == 'eb_video':
        a = fit_within_video_shape(N_eff, S) if prior is None else prior[0]
        prec = video_level_precision(N_eff, S, a, prior_scale=prior_scale)
        if precision_cap is not None and np.isfinite(precision_cap):
            prec = np.minimum(prec, precision_cap)
        return prec, (a, None)

    if estimator in ('eb', 'eb_window'):
        # Fit on the same statistics the estimate will use.
        if prior is not None:
            a, b = prior
        elif eb_fit == 'mle':
            a, b = fit_gamma_prior(Nb, Sb)
        else:
            a, b = fit_gamma_prior_moment(Nb, Sb)
        a_s, b_s = a * prior_scale, b * prior_scale
        prec = (a_s + Nb / 2.0) / np.maximum(b_s + Sb / 2.0, EPS)
        # A cap is not needed under the prior, but honour it if asked.
        if precision_cap is not None and np.isfinite(precision_cap):
            prec = np.minimum(prec, precision_cap)
        return prec, (a, b)

    with np.errstate(divide='ignore', invalid='ignore'):
        val = np.where((Nb > 1e-10) & (Sb > 1e-10), Nb / np.maximum(Sb, EPS),
                       np.nan)
    val = np.where((Nb > 1e-10) & (Sb <= 1e-10), precision_cap, val)
    prec = np.where(np.isnan(val), FALLBACK_PRECISION,
                    np.minimum(val, precision_cap))
    return prec, None


# =====================================================================
# The EM driver
# =====================================================================

def initialize_k(timestamps, N, J, k_rule='max'):
    """
    The base method's `initialize`, with the step-count rule made selectable.

    K is initialised to the MAX of the non-zero annotator counts: over-
    estimation is recoverable by pruning, while under-estimation loses
    boundaries permanently.  Which rule is best depends on the matching
    tolerance relative to the annotators' mark density, so it is selectable.

    Every comparator tunes its own step-count rule -- CRH, GTM, CATD, EvolvT
    and DynaTD expose `K_rule`, while HDBSCAN, KDE+peaks and KDEm choose K
    implicitly through a density threshold -- so the rule is exposed here too
    and swept on the same tuning half.  `k_rule='max'` is the published
    setting.
    """
    K, active_videos, X_star = [], [], {}
    for t in range(N):
        K_t = _resolve_K(timestamps, t, J, k_rule)
        K.append(K_t)
        if K_t == 0:
            continue
        all_ts = []
        for j in range(J):
            all_ts.extend(timestamps[(j, t)])
        all_ts.sort()
        if not all_ts:
            K[t] = 0
            continue
        active_videos.append(t)
        for k in range(K_t):
            q = (k + 1) / (K_t + 1)
            X_star[(t, k)] = all_ts[min(int(q * len(all_ts)), len(all_ts) - 1)]
    precision = np.full((J, N), FALLBACK_PRECISION)
    return K, active_videos, X_star, precision


DEFAULT_CFG = dict(
    estimator='eb',
    k_rule='max',
    window_radius=15,
    max_iters=10,
    convergence_threshold=2.0,
    pruning_threshold=0.3,
    precision_cap=8.0,
    initial_precision=1.0,
    degeneracy_threshold=0.5,
    prior_scale=1.0,
    eb_fit='moment',
    m_shrink=5.0,
    sigma2_bar_mode='iter',
    init_from_f1consis=False,
    shrink_centre='pooled',
)


def ctd_run(annotator_data_list, return_precision=False, **kwargs):
    """
    CTD-RR with a selectable precision estimator.

    `estimator='window'` reproduces the published method exactly (it calls the
    same E-step and the same prune/merge, and `_apply_window` is the base
    method's smoothing written in array form), so the variants below are
    strictly a change of one component.
    """
    cfg = dict(DEFAULT_CFG)
    cfg.update({k: v for k, v in kwargs.items() if k in DEFAULT_CFG})

    timestamps, seg_starts, seg_ends, N, J = extract_annotations(
        annotator_data_list)
    K, active, X_star, precision = initialize_k(timestamps, N, J,
                                                k_rule=cfg['k_rule'])
    precision[:, :] = cfg['initial_precision']
    if cfg['init_from_f1consis']:
        seed = f1_consis_init(annotator_data_list, J, N,
                              base_precision=cfg['initial_precision'])
        if seed is not None:
            precision = seed

    resp, fitted = None, None
    frozen_sb = None
    f1_mat = None
    if cfg['estimator'] == 'shrink_f1':
        f1_mat = np.full((J, N), np.nan)
        for j in range(J):
            col = annotator_data_list[j]
            for t in range(min(N, len(col))):
                f = col[t].get('f1_consis')
                if f is not None and np.isfinite(f):
                    f1_mat[j, t] = float(f)
    for it in range(cfg['max_iters']):
        resp, shift = e_step(timestamps, X_star, precision, K, active,
                             seg_starts, seg_ends, J)
        N_eff, S = raw_stats(timestamps, X_star, resp, K, active, J, N)
        # 'first' freezes sigma2_bar at the first plain pass; 'iter' refreshes
        # it every outer iteration.  'iter' is the default.
        sb = frozen_sb if cfg['sigma2_bar_mode'] == 'first' else None
        precision, fitted = estimate_precision(
            N_eff, S, K, active, J, N, cfg['estimator'],
            window_radius=cfg['window_radius'],
            precision_cap=cfg['precision_cap'],
            prior_scale=cfg['prior_scale'],
            eb_fit=cfg['eb_fit'],
            m_shrink=cfg['m_shrink'],
            sigma2_bar=sb,
            shrink_centre=cfg['shrink_centre'],
            f1_matrix=f1_mat)
        if (frozen_sb is None and cfg['estimator'] in ('shrink', 'shrink_log')
                and isinstance(fitted, tuple) and fitted):
            frozen_sb = fitted[0]
        if it > 0 and shift < cfg['convergence_threshold']:
            break

    truths, _, _, _ = prune_and_merge_degenerate(
        resp, X_star, K, active, J,
        pruning_threshold=cfg['pruning_threshold'],
        degeneracy_threshold=cfg['degeneracy_threshold'])

    if return_precision:
        return truths, precision, fitted
    return truths


def make_variant(estimator, **fixed):
    """A `method(annotator_data_list, **kw) -> truths` closure for the harness."""
    def fn(annotator_data_list, **kw):
        cfg = dict(fixed)
        cfg.update(kw)
        cfg['estimator'] = estimator
        return ctd_run(annotator_data_list, **cfg)
    fn.__name__ = f"ctd_{estimator}"
    return fn
