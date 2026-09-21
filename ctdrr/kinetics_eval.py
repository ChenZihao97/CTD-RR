"""
gebd_eval.py
============
Evaluation protocols for Kinetics-GEBD, where there is no ground truth.

Four protocols, in decreasing order of how much they can be trusted:

  1. EXTERNAL REFERENCE (GEB+).  For the 6,394 Kinetics-GEBD videos that also
     appear in Kinetics-GEB+, the GEB+ boundary set comes from a separate
     annotation round under a different task definition, and no method under
     test has seen it.  This is the only protocol here whose reference is not
     built out of the very annotations being aggregated, so it is the primary
     number.

  2. LEAVE-ONE-ANNOTATOR-OUT.  Hold out annotator slot h, aggregate the rest,
     score the result against the held-out annotator.  Needs no ground truth
     and no external data, and every method is measured identically.  What it
     measures is predictive: can the aggregate of J-1 annotators stand in for
     an unseen one.  Its blind spot is that it rewards reproducing annotator
     idiosyncrasy as well as signal, so a method that correctly discards noise
     the held-out annotator shares is charged for it.

  3. OFFICIAL GEBD F1.  The dataset's own protocol: score against every rater
     and keep the best-scoring rater per video.  Reported because it is the
     number the GEBD literature uses, but it is IN-SAMPLE - the methods were
     given all raters, including the one the metric then scores them against -
     so it flatters every method here and cannot separate them fairly.

  4. RELIABILITY VALIDATION.  Kinetics-GEBD ships `f1_consis`, the consistency
     of each annotator against the others on that video.  Correlating a
     method's estimated per-annotator reliability against it tests the
     reliability model directly rather than through boundary F1.  This is an
     external check on the part of the model the F1 numbers cannot isolate.

All boundary matching uses the official Rel.Dis rule: a prediction matches a
reference boundary if they are within 0.05 x video_duration, resolved by the
greedy nearest-first assignment from the GEBD challenge evaluation script.
"""

import numpy as np

REL_DIS = 0.05


# =====================================================================
# Matching
# =====================================================================

def match_counts(pred, ref, tol):
    """
    (tp, n_ref, n_pred) under the official greedy rule: walk the reference
    boundaries in order, take each one's nearest surviving prediction, and
    consume it if it is within `tol`.

    Ported from Challenge_eval_Code/eval.py so the numbers are comparable with
    published GEBD results.  Note it is greedy in reference order, not optimal
    bipartite matching; that is the published behaviour and is kept.
    """
    pred = list(pred)
    ref = list(ref)
    if not ref or not pred:
        return 0, len(ref), len(pred)
    offset = np.abs(np.asarray(ref)[:, None] - np.asarray(pred)[None, :])
    tp = 0
    for i in range(len(ref)):
        if offset.shape[1] == 0:
            break
        j = int(np.argmin(offset[i, :]))
        if offset[i, j] <= tol:
            tp += 1
            offset = np.delete(offset, j, 1)
    return tp, len(ref), len(pred)


def _unmatched(pred, ref, tol):
    """Predictions left over after the greedy reference-order matching."""
    if not ref:
        return list(pred)
    import numpy as _np
    off = _np.abs(_np.asarray(ref)[:, None] - _np.asarray(pred)[None, :])
    alive = list(range(len(pred)))
    for i in range(len(ref)):
        if not alive:
            break
        j = min(alive, key=lambda c: off[i, c])
        if off[i, j] <= tol:
            alive.remove(j)
    return [pred[j] for j in alive]


def _prf(tp, n_ref, n_pred):
    rec = tp / n_ref if n_ref else 1.0
    prec = tp / n_pred if n_pred else 0.0
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
    return prec, rec, f1


# =====================================================================
# 1. External reference (GEB+)
# =====================================================================

def eval_external(preds, videos, ext_ref, rel_dis=REL_DIS, mask=None):
    """
    Micro- and macro-F1 against the GEB+ boundary set, over the videos present
    in both.  `preds` is {video index in `videos`: [seconds]}.
    """
    tp_all = nref_all = npred_all = 0
    per_video = []
    n_used = 0
    for i, v in enumerate(videos):
        ref = ext_ref.get(v['vid'])
        if not ref:
            continue
        n_used += 1
        tol = rel_dis * v['duration']
        pred = sorted(preds.get(i, []))
        tp, nr, npd = match_counts(pred, ref, tol)
        if mask:
            # Don't-care: a prediction that matches no reference boundary but
            # does land on an excluded candidate is neither TP nor FP.
            mk = mask.get(v['vid']) or []
            if mk:
                unmatched = _unmatched(pred, ref, tol)
                dc = sum(1 for x in unmatched
                         if min(abs(x - m) for m in mk) <= tol)
                npd -= dc
        tp_all += tp
        nref_all += nr
        npred_all += npd
        per_video.append(_prf(tp, nr, npd)[2])
    p, r, f = _prf(tp_all, nref_all, npred_all)
    return {'n_videos': n_used, 'precision': p, 'recall': r, 'f1': f,
            'macro_f1': float(np.mean(per_video)) if per_video else 0.0,
            'n_pred': npred_all, 'n_ref': nref_all}


# =====================================================================
# 2. Leave-one-annotator-out
# =====================================================================

def loao_folds(videos, J, min_input=2):
    """
    For each held-out slot h, the videos where h actually annotated and at
    least `min_input` other slots did.  Those are the only videos where the
    fold is both scorable and a genuine multi-annotator aggregation.
    """
    folds = {}
    for h in range(J):
        idxs = []
        for i, v in enumerate(videos):
            anns = v['anns']
            if h >= len(anns) or not anns[h]:
                continue
            others = sum(1 for j, a in enumerate(anns) if j != h and a)
            if others >= min_input:
                idxs.append(i)
        folds[h] = idxs
    return folds


def eval_loao(run_fn, videos, J, rel_dis=REL_DIS, min_input=2,
              return_folds=False):
    """
    `run_fn(annotator_data_list) -> {video index: [seconds]}` is called once
    per held-out slot, on the corpus with that slot removed.

    Removing slot h corpus-wide (rather than per video) is the right move here
    precisely because slots are video-local: there is no person being removed,
    only one annotation per video, so the J runs are J independent resamplings
    of "aggregate J-1 annotators".
    """
    from .gebd_data import to_annotator_data

    folds = loao_folds(videos, J, min_input)
    tp_all = nref_all = npred_all = 0
    per_fold = {}
    macro = []

    for h in range(J):
        idxs = folds[h]
        if not idxs:
            continue
        sub = []
        for v in videos:
            w = dict(v)
            w['anns'] = [a for j, a in enumerate(v['anns']) if j != h]
            sub.append(w)
        data, Jsub = to_annotator_data(sub, J=max(J - 1, 2))
        preds = run_fn(data)

        tp_f = nr_f = np_f = 0
        for i in idxs:
            ref = videos[i]['anns'][h]
            tol = rel_dis * videos[i]['duration']
            tp, nr, npd = match_counts(preds.get(i, []), ref, tol)
            tp_f += tp
            nr_f += nr
            np_f += npd
            macro.append(_prf(tp, nr, npd)[2])
        tp_all += tp_f
        nref_all += nr_f
        npred_all += np_f
        per_fold[h] = dict(zip(('precision', 'recall', 'f1'),
                               _prf(tp_f, nr_f, np_f)))
        per_fold[h]['n_videos'] = len(idxs)

    p, r, f = _prf(tp_all, nref_all, npred_all)
    out = {'precision': p, 'recall': r, 'f1': f,
           'macro_f1': float(np.mean(macro)) if macro else 0.0,
           'n_pred': npred_all, 'n_ref': nref_all,
           'n_scored': len(macro)}
    if return_folds:
        out['per_fold'] = per_fold
    return out


# =====================================================================
# 3. Official GEBD F1 (in-sample)
# =====================================================================

def eval_official(preds, videos, rel_dis=REL_DIS):
    """
    The challenge metric: per video, score against every rater and keep the
    rater giving the best F1; accumulate that rater's tp and reference count,
    but all predictions, then micro-average.
    """
    tp_all = nref_all = npred_all = 0
    macro = []
    for i, v in enumerate(videos):
        pred = preds.get(i, [])
        tol = rel_dis * v['duration']
        npred_all += len(pred)
        best = None
        for ann in v['anns']:
            if not ann:
                continue
            tp, nr, npd = match_counts(pred, ann, tol)
            f1 = _prf(tp, nr, npd)[2]
            if best is None or f1 > best[0]:
                best = (f1, tp, nr)
        if best is None:
            continue
        macro.append(best[0])
        tp_all += best[1]
        nref_all += best[2]
    p, r, f = _prf(tp_all, nref_all, npred_all)
    return {'precision': p, 'recall': r, 'f1': f,
            'macro_f1': float(np.mean(macro)) if macro else 0.0,
            'n_pred': npred_all, 'n_ref': nref_all}


# =====================================================================
# 4. Reliability validation against f1_consis
# =====================================================================

def reliability_correlation(precision, videos, J):
    """
    Spearman and Pearson correlation between a method's estimated precision
    lambda_{j,t} and the dataset's own `f1_consis[j]` for that video.

    Both are per-(annotator, video) quantities, and `f1_consis` was computed by
    the dataset authors from the annotations directly, not from any method
    under test.  A positive correlation says the reliability model recovers
    something real about who annotated well on which clip; it is the one check
    here that looks inside the model rather than at its boundary output.

    Reported three ways, because they answer different questions:
      pooled   over all (j,t) pairs - dominated by between-video differences
      within   partial-out the video mean first, so only the *ranking of
               annotators inside a video* counts.  This is the harder and more
               meaningful version: it cannot be won by tracking video
               difficulty.
      video    correlation of the per-video mean precision with f1_consis_avg
    """
    from scipy.stats import spearmanr, pearsonr

    lam, f1c, vid = [], [], []
    for i, v in enumerate(videos):
        for j, sc in enumerate(v['f1_consis']):
            if j >= J:
                continue
            if j < len(v['anns']) and v['anns'][j]:
                lam.append(float(precision[j, i]))
                f1c.append(float(sc))
                vid.append(i)
    lam = np.array(lam)
    f1c = np.array(f1c)
    vid = np.array(vid)
    if lam.size < 10:
        return {'n': int(lam.size)}

    out = {'n': int(lam.size)}
    out['spearman_pooled'] = float(spearmanr(lam, f1c).statistic)
    out['pearson_pooled'] = float(pearsonr(lam, f1c)[0])

    # Within-video: centre both variables on their per-video mean.
    lam_c = np.array(lam, copy=True)
    f1_c = np.array(f1c, copy=True)
    for u in np.unique(vid):
        m = vid == u
        if m.sum() < 2:
            lam_c[m] = 0.0
            f1_c[m] = 0.0
            continue
        lam_c[m] -= lam_c[m].mean()
        f1_c[m] -= f1_c[m].mean()
    keep = ~((lam_c == 0) & (f1_c == 0))
    if keep.sum() >= 10 and np.std(lam_c[keep]) > 0 and np.std(f1_c[keep]) > 0:
        out['spearman_within'] = float(spearmanr(lam_c[keep],
                                                 f1_c[keep]).statistic)
        out['pearson_within'] = float(pearsonr(lam_c[keep], f1_c[keep])[0])

    vm_lam, vm_f1 = [], []
    for i, v in enumerate(videos):
        sel = [j for j in range(min(J, len(v['anns']))) if v['anns'][j]]
        if sel:
            vm_lam.append(float(np.mean([precision[j, i] for j in sel])))
            vm_f1.append(float(v['f1_consis_avg']))
    if len(vm_lam) >= 10:
        out['spearman_video'] = float(spearmanr(vm_lam, vm_f1).statistic)
    return out


# =====================================================================
# Bootstrap
# =====================================================================

def boot_ci(per_item, B=10000, seed=0):
    """Percentile CI of the mean of a per-item score list."""
    a = np.asarray(per_item, dtype=float)
    if a.size < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(B, a.size))
    draws = a[idx].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_boot(per_item_a, per_item_b, B=10000, seed=0):
    """Paired bootstrap of mean(a) - mean(b): (delta, lo, hi, P(delta>0))."""
    a = np.asarray(per_item_a, dtype=float)
    b = np.asarray(per_item_b, dtype=float)
    assert a.shape == b.shape
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(B, d.size))
    draws = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), float((draws > 0).mean()))
