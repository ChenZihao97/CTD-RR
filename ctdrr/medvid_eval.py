"""
MedVid metrics, exactly as the paper defines them.

  F1@d   micro F1 over transitions; Hungarian matching on |t_ref - t_pred|,
         a pair is a TP if its error <= d.  An unmatched prediction within d
         of a masked (unverified) transition is don't-care: neither TP nor FP.
  MAE    mean |error| over all Hungarian-assigned pairs, no tolerance cut.
  dK     mean over held-out segments of |K_hat - K_ref|.

Copied unchanged from the working tree except for imports.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def per_sample_counts(truths, pool, sids, s2i, tolerances):
    """
    Returns {sid: {'tp':{d:..}, 'fp':{d:..}, 'fn':{d:..}, 'errors':[..],
                   'n_pred':int, 'n_ref':int, 'n_masked_hits':{d:..}}}
    """
    out = {}
    for sid in sids:
        ref = pool[sid]['ref']
        mask = pool[sid]['mask']
        pred = sorted(truths.get(s2i[sid], []))
        rec = {'tp': {}, 'fp': {}, 'fn': {}, 'errors': [],
               'n_pred': len(pred), 'n_ref': len(ref), 'n_masked_hits': {}}

        pairs = []
        if ref and pred:
            cost = np.abs(np.array(ref)[:, None] - np.array(pred)[None, :])
            rows, cols = linear_sum_assignment(cost)
            pairs = [(int(r), int(c), float(cost[r, c]))
                     for r, c in zip(rows, cols)]
            rec['errors'] = [e for _, _, e in pairs]

        for d in tolerances:
            tp_pairs = [(r, c) for r, c, e in pairs if e <= d]
            used = {c for _, c in tp_pairs}
            tp = len(tp_pairs)
            fn = len(ref) - tp
            fp = 0
            masked_hits = 0
            for j, x in enumerate(pred):
                if j in used:
                    continue
                if mask and min(abs(x - m) for m in mask) <= d:
                    masked_hits += 1      # don't-care: neither TP nor FP
                    continue
                fp += 1
            rec['tp'][d], rec['fp'][d], rec['fn'][d] = tp, fp, fn
            rec['n_masked_hits'][d] = masked_hits
        out[sid] = rec
    return out


def aggregate(counts, sids, tolerances):
    errs = [e for s in sids for e in counts[s]['errors']]
    res = {'matched_mae': float(np.mean(errs)) if errs else None,
           'n_pred': sum(counts[s]['n_pred'] for s in sids),
           'n_ref': sum(counts[s]['n_ref'] for s in sids),
           'f1_at': {}}
    for d in tolerances:
        tp = sum(counts[s]['tp'][d] for s in sids)
        fp = sum(counts[s]['fp'][d] for s in sids)
        fn = sum(counts[s]['fn'][d] for s in sids)
        mh = sum(counts[s]['n_masked_hits'][d] for s in sids)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        res['f1_at'][d] = {
            'precision': p, 'recall': r,
            'f1': 2 * p * r / (p + r) if p + r else 0.0,
            'tp': tp, 'fp': fp, 'fn': fn, 'masked_hits': mh,
        }
    return res



def macro_f1(counts, sids, tol):
    """Mean of per-sample F1, over samples that have reference boundaries."""
    vals = []
    for s in sids:
        c = counts[s]
        if c['n_ref'] == 0:
            continue
        tp, fp, fn = c['tp'][tol], c['fp'][tol], c['fn'][tol]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(vals)) if vals else None


def delta_k(truths, pool, sids, s2i):
    """dK = mean |predicted count - reference count| over `sids`."""
    pred = np.array([len(truths.get(s2i[s], [])) for s in sids], dtype=float)
    ref = np.array([len(pool[s]['ref']) for s in sids], dtype=float)
    return float(np.abs(pred - ref).mean())
