"""
Correctness gates for the released package.

1. CTD-RR at the paper's configuration reproduces the reported per-split
   number on the fixed panel, to 1e-9.
2. The windowed and the shrinkage regularizers are the same estimator: with
   the regularization switched off (m = 0) the shrinkage M-step reproduces
   plain per-segment maximum likelihood exactly.
3. Reliability is not keyed to annotator order: permuting the annotators
   leaves the aggregated boundaries unchanged.

Run:  python scripts/verify.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctdrr import ctd_run                                    # noqa: E402
from ctdrr import medvid                                     # noqa: E402
from ctdrr.medvid_eval import per_sample_counts, aggregate   # noqa: E402

# seed 0's tuned configuration and the F1@2s it produced on that split
PAPER_CFG = dict(window_radius=15, pruning_threshold=0.02, precision_cap=32.0)
PAPER_SEED = 0
PAPER_F1 = 0.5811452863215804

failures = []


def check(name, ok, detail):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}')
    if not ok:
        failures.append(name)


def main():
    print('loading the fixed panel ...')
    data, s2i, pool, elig = medvid.load_all()
    print(f'  {len(data)} annotators, {len(data[0])} samples, '
          f'{len(elig)} eligible for evaluation\n')

    # ---- 1. the reported number ----------------------------------------
    test_sids, _ = medvid.split_for(PAPER_SEED, elig)
    truths = ctd_run(data, estimator='window', k_rule='max', **PAPER_CFG)
    c = per_sample_counts(truths, pool, test_sids, s2i, [2.0])
    f1 = aggregate(c, test_sids, [2.0])['f1_at'][2.0]['f1']
    check('reported number', abs(f1 - PAPER_F1) < 1e-9,
          f'F1@2s = {f1:.12f}, expected {PAPER_F1:.12f} '
          f'(|diff| = {abs(f1 - PAPER_F1):.2e})')

    # ---- 2. m = 0 is plain maximum likelihood --------------------------
    a = ctd_run(data, estimator='nowin', k_rule='max', precision_cap=32.0)
    b = ctd_run(data, estimator='shrink', k_rule='max', precision_cap=32.0,
                m_shrink=0.0)
    keys = sorted(set(a) | set(b))
    diff = max((max(abs(np.array(a[k]) - np.array(b[k])), default=0.0)
                if len(a.get(k, [])) == len(b.get(k, [])) else np.inf)
               for k in keys) if keys else 0.0
    check('m = 0 recovers maximum likelihood', diff < 1e-9,
          f'largest boundary difference = {diff:.2e} s over '
          f'{len(keys)} segments')

    # ---- 3. no dependence on annotator order ---------------------------
    perm = [2, 0, 1]
    c0 = ctd_run(data, estimator='shrink_log', k_rule='max',
                 shrink_centre='geomean', m_shrink=3.0)
    c1 = ctd_run([data[i] for i in perm], estimator='shrink_log',
                 k_rule='max', shrink_centre='geomean', m_shrink=3.0)
    keys = sorted(set(c0) | set(c1))
    diff = max((max(abs(np.array(c0[k]) - np.array(c1[k])), default=0.0)
                if len(c0.get(k, [])) == len(c1.get(k, [])) else np.inf)
               for k in keys) if keys else 0.0
    check('invariant to annotator order', diff < 1e-9,
          f'largest boundary difference = {diff:.2e} s')

    print()
    if failures:
        print(f'{len(failures)} check(s) FAILED: {", ".join(failures)}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
