"""
The fixed, ordered panel: 2,904 medical instructional segments, three trained
annotators.  Reproduces the left half of Table 1.

Ten random splits into 500 tuning / 500 test.  Every method is tuned on the
tuning half by coordinate ascent over all of its parameters, then scored once
on the disjoint test half.  Reported: F1@1s, F1@2s, MAE and dK, mean +/- sd
over the ten splits, plus paired per-split comparisons against CTD-RR.

Run:
    python scripts/run_medvid.py --seeds 10 --workers 6
"""

import os
import sys
import json
import time
import argparse
import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                            # noqa: E402

from ctdrr import ctd_run, coordinate_ascent                  # noqa: E402
from ctdrr import medvid                                      # noqa: E402
from ctdrr.config import RESULTS_DIR                          # noqa: E402
from ctdrr.grids import BASELINES, CTD_MEDVID, PAPER_ORDER    # noqa: E402
from ctdrr.medvid_eval import (per_sample_counts, aggregate,  # noqa: E402
                               delta_k)

OURS = 'CTD-RR (ours)'
TOLS = [1.0, 2.0]
TUNING_TOL = 2.0

_G = {}


def _init():
    data, s2i, pool, elig = medvid.load_all()
    _G.update(data=data, s2i=s2i, pool=pool, elig=elig)


def _predict(name, cfg):
    if name == OURS:
        return ctd_run(_G['data'], estimator=CTD_MEDVID['estimator'],
                       k_rule=CTD_MEDVID['k_rule'], **cfg)
    return BASELINES[name]['fn'](_G['data'], **cfg)


def _score(name, cfg, sids):
    """F1 at the tuning tolerance, or None if the configuration fails."""
    try:
        truths = _predict(name, cfg)
    except Exception:
        return None
    c = per_sample_counts(truths, _G['pool'], sids, _G['s2i'],
                          [TUNING_TOL])
    return aggregate(c, sids, [TUNING_TOL])['f1_at'][TUNING_TOL]['f1']


def _one(job):
    name, seed = job
    t0 = time.time()
    test_sids, tune_sids = medvid.split_for(seed, _G['elig'])
    grid = (CTD_MEDVID['grid'] if name == OURS
            else BASELINES[name]['medvid'])
    cfg, dev_f1, n_eval = coordinate_ascent(
        lambda kw: _score(name, kw, tune_sids), grid)

    truths = _predict(name, cfg)
    pool = _G['pool']
    c = per_sample_counts(truths, pool, test_sids, _G['s2i'], TOLS)
    a = aggregate(c, test_sids, TOLS)
    return {
        'method': name, 'seed': seed, 'config': cfg, 'dev_f1': dev_f1,
        'n_eval': n_eval, 'seconds': time.time() - t0,
        'f1_1s': a['f1_at'][1.0]['f1'], 'f1_2s': a['f1_at'][2.0]['f1'],
        'precision': a['f1_at'][2.0]['precision'],
        'recall': a['f1_at'][2.0]['recall'],
        'mae': a['matched_mae'],
        'dk': delta_k(truths, pool, test_sids, _G['s2i']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--methods', default='all',
                    help='comma-separated subset, or "all"')
    args = ap.parse_args()

    names = ([n for n in PAPER_ORDER] if args.methods == 'all'
             else args.methods.split(','))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log = open(RESULTS_DIR / f'medvid_{stamp}.txt', 'w', encoding='utf-8')

    def P(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()

    P('=' * 88)
    P('Fixed, ordered panel -- 10 held-out splits of 500 tuning / 500 test')
    P(f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    P('=' * 88)
    P(f'methods: {len(names)}   seeds: {args.seeds}\n')

    jobs = [(n, s) for s in range(args.seeds) for n in names]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init) as ex:
        for i, r in enumerate(ex.map(_one, jobs), 1):
            rows.append(r)
            if i % len(names) == 0:
                P(f'  seed {r["seed"]} done ({time.time() - t0:.0f}s)')

    by = {}
    for r in rows:
        by.setdefault(r['method'], []).append(r)

    def ms(name, key):
        v = np.array([r[key] for r in by[name]], dtype=float)
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)

    P('\n' + '=' * 88)
    P(f'RESULTS -- mean +/- sd over {args.seeds} splits')
    P('=' * 88)
    P(f'  {"Method":<18}{"F1@1s":>15}{"F1@2s":>15}{"MAE (s)":>14}{"dK":>13}')
    for name in [n for n in PAPER_ORDER if n in by]:
        o = f'  {name:<18}'
        for k in ('f1_1s', 'f1_2s'):
            m, s = ms(name, k)
            o += f'{m:>9.3f}+/-{s:<5.3f}'
        for k in ('mae', 'dk'):
            m, s = ms(name, k)
            o += f'{m:>8.2f}+/-{s:<4.2f}'
        P(o)

    if OURS in by:
        P(f'\n  paired per split, {OURS} minus comparator '
          f'(error metrics signed so positive = ours better)')
        P(f'  {"Comparator":<18}{"dF1@1s":>10}{"wins":>7}{"dF1@2s":>10}'
          f'{"wins":>7}{"dMAE":>9}{"wins":>7}{"ddK":>9}{"wins":>7}')
        base = {r['seed']: r for r in by[OURS]}
        for name in [n for n in PAPER_ORDER if n in by and n != OURS]:
            comp = {r['seed']: r for r in by[name]}
            seeds = sorted(set(base) & set(comp))
            o = f'  {name:<18}'
            for k, higher_is_better in (('f1_1s', True), ('f1_2s', True),
                                        ('mae', False), ('dk', False)):
                d = np.array([base[s][k] - comp[s][k] for s in seeds])
                if not higher_is_better:
                    d = -d
                w = 9 if k in ('mae', 'dk') else 10
                o += f'{d.mean():>+{w}.3f}{int((d > 0).sum()):>4d}/{len(seeds)}'
            P(o)

    out = RESULTS_DIR / f'medvid_{stamp}.json'
    json.dump({'seeds': args.seeds, 'rows': rows},
              open(out, 'w', encoding='utf-8'), indent=1, default=float)
    P(f'\nTotal {time.time() - t0:.0f}s -> {out}')
    log.close()


if __name__ == '__main__':
    main()
