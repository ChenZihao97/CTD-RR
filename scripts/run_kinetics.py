"""
The anonymous crowd panels: Kinetics-GEBD and Kinetics-GEB+.  Reproduces the
right column of Table 1 and the reliability ablation.

Five random splits of the clips into disjoint dev and test halves.  Every
method is tuned on the dev half by coordinate ascent over all of its
parameters -- including a post-hoc support filter offered identically to all
of them -- and scored once on the test half against a reference built from
the other corpus's annotators.

Run:
    python scripts/run_kinetics.py --seeds 5 --workers 6
    python scripts/run_kinetics.py --direction gebplus --ablation
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
from ctdrr import kinetics                                    # noqa: E402
from ctdrr.config import RESULTS_DIR                          # noqa: E402
from ctdrr.grids import (BASELINES, CTD_KINETICS,             # noqa: E402
                         CTD_ABLATIONS, PAPER_ORDER)
from ctdrr.kinetics_eval import eval_external                 # noqa: E402

OURS = 'CTD-RR (ours)'
_G = {}


def _init(videos, ref, mask, direction, ablation):
    """Workers inherit the corpus; building it once costs a minute."""
    _G.update(videos=videos, ref=ref, mask=mask, direction=direction,
              ablation=ablation, halves={})


def _half(seed):
    if seed not in _G['halves']:
        dev, test = kinetics.split_videos(_G['videos'], seed)
        _G['halves'][seed] = (dev, kinetics.as_annotator_data(dev),
                              test, kinetics.as_annotator_data(test))
    return _G['halves'][seed]


def _predict(name, cfg, data):
    if name in CTD_ABLATIONS:
        return ctd_run(data, **{**CTD_ABLATIONS[name], **cfg})
    if name == OURS:
        return ctd_run(data, estimator=CTD_KINETICS['estimator'],
                       shrink_centre=CTD_KINETICS['shrink_centre'], **cfg)
    return BASELINES[name]['fn'](data, **cfg)


def _f1(truths, videos):
    return eval_external(truths, videos, _G['ref'], mask=_G['mask'])


def _score(name, cfg, data, videos):
    """Best F1 over the support filter, or None if the configuration fails."""
    try:
        truths = _predict(name, cfg, data)
    except Exception:
        return None, 1
    best, best_s = -1.0, 1
    for s in kinetics.SUPPORT_FILTER:
        f = _f1(kinetics.apply_support(truths, videos, s), videos)['f1']
        if f > best:
            best, best_s = f, s
    return best, best_s


def _one(job):
    name, seed = job
    t0 = time.time()
    dev, dev_data, test, test_data = _half(seed)

    if name in CTD_ABLATIONS and name != OURS:
        grid = {k: v for k, v in CTD_KINETICS['grid'].items()
                if k != 'm_shrink'}
    elif name == OURS or name in CTD_ABLATIONS:
        grid = CTD_KINETICS['grid']
    else:
        grid = BASELINES[name]['kinetics']

    support = {'value': 1}

    def score(kw):
        f, s = _score(name, kw, dev_data, dev)
        if f is not None:
            support['value'] = s
        return f

    cfg, dev_f1, n_eval = coordinate_ascent(score, grid)
    _score(name, cfg, dev_data, dev)          # restore the chosen filter

    truths = kinetics.apply_support(_predict(name, cfg, test_data), test,
                                    support['value'])
    e = _f1(truths, test)
    return {'method': name, 'seed': seed, 'config': cfg,
            'support_filter': support['value'], 'dev_f1': dev_f1,
            'n_eval': n_eval, 'seconds': time.time() - t0,
            'f1': e['f1'], 'precision': e['precision'], 'recall': e['recall']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--direction', default='gebd',
                    help="'gebd' aggregates Kinetics-GEBD against a GEB+ "
                         "reference (the reported setting); 'gebplus' is the "
                         "mirrored direction; 'gebd,gebplus' runs both")
    ap.add_argument('--ablation', action='store_true',
                    help='run the reliability ablation instead of the '
                         'baseline comparison')
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = 'ablation' if args.ablation else 'baselines'
    log = open(RESULTS_DIR / f'kinetics_{tag}_{stamp}.txt', 'w',
               encoding='utf-8')

    def P(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()

    names = (list(CTD_ABLATIONS) if args.ablation
             else [n for n in PAPER_ORDER])
    P('=' * 88)
    P('Anonymous crowd panels -- 5 seeded dev/test halves, '
      'cross-corpus reference')
    P(f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    P('=' * 88)

    payload = {}
    t0 = time.time()
    for direction in args.direction.split(','):
        ts = time.time()
        videos, ref, mask = kinetics.build_direction(direction)
        n_ref = sum(len(r) for r in ref.values())
        P(f'\n{direction}: {len(videos)} clips, {n_ref} reference boundaries '
          f'({n_ref / max(len(videos), 1):.2f} per clip), '
          f'{sum(len(m) for m in mask.values())} masked  '
          f'[{time.time() - ts:.0f}s]')
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(videos, ref, mask, direction,
                                           args.ablation)) as ex:
            rows = list(ex.map(_one, [(n, s) for s in range(args.seeds)
                                      for n in names]))
        by = {}
        for r in rows:
            by.setdefault(r['method'], []).append(r)

        def ms(name, key):
            v = np.array([r[key] for r in by[name]], dtype=float)
            return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)

        other = 'GEB+' if direction == 'gebd' else 'GEBD'
        P('\n' + '=' * 88)
        P(f'{direction.upper()} aggregated, reference from {other}')
        P('=' * 88)
        P(f'  {"Method":<28}{"F1":>15}{"P":>9}{"R":>9}   filter')
        order = sorted(by, key=lambda n: -ms(n, 'f1')[0])
        for name in order:
            m, s = ms(name, 'f1')
            P(f'  {name:<28}{m:>9.4f}+/-{s:<5.4f}'
              f'{ms(name, "precision")[0]:>9.3f}{ms(name, "recall")[0]:>9.3f}'
              f'   {by[name][0]["support_filter"]}')

        if OURS in by:
            P(f'\n  paired per split, comparator minus {OURS}')
            base = {r['seed']: r['f1'] for r in by[OURS]}
            for name in order:
                if name == OURS:
                    continue
                comp = {r['seed']: r['f1'] for r in by[name]}
                seeds = sorted(set(base) & set(comp))
                d = np.array([comp[s] - base[s] for s in seeds])
                P(f'    {name:<28}{d.mean():>+9.4f}   ours higher on '
                  f'{int((d < 0).sum())}/{len(seeds)} splits')
        payload[direction] = rows

    out = RESULTS_DIR / f'kinetics_{tag}_{stamp}.json'
    json.dump(payload, open(out, 'w', encoding='utf-8'), indent=1,
              default=float)
    P(f'\nTotal {time.time() - t0:.0f}s -> {out}')
    log.close()


if __name__ == '__main__':
    main()
