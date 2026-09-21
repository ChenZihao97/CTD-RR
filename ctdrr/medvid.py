"""
The fixed, ordered panel: instructional video segments, each annotated
independently by three annotators.

Loads the annotations, the evaluation reference, and the held-out splits.
Every sample carries a reference boundary set and a masked set; a prediction
falling within tolerance of a masked boundary counts as neither a true nor a
false positive.
"""

import json

import numpy as np

from .config import MEDVID_ANNOTATORS, MEDVID_REFERENCE, require

N_TEST = 500
N_TUNE = 500
TOLERANCES = (1.0, 2.0)
PRIMARY_TOL = 2.0


def load_annotations(paths=None):
    """One list of per-sample records per annotator, sorted by sample_id."""
    paths = paths or MEDVID_ANNOTATORS
    out = []
    for p in paths:
        with open(require(p), encoding='utf-8') as f:
            out.append(sorted(json.load(f), key=lambda r: r['sample_id']))
    n = {len(a) for a in out}
    if len(n) != 1:
        raise ValueError(f'annotator files disagree on sample count: {n}')
    return out


def build_sample_index(annotator_data_list):
    """{sample_id: index into the sorted per-annotator lists}."""
    return {rec['sample_id']: t
            for t, rec in enumerate(annotator_data_list[0])}


def load_reference(path=None, sample_index=None):
    """{sample_id: {'ref': [seconds], 'mask': [seconds]}}."""
    with open(require(path or MEDVID_REFERENCE), encoding='utf-8') as f:
        raw = json.load(f)
    pool = {}
    for sid, rec in raw.items():
        sid = int(sid)
        if sample_index is not None and sid not in sample_index:
            continue
        pool[sid] = {'ref': list(rec['reference']),
                     'mask': list(rec['masked'])}
    return pool


def split_for(seed, eligible):
    """(test_ids, tune_ids) -- disjoint halves of 500 samples each."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eligible))
    test = sorted(eligible[i] for i in perm[:N_TEST])
    tune = sorted(eligible[i] for i in perm[N_TEST:N_TEST + N_TUNE])
    return test, tune


def load_all():
    """(annotations, sample_index, pool, eligible_sample_ids)."""
    data = load_annotations()
    s2i = build_sample_index(data)
    pool = load_reference(sample_index=s2i)
    return data, s2i, pool, sorted(pool)
