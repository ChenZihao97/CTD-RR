"""
The fixed, ordered panel: 2,904 medical instructional segments annotated by
three trained annotators, with a tier-A reference over 2,009 of them.

The reference is not ground truth.  Each interior boundary carries the sources
that proposed it and the independent signals that corroborated it, which gives
three nested reference definitions:

  R1_all        every tier-A boundary, nothing masked
  R2_verified   every boundary with independent support; the rest become
                don't-care, so a prediction landing on one is neither TP nor FP
                -- this is the definition the paper reports
  R3_proposed   only boundaries an independent source proposed

Held-out protocol: ten random splits of the eligible samples into 500 tuning
and 500 test; every method is tuned on the tuning half and scored once on the
disjoint test half.
"""

import json

import numpy as np

from .config import MEDVID_ANNOTATORS, MEDVID_REFERENCE, require

INDEPENDENT = {'gemini', 'fourth_source'}
REFKEYS = ('R1_all', 'R2_verified', 'R3_proposed')
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
    """{sample_id: {'all', 'verified', 'proposed', 'unverified'}}."""
    with open(require(path or MEDVID_REFERENCE), encoding='utf-8') as f:
        raw = json.load(f)
    out = {}
    for sid, rec in raw.items():
        sid = int(sid)
        if sample_index is not None and sid not in sample_index:
            continue
        allb, ver, prop, unver = [], [], [], []
        for b in rec['boundaries']:
            src, sig = set(b['sources']), set(b['signals'])
            allb.append(b['t'])
            independent = bool(src & INDEPENDENT)
            if independent:
                prop.append(b['t'])
            if independent or sig:
                ver.append(b['t'])
            else:
                unver.append(b['t'])
        if allb:
            out[sid] = {'all': sorted(allb), 'verified': sorted(ver),
                        'proposed': sorted(prop), 'unverified': sorted(unver)}
    return out


def make_pools(refs):
    """Reference and don't-care mask for each of the three definitions."""
    pools = {k: {} for k in REFKEYS}
    for sid, v in refs.items():
        proposed = set(v['proposed'])
        pools['R1_all'][sid] = {'ref': v['all'], 'mask': []}
        pools['R2_verified'][sid] = {'ref': v['verified'],
                                     'mask': v['unverified']}
        pools['R3_proposed'][sid] = {'ref': v['proposed'],
                                     'mask': [t for t in v['all']
                                              if t not in proposed]}
    return pools


def split_for(seed, eligible):
    """(test_ids, tune_ids) -- disjoint halves of 500 samples each."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eligible))
    test = sorted(eligible[i] for i in perm[:N_TEST])
    tune = sorted(eligible[i] for i in perm[N_TEST:N_TEST + N_TUNE])
    return test, tune


def load_all():
    """(annotations, sample_index, refs, pools, eligible_sample_ids)."""
    data = load_annotations()
    s2i = build_sample_index(data)
    refs = load_reference(sample_index=s2i)
    return data, s2i, refs, make_pools(refs), sorted(refs)
