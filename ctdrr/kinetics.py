"""
The anonymous crowd panels: Kinetics-GEBD and Kinetics-GEB+.

Neither corpus has ground truth, and neither releases annotator identity --
each clip is labelled by a few workers drawn from a pool, with no identity
persisting across clips and no annotation order.  So the two corpora are used
to score each other: one corpus's annotations are aggregated, and the result
is scored against a reference built from the *other* corpus's annotators on
the same clips.  No method ever sees the annotations its reference came from.

The reference keeps a mark supported by at least `ref_support` distinct
annotators of the other corpus, within the Rel.Dis tolerance of 0.05 x clip
duration; surviving marks within tolerance are merged into one boundary at
their mean.  Marks that fail the support rule were still placed by real
annotators, so they become don't-care: a prediction landing on one is neither
a true nor a false positive.

Only the 5,494 clips present in both corpora are used, with GEBD restricted to
the five transition causes that GEB+ also defines, so both sides annotate the
same notion of a boundary.
"""

import numpy as np

from .gebd_data import load_gebd, to_annotator_data, SHARED_CAUSES
from .gebplus_data import load_gebplus_raw, supported_reference, gebd_durations
from .kinetics_eval import REL_DIS

DIRECTIONS = ('gebd', 'gebplus')
REF_SUPPORT = 3
SUPPORT_FILTER = (1, 2, 3)


def build_direction(direction='gebd', ref_support=REF_SUPPORT,
                    min_annotators=3, min_f1_consis=0.3):
    """
    (videos, reference, mask) for one scoring direction.

    `direction` names the corpus being AGGREGATED; the reference comes from
    the other one.  Clips whose reference is empty after the support rule are
    dropped.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f'direction must be one of {DIRECTIONS}')

    durations = {}
    for split in ('train', 'val'):
        durations.update(gebd_durations(split))

    gebplus = []
    for split in ('train', 'val', 'test'):
        part, _ = load_gebplus_raw(split, min_annotators=min_annotators,
                                   durations=durations)
        gebplus += part
    gebd = []
    for split in ('train', 'val'):
        gebd += load_gebd(split, min_annotators=min_annotators,
                          min_f1_consis=min_f1_consis,
                          keep_causes=SHARED_CAUSES)

    by_gp = {v['vid']: v for v in gebplus}
    by_gd = {v['vid']: v for v in gebd}
    overlap = sorted(set(by_gp) & set(by_gd))

    if direction == 'gebd':
        videos, source = [by_gd[v] for v in overlap], [by_gp[v] for v in overlap]
    else:
        videos, source = [by_gp[v] for v in overlap], [by_gd[v] for v in overlap]

    reference, mask = supported_reference(source, min_support=ref_support,
                                          return_excluded=True)
    videos = [v for v in videos if v['vid'] in reference]
    return videos, reference, mask


def split_videos(videos, seed):
    """(dev, test) -- disjoint halves of the clips for this seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(videos))
    half = len(videos) // 2
    return ([videos[i] for i in sorted(perm[:half])],
            [videos[i] for i in sorted(perm[half:])])


def apply_support(truths, videos, min_support, rel_dis=REL_DIS):
    """
    Post-hoc filter offered identically to every method: keep a predicted
    boundary only if at least `min_support` input annotators marked within
    the matching tolerance of it.
    """
    if min_support <= 1:
        return truths
    out = {}
    for i, v in enumerate(videos):
        tol = rel_dis * v['duration']
        out[i] = [x for x in truths.get(i, [])
                  if sum(1 for a in v['anns']
                         if a and min(abs(y - x) for y in a) <= tol)
                  >= min_support]
    return out


def as_annotator_data(videos):
    """Videos -> the per-annotator record lists the methods consume."""
    return to_annotator_data(videos)[0]
