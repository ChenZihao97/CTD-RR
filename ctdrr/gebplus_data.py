"""
gebplus_data.py
===============
Loader for the *raw* (version b) Kinetics-GEB+ release — the multi-annotator
one, 170K boundaries over 12,434 videos.

Provenance, because the README links matter here: the Google Drive links in
`showlab/GEB-Plus` are dead (404), but `Yuxuan-W/GEB-plus` (pushed 2025-02-22)
carries updated NTU SharePoint links.  The raw folder resolves to
`/personal/yuxuan003_e_ntu_edu_sg/Documents/Public/GEBC/dataset/version b (raw)`
and holds `{train,val,test}_all_annotation.json`.  Anonymous download needs the
FedAuth cookie that the sharing link sets, after which the SharePoint REST API
(`_api/web/GetFileByServerRelativePath(...)/$value`) serves the files.

Layout:

    {"<video_id>[<kinetics class>]": [
        {"f1_score": float,                      # this annotator's consistency
         "boundary_list": [{"start_time", "end_time", "label",
                            "subject", "status_before", "status_after",
                            "action_of_cause"}, ...]},
        ...                                      # one entry per annotator
    ]}

How this differs from Kinetics-GEBD, and why both are worth running:

  * TASK DEFINITION.  GEB+ boundaries are *status changes* attached to a
    subject and a caption ("Change of Action", "Change of Subject", ...).
    Kinetics-GEBD boundaries are generic event boundaries and include shot
    changes.  The two are separate annotation rounds over overlapping videos,
    so each is an independent reference for the other.
  * DENSITY.  ~3.15 marks per annotator here against ~4.6 in GEBD, on the same
    10 s clips — so GEB+ is the sparser, more selective corpus.
  * PANEL SIZE.  1 to 9 annotators per video (mode 5), against a near-uniform 5
    in GEBD.  That variation is useful: it lets reliability weighting be
    measured against the number of annotators actually available.

What it shares with GEBD: the clips have no order, so CTD-RR's Gaussian window
over segment index has nothing to smooth along here either.

WHERE IT DIFFERS, AND IT MATTERS: GEB+ does NOT list its annotators in an
arbitrary order.  Every video's annotator list is sorted by `f1_score`
descending - verified on 100% of videos, mean f1_score by slot 0.710 / 0.686 /
0.652 / 0.605 / 0.513.  So slot index here is a QUALITY RANK, not an anonymous
label, and it is a rank derived from inter-annotator agreement.  Every method
that carries one weight per source across the corpus (CTD-RR's `global` and
`window` estimators; CRH, GTM, CATD, DynaTD and KDEm) is therefore handed real
information that Kinetics-GEBD does not contain.  `shuffle_slots` exists to
control for exactly this, and results on GEB+ should be read with the shuffled
numbers beside the natural ones.
"""

import os
import re
import json
import zlib

import numpy as np

from .config import GEBD_RAW_DIR, GEBPLUS_RAW_DIR as _GEBPLUS_RAW

GEBPLUS_RAW_DIR = str(_GEBPLUS_RAW)
NOMINAL_DURATION = 10.0

_CLASS_SUFFIX = re.compile(r'\[.*\]$')


def strip_id(key):
    """'8DoMr9pn4GQ[reading book]' -> '8DoMr9pn4GQ'."""
    return _CLASS_SUFFIX.sub('', key)


def load_gebplus_raw(split='val', min_annotators=3, min_f1=0.0,
                     durations=None, limit=None, seed=0,
                     shuffle_slots=None):
    """
    Records in the same shape `gebd_data.load_gebd` returns, so every protocol
    in `gebd_eval.py` and every method wrapper works unchanged:

        {'vid', 'duration', 'fps', 'f1_consis', 'f1_consis_avg', 'anns'}

    `durations` is an optional {video_id: seconds} map — pass Kinetics-GEBD's
    exact durations for the overlapping videos.  GEB+ ships no duration field,
    and the Rel.Dis tolerance is a fraction of it, so the fallback matters: all
    observed timestamps lie below 10 s and Kinetics clips are nominally 10 s,
    so `NOMINAL_DURATION` is used where no exact value is available.  The
    fraction of videos on the fallback is reported by the caller.

    A boundary annotated as an interval (end_time != start_time, ~2.6% of them)
    is reduced to its midpoint, matching the convention Kinetics-GEBD's own
    preprocessing uses for its range labels.

    `shuffle_slots` (an int seed) permutes the annotator order within every
    video.  THIS IS NOT COSMETIC.  Unlike Kinetics-GEBD, whose annotator list
    order is arbitrary, GEB+ ships its annotators SORTED BY `f1_score`
    DESCENDING - verified on 100% of videos, with mean f1_score by slot running
    0.710 / 0.686 / 0.652 / 0.605 / 0.513.  So slot index is a quality rank,
    and any method that pools a source's statistics across videos (CTD-RR's
    `global` and `window` estimators, and CRH / GTM / CATD / DynaTD / KDEm,
    which all hold one weight per source corpus-wide) is handed a genuine
    signal that the corpus would not contain if the annotators were listed in
    collection order.  Worse, `f1_score` is itself derived from inter-annotator
    agreement, so the ordering leaks agreement into the slot index.

    Shuffling the slots destroys that artifact and nothing else, which makes it
    the control that separates "pooling helps because per-annotator evidence is
    thin" from "pooling helps because the release sorted the annotators for
    us".  See `run_gebplus_slot_control.py`.

    No further filtering is applied.  Kinetics-GEBD has a published
    preprocessing rule (`min_change_duration`, shot-range merging) and it is
    followed there; GEB+ publishes none, so inventing one would be a silent
    departure from the released data.
    """
    path = os.path.join(GEBPLUS_RAW_DIR, f"{split}_all_annotation.json")
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    durations = durations or {}
    out, n_fallback = [], 0
    for key in sorted(raw):
        vid = strip_id(key)
        entries = raw[key] or []
        if len(entries) < min_annotators:
            continue

        if shuffle_slots is not None:
            # crc32, not hash(): Python's hash() is salted per process, so a
            # worker pool would shuffle each video differently and the control
            # would not be reproducible.
            rng_s = np.random.default_rng(
                zlib.crc32(key.encode()) ^ int(shuffle_slots))
            entries = [entries[i] for i in rng_s.permutation(len(entries))]

        anns, f1c = [], []
        for e in entries:
            bl = e.get('boundary_list') or []
            ts = sorted((float(b['start_time']) + float(b['end_time'])) / 2.0
                        for b in bl)
            anns.append(ts)
            s = e.get('f1_score')
            f1c.append(float(s) if s is not None else float('nan'))

        if all(len(a) == 0 for a in anns):
            continue
        valid = [x for x in f1c if not np.isnan(x)]
        f1a = float(np.mean(valid)) if valid else float('nan')
        if valid and f1a < min_f1:
            continue

        dur = durations.get(vid)
        if dur is None:
            dur = NOMINAL_DURATION
            n_fallback += 1
        out.append({'vid': vid, 'key': key, 'duration': float(dur),
                    'fps': None, 'f1_consis': f1c, 'f1_consis_avg': f1a,
                    'anns': anns})

    if limit is not None and len(out) > limit:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(out), size=limit, replace=False))
        out = [out[i] for i in idx]
    return out, n_fallback


# =====================================================================
# Cross-dataset references
# =====================================================================

def supported_reference(videos, min_support=3, rel_dis=0.05,
                        return_excluded=False):
    """
    {video_id: [seconds]} keeping only marks that at least `min_support`
    distinct annotators of THAT corpus placed within the matching tolerance,
    each cluster reported at its member mean.

    This is how one corpus becomes a reference for the other.  Using a single
    annotator would make the reference as noisy as the thing being measured;
    using every pooled mark would make it far denser than any method's output.
    Requiring majority support gives a reference whose density is close to one
    annotator's own count while being much more stable than one annotator, and
    it is computed from annotations no method under test has seen when the two
    corpora are swapped.
    """
    out = {}
    excluded = {}
    for v in videos:
        tol = rel_dis * v['duration']
        pts = sorted(t for a in v['anns'] for t in a)
        if not pts:
            continue
        keep, drop = [], []
        for t in pts:
            n = sum(1 for a in v['anns']
                    if a and min(abs(y - t) for y in a) <= tol)
            (keep if n >= min_support else drop).append(t)
        if drop:
            # Candidates the support rule removed.  They are marks real
            # annotators placed, so a prediction landing on one is not evidence
            # of error - it is charged as a false positive only if we forget to
            # mask it.  Same don't-care rule the MedVid R2/R3 references use.
            excluded[v['vid']] = sorted(drop)
        if not keep:
            continue
        # collapse the surviving marks into clusters, one boundary each
        merged, cur = [], [keep[0]]
        for t in keep[1:]:
            if t - cur[-1] <= tol:
                cur.append(t)
            else:
                merged.append(float(np.mean(cur)))
                cur = [t]
        merged.append(float(np.mean(cur)))
        out[v['vid']] = merged
    return (out, excluded) if return_excluded else out


def gebd_durations(split='val'):
    """{video_id: duration} from the Kinetics-GEBD raw annotations."""
    import pickle
    p = os.path.join(str(GEBD_RAW_DIR),
                 f"k400_{split}_raw_annotation.pkl")
    if not os.path.exists(p):
        return {}
    with open(p, 'rb') as f:
        raw = pickle.load(f, encoding='lartin1')
    out = {}
    for vid, v in raw.items():
        try:
            out[vid] = float(v['video_duration'])
        except (KeyError, TypeError, ValueError):
            continue
    return out
