"""
Loader for the raw human annotations of Kinetics-GEBD.

Raw layout (k400_{train,val}_raw_annotation.pkl):

    {video_id: {'num_frames', 'path_video', 'fps', 'video_duration',
                'f1_consis': [per-annotator consistency],
                'f1_consis_avg': float,
                'substages_timestamps': [ann_0, ann_1, ...]}}

where each annotation is a list of {'start_time', 'end_time', 'label'}.

Annotators are anonymous and clip-local: slot j in one clip is not the same
person as slot j in another, and the clips carry no order, so per-annotator
quantities are defined only within a clip.  `official_preprocess` follows the
preprocessing released by the corpus authors.
"""

import os
import pickle

import numpy as np

from .config import GEBD_RAW_DIR, GEBPLUS_FILTERED_DIR

GEBD_DIR = str(GEBD_RAW_DIR)
GEBPLUS_DIR = str(GEBPLUS_FILTERED_DIR)
MIN_CHANGE_DURATION = 0.3


# =====================================================================
# Official preprocessing
# =====================================================================

EVENT_ONLY = ('EventChange',)

# The boundary CAUSE taxonomy shared with Kinetics-GEB+.  GEBD records the
# cause after the colon in `label`; GEB+ records the same vocabulary in its own
# `label` field.  The two agree on a subject-centric core and differ only in
# that GEBD adds six CAMERA-EFFECT causes GEB+ has no category for:
#   Change due to Cut / Pan / Fade-Dissolve-Gradual / Zoom, and
#   Change from/to slow motion / fast motion   (18.8% of GEBD marks)
# GEB+ contributes no category GEBD lacks.  'Change of Actor/Subject' and
# GEB+'s 'Change of Subject' are the same category under different wording.
SHARED_CAUSES = (
    'Change of Action',
    'Change of Actor/Subject',
    'Change of Object Being Interacted',
    'Change of Color',
    'Multiple',
)


def official_preprocess(ann, video_duration,
                        min_change_duration=MIN_CHANGE_DURATION,
                        keep_labels=None, keep_causes=None):
    """
    Port of `generate_frameidx_from_raw` from the GEBD authors'
    data/export/prepare_k400_release.ipynb, restricted to the timestamp path
    (the frame-index path is irrelevant here).

    This is a VERBATIM port, including the in-place `list.remove` inside `for`
    loops that the original uses.  Those loops skip elements, so the published
    k400_mr345_*.pkl ground-truth files contain exactly this behaviour; a
    "corrected" version would produce a different annotation set from the one
    every GEBD result in the literature is computed against.

    `keep_labels` optionally restricts which annotation categories survive,
    applied to the RAW marks before the rest of the pipeline runs, so the
    result is "what this annotator would have produced had only those
    categories been collected".  `EVENT_ONLY` keeps `EventChange` and drops
    both shot-change categories - which is what makes a GEBD-derived reference
    comparable to Kinetics-GEB+, whose taxonomy has no shot-change class at
    all.  Default `None` keeps everything, i.e. the published behaviour.

    Returns a sorted list of boundary timestamps in seconds.
    """
    if keep_labels is not None:
        ann = [p for p in ann
               if p['label'].split(' ')[0].strip().rstrip(':')
               in {k.rstrip(':') for k in keep_labels}]
    if keep_causes is not None:
        # Filter on the CAUSE (after the colon), which is the field whose
        # vocabulary GEB+ shares.  Exact rather than a proxy: the `kind` field
        # separates the same marks to within 650 of 458k.
        keep = set(keep_causes)
        ann = [p for p in ann
               if (p['label'].split(':', 1)[1].strip()
                   if ':' in p['label'] else '') in keep]
    change_shot_range_start, change_shot_range_end = [], []
    change_event, change_shot_timestamp = [], []

    for p in ann:
        st, et = p['start_time'], p['end_time']
        l = p['label'].split(' ')[0]
        if (st + et) / 2 < min_change_duration or \
           (st + et) / 2 > (video_duration - min_change_duration):
            continue
        if l == 'EventChange':
            change_event.append((st + et) / 2)
        elif l == 'ShotChangeGradualRange:':
            change_shot_range_start.append(st)
            change_shot_range_end.append(et)
        else:
            change_shot_timestamp.append((st + et) / 2)

    # merge overlapping shot ranges
    i = 0
    while i < len(change_shot_range_start) - 1:
        while change_shot_range_end[i] >= change_shot_range_start[i + 1]:
            change_shot_range_start.remove(change_shot_range_start[i + 1])
            if change_shot_range_end[i] <= change_shot_range_end[i + 1]:
                change_shot_range_end.remove(change_shot_range_end[i])
            else:
                change_shot_range_end.remove(change_shot_range_end[i + 1])
            if i == len(change_shot_range_start) - 1:
                break
        i += 1

    # drop instants that fall inside a shot range
    for cg in change_event:
        for i in range(len(change_shot_range_start)):
            if cg <= (change_shot_range_end[i] + min_change_duration) and \
               cg >= (change_shot_range_start[i] - min_change_duration):
                change_event.remove(cg)
                break
    for cg in change_shot_timestamp:
        for i in range(len(change_shot_range_start)):
            if cg <= (change_shot_range_end[i] + min_change_duration) and \
               cg >= (change_shot_range_start[i] - min_change_duration):
                change_shot_timestamp.remove(cg)
                break

    change_event.sort()
    change_shot_timestamp.sort()
    tmp_change_shot_timestamp = change_shot_timestamp
    tmp_change_event = change_event

    i = 0
    while i <= (len(change_event) - 2):
        if (change_event[i + 1] - change_event[i]) <= 2 * min_change_duration:
            tmp_change_event.remove(change_event[i + 1])
        else:
            i += 1
    i = 0
    while i <= (len(change_shot_timestamp) - 2):
        if (change_shot_timestamp[i + 1]
                - change_shot_timestamp[i]) <= 2 * min_change_duration:
            tmp_change_shot_timestamp.remove(change_shot_timestamp[i + 1])
        else:
            i += 1
    for i in range(len(tmp_change_shot_timestamp) - 1):
        j = 0
        while j <= (len(tmp_change_event) - 1):
            if abs(tmp_change_shot_timestamp[i]
                   - tmp_change_event[j]) <= 2 * min_change_duration:
                tmp_change_event.remove(tmp_change_event[j])
            else:
                j += 1

    change_shot_range = [(change_shot_range_start[i]
                          + change_shot_range_end[i]) / 2
                         for i in range(len(change_shot_range_start))]
    change_all = tmp_change_event + tmp_change_shot_timestamp \
        + change_shot_range
    change_all.sort()
    return change_all


# =====================================================================
# Loading
# =====================================================================

def load_gebd(split='val', min_annotators=3, min_f1_consis=0.0,
              preprocess=True, limit=None, seed=0, keep_labels=None,
              keep_causes=None):
    """
    Returns a list of video records, sorted by video id:

        {'vid', 'duration', 'fps', 'f1_consis' (list), 'f1_consis_avg',
         'anns': [[t, ...], ...]}

    `min_annotators=3` mirrors the official release, which keeps only videos
    with at least three annotations.  `min_f1_consis=0.3` mirrors the official
    evaluation script, which skips low-agreement videos.
    """
    path = os.path.join(GEBD_DIR, f"k400_{split}_raw_annotation.pkl")
    with open(path, 'rb') as f:
        raw = pickle.load(f, encoding='lartin1')

    out = []
    for vid in sorted(raw):
        v = raw[vid]
        st = v.get('substages_timestamps') or []
        if len(st) < min_annotators:
            continue
        try:
            dur = float(v['video_duration'])
            fps = float(v['fps'])
            f1a = float(v['f1_consis_avg'])
            f1c = [float(x) for x in v['f1_consis']]
        except (KeyError, TypeError, ValueError):
            continue
        if f1a < min_f1_consis:
            continue
        if preprocess:
            anns = [official_preprocess(a, dur, keep_labels=keep_labels,
                                        keep_causes=keep_causes)
                    for a in st]
        else:
            anns = [sorted((p['start_time'] + p['end_time']) / 2 for p in a)
                    for a in st]
        if all(len(a) == 0 for a in anns):
            continue
        out.append({'vid': vid, 'duration': dur, 'fps': fps,
                    'f1_consis': f1c, 'f1_consis_avg': f1a, 'anns': anns})

    if limit is not None and len(out) > limit:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(out), size=limit, replace=False))
        out = [out[i] for i in idx]
    return out


def load_gebplus(splits=('train', 'val', 'test')):
    """
    {video_id: sorted list of boundary seconds} from the Kinetics-GEB+ release.

    This is GEB+ version (a), the filtered set: for each video the GEB+ authors
    kept the boundaries of the single annotator most consistent with the rest,
    then merged the other annotators' captions onto those anchors.  So it is
    one annotator's boundary set per video, NOT a multi-annotator set - which
    is precisely what makes it useful here.  It comes from a separate
    annotation round with a different task definition (status change with a
    caption), so for the Kinetics-GEBD videos it overlaps, it is an *external*
    boundary reference that no method under test has seen.
    """
    import json
    out = {}
    for sp in splits:
        p = os.path.join(GEBPLUS_DIR, f"{sp}.json")
        if not os.path.exists(p):
            continue
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
        for vid, bl in d.items():
            ts = sorted(float(b['timestamp']) for b in bl
                        if b.get('timestamp') is not None)
            if ts:
                out.setdefault(vid, [])
                out[vid] = sorted(set(out[vid] + ts))
    return out


# =====================================================================
# Bridge into the truth-discovery harness
# =====================================================================

def to_annotator_data(videos, J=None):
    """
    Reshape video records into the `annotator_data_list` the truth-discovery
    methods expect: a list of J per-annotator "datasets", each a list of N
    per-video dicts with a `steps_list`.

    J defaults to the largest annotator count in `videos`; videos with fewer
    annotators leave the surplus slots empty, which every method already
    treats as "this source made no claim".

    Slot j is not a person: Kinetics-GEBD does not identify annotators
    across clips, so slot j means only "the j-th annotation of this clip".

    `step_caption_boundary` carries a float here rather than the 'MM:SS'
    string the fixed panel uses; `em.time_to_sec` passes numbers through
    unchanged.
    """
    if J is None:
        J = max(len(v['anns']) for v in videos)
    data = [[] for _ in range(J)]
    for v in videos:
        for j in range(J):
            ts = v['anns'][j] if j < len(v['anns']) else []
            f1c = v.get('f1_consis') or []
            data[j].append({
                'sample_id': v['vid'],
                'video_id': v['vid'],
                # per-(video, slot) consistency score, carried through purely
                # so `ctd_eb.f1_consis_init` can seed lambda from it.  It is
                # never read by the objective - see that function.
                'f1_consis': (float(f1c[j])
                              if j < len(f1c) and f1c[j] is not None
                              and np.isfinite(f1c[j]) else None),
                'segment_start': None,
                'segment_end': None,
                'segment_start_second': 0.0,
                'video_length': v['duration'],
                'steps_list': [{'step_caption_boundary': float(t)}
                               for t in ts],
            })
    return data, J

