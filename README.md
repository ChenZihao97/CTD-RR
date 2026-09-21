# CTD-RR

Aggregating temporal boundary annotations from a few annotators is a
continuous truth discovery problem: the number of true boundaries, their
correspondence to the annotated ones, and each annotator's reliability are all
unknown. CTD-RR is a latent-alignment Gaussian mixture model whose EM inference
jointly estimates latent boundaries, their soft correspondence to the annotated
ones, and per-segment per-annotator reliability, so that matching and
aggregation are solved together rather than in sequence.

Per-segment reliability rests on few residuals, so it is regularized by
borrowing strength from neighboring segments when fixed annotators label the
corpus in sequence, and from the corpus as a whole when annotators are
anonymous crowd workers — one estimator that adapts to both annotation regimes.

```python
from ctdrr import ctd_run

# fixed, ordered panel — borrow strength from neighboring segments
truths = ctd_run(annotations, estimator='window', window_radius=15)

# anonymous crowd panel — borrow strength across the corpus
truths = ctd_run(annotations, estimator='shrink_log',
                 shrink_centre='geomean', m_shrink=3.0)
```

`annotations` is a list with one entry per annotator; each entry is a list of
per-segment records holding that annotator's boundary times. The call returns
`{segment index: [boundary seconds]}`.

## Results

Held-out results in both regimes, mean ± sd.

| Method | F1@1s | F1@2s | MAE (s) | ΔK | Anon. F1 |
|---|---|---|---|---|---|
| HDBSCAN | 0.391±.008 | 0.520±.010 | 4.08±.13 | 1.42±.06 | 0.795±.002 |
| KDE+peaks | 0.406±.014 | 0.520±.008 | 3.10±.29 | 2.23±.34 | 0.800±.003 |
| BSC | 0.364±.013 | 0.515±.013 | 3.88±.46 | 1.92±.47 | 0.791±.003 |
| CRH | 0.351±.011 | 0.501±.010 | 3.76±.12 | 1.36±.07 | 0.766±.001 |
| GTM | 0.378±.012 | 0.505±.010 | 3.86±.34 | 1.37±.08 | 0.765±.002 |
| CATD | 0.316±.008 | 0.471±.008 | 4.05±.11 | 1.36±.07 | 0.765±.002 |
| KDEm | 0.390±.012 | 0.517±.010 | **2.99**±.17 | 2.32±.27 | **0.807**±.002 |
| EvolvT | 0.383±.017 | 0.512±.012 | 3.69±.18 | 1.36±.07 | 0.766±.001 |
| DynaTD | 0.404±.016 | 0.524±.012 | 3.69±.16 | 1.36±.07 | 0.765±.002 |
| &nbsp;&nbsp;with decay | 0.391±.014 | 0.514±.010 | 3.82±.13 | 1.35±.07 | 0.757±.004 |
| **CTD-RR (ours)** | **0.432**±.017 | **0.561**±.013 | 3.20±.13 | **1.32**±.08 | 0.794±.003 |

The fixed panel is 2,904 medical instructional segments annotated by three
trained annotators, over ten random splits into 500 tuning and 500 test. The
anonymous panel aggregates crowd annotations of Kinetics-GEBD video segments
and scores them against a reference built from the GEB+ annotations of the same
clips, over five seeded dev/test halves.

Every method — ours and all nine baselines — is tuned the same way: coordinate
ascent over every parameter the method accepts, maximizing F1 on the tuning
half, then scored once on the disjoint test half.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.10; NumPy, SciPy and scikit-learn are the only dependencies.

## Data

The fixed panel ships with the repository under `data/medvid/`: three
annotators' boundary times and the evaluation reference.

The Kinetics-GEBD and GEB+ annotations are third-party releases and are not
redistributed here. Run `python scripts/download_kinetics.py` for the sources
and the expected layout under `data/kinetics/`. Both are annotation-only — no
video is needed.

To keep data or results elsewhere:

```bash
export CTDRR_DATA=/path/to/data CTDRR_RESULTS=/path/to/results
```

## Reproduce

```bash
python scripts/verify.py                   # correctness checks, ~1 min
python scripts/run_medvid.py               # fixed panel, 10 splits
python scripts/run_kinetics.py             # anonymous panel, 5 seeds
python scripts/run_kinetics.py --ablation  # reliability ablation
```

`--workers` sets the process count and `--seeds` shortens a run. Each script
writes a readable table and a JSON record carrying every per-split
configuration.

`verify.py` checks that CTD-RR at the reported configuration reproduces the
reported number, that switching the regularization off (m = 0) recovers plain
per-segment maximum likelihood, and that the output does not depend on the
order in which annotators are listed.

## Layout

```
ctdrr/
  em.py             core EM: soft correspondence, truth update, pruning
  reliability.py    the reliability estimators and ctd_run
  baselines.py      HDBSCAN, KDE+peaks, BSC
  baselines_td.py   CRH, GTM, CATD, KDEm, EvolvT, DynaTD (+decay)
  medvid.py         fixed panel: annotations, reference, splits
  kinetics.py       anonymous panel: cross-corpus reference, splits
  medvid_eval.py    F1@δ, MAE, ΔK
  kinetics_eval.py  Rel.Dis matching and F1
  grids.py          the methods compared and their tuning grids
  tuning.py         coordinate ascent
scripts/            runners, the download helper, the correctness checks
data/medvid/        the fixed panel (shipped)
data/kinetics/      the anonymous panel (fetched)
```

Two baselines carry names that are swapped relative to their paper titles:
**EvolvT** is the method of *"Dynamic Truth Discovery on Numerical Data"*
(ICDM 2018), and **DynaTD** is the method of *"On the Discovery of Evolving
Truth"* (KDD 2015). Each is used as its own authors named it.



## License

Apache-2.0 (see `LICENSE`).
