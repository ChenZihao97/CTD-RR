"""
CTD-RR -- continuous truth discovery with regularized reliability.

Aggregating temporal boundary annotations from a few annotators: a latent
alignment Gaussian mixture whose EM jointly estimates the latent boundaries,
their soft correspondence to the annotated ones, and per-segment
per-annotator reliability.

Per-segment reliability rests on few residuals, so it is regularized by
borrowing strength:

  * along the annotation sequence, when fixed annotators label the corpus in
    order        -- `ctd_run(..., estimator='window')`
  * across the corpus, when the annotators are anonymous crowd workers
    -- `ctd_run(..., estimator='shrink_log', shrink_centre='geomean')`

Both are the same estimator with a different source of strength, selected by
one argument.
"""

from .reliability import ctd_run, estimate_precision, initialize_k, raw_stats
from .tuning import coordinate_ascent

__all__ = ['ctd_run', 'estimate_precision', 'initialize_k', 'raw_stats',
           'coordinate_ascent']
__version__ = '1.0.0'
