"""
The methods compared in the paper, and the grids they are tuned over.

Every method -- ours and all nine baselines -- is tuned over EVERY parameter
it accepts, by the same coordinate ascent on the same tuning half.  The two
regimes need different ranges because their time scales differ by two orders
of magnitude: minutes-long instructional segments with a 2 s tolerance, and
~10 s clips with a ~0.5 s tolerance.
"""

from .baselines import baseline_hdbscan, baseline_kde_peaks, baseline_bsc
from .baselines_td import (baseline_crh, baseline_gtm, baseline_catd,
                           baseline_kdem, baseline_evolvt_kalman,
                           baseline_dynatd_kdd15, baseline_dynatd_decay)

# --- our method ----------------------------------------------------------
# Fixed in both regimes: initial reliability lambda_0 = 1 (sigma_0 = 1 s),
# at most 10 EM iterations, degenerate-slot merge 0.5 s.  The convergence
# threshold is a time scale, so it is set per corpus (2 s / 0.02 s).
CTD_MEDVID = {
    'estimator': 'window',
    'k_rule': 'max',
    'grid': {
        'window_radius': [5, 10, 15, 20, 30],
        'pruning_threshold': [0.02, 0.05, 0.10, 0.20, 0.30, 1.0],
        'precision_cap': [4.0, 8.0, 16.0, 32.0],
    },
}

CTD_KINETICS = {
    'estimator': 'shrink_log',
    'shrink_centre': 'geomean',
    'grid': {
        'k_rule': ['max', 'median', 'majority', 'mean'],
        'pruning_threshold': [0.05, 1.0, 3.0],
        'degeneracy_threshold': [0.15, 0.5],
        'precision_cap': [8.0, 32.0],
        'convergence_threshold': [0.02],
        # 0 is deliberately excluded: "shrinkage, tuned" and "no shrinkage"
        # are separate rows of the ablation, so the tuned row can never win
        # by quietly selecting the ablated one.
        'm_shrink': [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0],
    },
}

# Ablation rows: the same estimator with one component removed.
CTD_ABLATIONS = {
    'CTD-RR (ours)': dict(estimator='shrink_log', shrink_centre='geomean'),
    'no regularization (m = 0)': dict(estimator='shrink', m_shrink=0.0),
    'one reliability per segment': dict(estimator='video_shared'),
    'no reliability weighting': dict(estimator='uniform'),
}

# --- the nine baselines --------------------------------------------------
BASELINES = {
    'HDBSCAN': {
        'fn': baseline_hdbscan,
        'medvid': {'min_cluster_size': [2, 3, 4], 'min_samples': [1, 2, 3],
                   'cluster_selection_epsilon': [0.0, 1.0, 2.0, 3.0, 5.0],
                   'cluster_selection_method': ['eom', 'leaf'],
                   'center': ['median', 'mean'], 'min_annotators': [1, 2]},
        'kinetics': {'min_cluster_size': [2, 3, 4], 'min_samples': [1, 2, 3],
                     'cluster_selection_epsilon': [0.0, 0.2, 0.5, 1.0],
                     'cluster_selection_method': ['eom', 'leaf'],
                     'center': ['median', 'mean'], 'min_annotators': [1]},
    },
    'KDE+peaks': {
        'fn': baseline_kde_peaks,
        'medvid': {'bandwidth': [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
                   'prominence_frac': [0.0, 0.02, 0.05, 0.10, 0.20],
                   'height_frac': [0.0, 0.05, 0.10], 'min_annotators': [1, 2]},
        'kinetics': {'bandwidth': [0.1, 0.2, 0.3, 0.5, 0.8, 1.2],
                     'prominence_frac': [0.0, 0.05, 0.10, 0.20],
                     'height_frac': [0.0, 0.05], 'min_annotators': [1]},
    },
    'BSC': {
        'fn': baseline_bsc,
        'medvid': {'bin_seconds': [2.0, 3.0, 4.0, 6.0, 8.0],
                   'alpha_diag': [10.0, 50.0, 200.0],
                   'posterior_threshold': [0.3, 0.5, 0.7],
                   'seq_model': [True, False], 'beta_self': [0.5, 1.0, 5.0],
                   'beta_switch': [0.5, 1.0, 5.0, 20.0],
                   'alpha_off': [0.5, 1.0, 2.0, 5.0],
                   'merge_runs': [True, False], 'n_vb_iter': [10, 25, 50]},
        'kinetics': {'bin_seconds': [0.2, 0.4, 0.8],
                     'alpha_diag': [10.0, 50.0, 200.0],
                     'posterior_threshold': [0.3, 0.5, 0.7],
                     'seq_model': [False, True], 'beta_self': [0.5, 1.0, 5.0],
                     'beta_switch': [0.5, 1.0, 5.0, 20.0],
                     'alpha_off': [0.5, 1.0, 2.0, 5.0],
                     'merge_runs': [True, False], 'n_vb_iter': [10, 25, 50]},
    },
    'CRH': {
        'fn': baseline_crh,
        'medvid': {'K_rule': ['median', 'majority', 'max'],
                   'scale_mode': ['segment_std', 'span', 'none'],
                   'realign_every': [1, 0], 'merge_within': [0.0, 0.5, 1.0]},
        'kinetics': {'K_rule': ['median', 'majority', 'max'],
                     'scale_mode': ['segment_std', 'span', 'none'],
                     'realign_every': [1, 0],
                     'merge_within': [0.0, 0.1, 0.25]},
    },
    'GTM': {
        'fn': baseline_gtm,
        'medvid': {'K_rule': ['median', 'majority', 'max'],
                   'alpha': [1.0, 10.0, 50.0, 200.0, 1000.0],
                   'beta': [0.01, 0.1, 1.0, 10.0, 50.0],
                   'mu0': [0.0], 'sigma0': [0.5, 1.0, 3.0],
                   'scale_mode': ['segment_std', 'span', 'none'],
                   'realign_every': [1, 0], 'merge_within': [0.0, 0.5, 1.0]},
        'kinetics': {'K_rule': ['median', 'majority', 'max'],
                     'alpha': [1.0, 10.0, 50.0, 200.0],
                     'beta': [0.01, 0.1, 1.0, 10.0],
                     'mu0': [0.0], 'sigma0': [0.5, 1.0, 3.0],
                     'scale_mode': ['segment_std', 'span', 'none'],
                     'realign_every': [1, 0],
                     'merge_within': [0.0, 0.1, 0.25]},
    },
    'CATD': {
        'fn': baseline_catd,
        'medvid': {'K_rule': ['median', 'majority', 'max'],
                   'scale_mode': ['segment_std', 'span', 'none'],
                   'alpha_sig': [0.01, 0.05, 0.10, 0.20, 0.40],
                   'realign_every': [1, 0], 'merge_within': [0.0, 0.5, 1.0]},
        'kinetics': {'K_rule': ['median', 'majority', 'max'],
                     'scale_mode': ['segment_std', 'span', 'none'],
                     'alpha_sig': [0.01, 0.05, 0.10, 0.20, 0.40],
                     'realign_every': [1, 0],
                     'merge_within': [0.0, 0.1, 0.25]},
    },
    'KDEm': {
        'fn': baseline_kdem,
        'medvid': {'kernel': ['gaussian', 'ep', 'triweight'],
                   'bandwidth': [-1.0, 1.0, 2.0, 3.0, 4.0, 6.0],
                   'bandwidth_mult': [1.0, 2.0], 'cut': [0.0, 0.05, 0.10],
                   'merge_tol': [0.5, 1.0], 'min_annotators': [1, 2]},
        'kinetics': {'kernel': ['gaussian', 'ep', 'triweight'],
                     'bandwidth': [-1.0, 0.2, 0.4, 0.8],
                     'bandwidth_mult': [1.0, 2.0], 'cut': [0.0, 0.05, 0.10],
                     'merge_tol': [0.1, 0.25, 0.5], 'min_annotators': [1]},
    },
    'EvolvT': {
        'fn': baseline_evolvt_kalman,
        'medvid': {'K_rule': ['median', 'majority', 'max'],
                   'alpha': [10.0, 200.0, 1e3, 1e4, 1e5, 1e6],
                   'beta': [0.001, 0.01, 0.1, 1.0],
                   'learn_A': [True, False],
                   'init_sigma': [0.02, 0.05, 0.1, 0.2],
                   'realign_every': [1, 0], 'merge_within': [0.0, 0.5, 1.0]},
        'kinetics': {'K_rule': ['median', 'majority', 'max'],
                     'alpha': [1.0, 10.0, 200.0, 1e4, 1e6],
                     'beta': [0.001, 0.01, 0.1, 1.0],
                     'learn_A': [True, False],
                     'init_sigma': [0.02, 0.05, 0.1],
                     'realign_every': [1, 0],
                     'merge_within': [0.0, 0.1, 0.25]},
    },
    'DynaTD': {
        'fn': baseline_dynatd_kdd15,
        'medvid': {'K_rule': ['median', 'majority', 'max'],
                   'lam': [0.0, 0.1, 1.0, 10.0], 'gamma': [0.7, 0.9, 1.0],
                   'theta': [0.1, 1.0, 10.0, 1e2, 1e3],
                   'alpha': [2.0, 10.0, 200.0, 1e4, 1e6],
                   'beta': [0.001, 0.1, 1.0],
                   'realign_every': [1, 0], 'merge_within': [0.0, 0.5, 1.0]},
        'kinetics': {'K_rule': ['median', 'majority', 'max'],
                     'lam': [0.0, 0.1, 1.0, 10.0], 'gamma': [0.7, 0.9, 1.0],
                     'theta': [0.1, 1.0, 10.0, 1e2, 1e3],
                     'alpha': [2.0, 10.0, 200.0, 1e4],
                     'beta': [0.001, 0.1, 1.0], 'realign_every': [1, 0],
                     'merge_within': [0.0, 0.1, 0.25]},
    },
    'DynaTD (decay)': {
        'fn': baseline_dynatd_decay,
        'medvid': {'K_rule': ['median', 'majority', 'max'],
                   'gamma': [0.5, 0.8, 0.9, 0.95, 0.99, 1.0],
                   'theta': [1.0, 10.0, 1e2, 1e3, 1e4, 1e5],
                   'alpha': [2.0, 10.0, 200.0, 1e4, 1e6],
                   'beta': [0.001, 0.1, 1.0], 'n_passes': [1, 2],
                   'merge_within': [0.0, 0.5, 1.0]},
        'kinetics': {'K_rule': ['median', 'majority', 'max'],
                     'gamma': [0.5, 0.8, 0.9, 0.95, 0.99, 1.0],
                     'theta': [1.0, 10.0, 1e2, 1e3, 1e4],
                     'alpha': [2.0, 10.0, 200.0, 1e4],
                     'beta': [0.001, 0.1, 1.0], 'n_passes': [1, 2],
                     'merge_within': [0.0, 0.1, 0.25]},
    },
}

# Reported in the paper as "DynaTD ... with decay", a variant of the same
# method rather than a tenth baseline.
PAPER_ORDER = ['HDBSCAN', 'KDE+peaks', 'BSC', 'CRH', 'GTM', 'CATD', 'KDEm',
               'EvolvT', 'DynaTD', 'DynaTD (decay)', 'CTD-RR (ours)']
