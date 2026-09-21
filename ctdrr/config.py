"""
Where the data lives.

Defaults are repo-relative, so a fresh clone works with no configuration.
Override with the environment variables CTDRR_DATA and CTDRR_RESULTS, e.g.

    CTDRR_DATA=/mnt/big/ctdrr-data python scripts/run_kinetics.py
"""

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent

DATA_DIR = Path(os.environ.get('CTDRR_DATA', REPO_ROOT / 'data'))
RESULTS_DIR = Path(os.environ.get('CTDRR_RESULTS', REPO_ROOT / 'results'))

# --- MedVid: the fixed, ordered panel (shipped with the repo) -------------
MEDVID_DIR = DATA_DIR / 'medvid'
MEDVID_ANNOTATORS = [MEDVID_DIR / f'annotator_{j}.json' for j in (1, 2, 3)]
MEDVID_REFERENCE = MEDVID_DIR / 'reference_tierA.json'

# --- Kinetics: the anonymous crowd panels (fetched, not redistributed) ----
KINETICS_DIR = DATA_DIR / 'kinetics'
GEBD_RAW_DIR = KINETICS_DIR / 'gebd_raw'
GEBPLUS_RAW_DIR = KINETICS_DIR / 'gebplus_raw'
GEBPLUS_FILTERED_DIR = KINETICS_DIR / 'gebplus_filtered'

MISSING_KINETICS = (
    "{path} not found.\n"
    "The Kinetics-GEBD and GEB+ annotations are third-party releases and are "
    "not redistributed here.\nRun  python scripts/download_kinetics.py  for "
    "the download instructions and expected layout."
)


def require(path, kinetics=False):
    """Return `path` as a string, with an actionable error if it is missing."""
    p = Path(path)
    if not p.exists():
        msg = (MISSING_KINETICS.format(path=p) if kinetics
               else f'{p} not found.')
        raise FileNotFoundError(msg)
    return str(p)
