"""
Fetch the anonymous-panel annotations.

Both corpora are third-party releases, annotation-only (no video is needed),
and are not redistributed here.  This script reports what is present, and
prints the sources for whatever is missing.

Run:  python scripts/download_kinetics.py [--fetch-gebd]
"""

import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctdrr.config import (KINETICS_DIR, GEBD_RAW_DIR,        # noqa: E402
                          GEBPLUS_RAW_DIR)

GEBD_FILES = ['k400_train_raw_annotation.pkl',
              'k400_val_raw_annotation.pkl']
GEBPLUS_FILES = ['train_all_annotation.json', 'val_all_annotation.json',
                 'test_all_annotation.json']
GEBD_DRIVE_FOLDER = '1AlPr63Q9D-HAGc5bOUNTzjCiWOC1a3xo'

NOTES = f"""
Expected layout
---------------
  {GEBD_RAW_DIR}/
      k400_train_raw_annotation.pkl
      k400_val_raw_annotation.pkl
  {GEBPLUS_RAW_DIR}/
      train_all_annotation.json
      val_all_annotation.json
      test_all_annotation.json

Kinetics-GEBD (raw per-annotator boundaries)
--------------------------------------------
  Released by the GEBD authors as a Google Drive folder:
      pip install gdown
      gdown --folder {GEBD_DRIVE_FOLDER} -O {KINETICS_DIR}
  The folder also contains HMDB, ADL and UTE annotations, which are not used
  here -- only the two k400_*.pkl files are needed.
  Or pass --fetch-gebd to run that command now.

Kinetics-GEB+ (raw per-annotator boundaries, "version b")
---------------------------------------------------------
  Use the Yuxuan-W/GEB-plus repository, NOT showlab/GEB-Plus: the latter's
  Drive links are dead.  The raw folder is
      dataset/version b (raw)/{{train,val,test}}_all_annotation.json
  linked from that repository's README.  Note the repository ALSO ships a
  filtered version under datasets/annotations/, which keeps one annotator per
  clip and cannot be used here -- the raw, multi-annotator release is
  required.

Licences are those of the respective corpora; check them before use.
"""


def status():
    ok = True
    for directory, files in ((GEBD_RAW_DIR, GEBD_FILES),
                             (GEBPLUS_RAW_DIR, GEBPLUS_FILES)):
        for name in files:
            p = directory / name
            have = p.exists()
            ok &= have
            size = f'{p.stat().st_size / 1e6:.0f} MB' if have else '--'
            print(f'  [{"x" if have else " "}] {p}  {size}')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch-gebd', action='store_true',
                    help='run the gdown command for Kinetics-GEBD')
    args = ap.parse_args()

    GEBD_RAW_DIR.mkdir(parents=True, exist_ok=True)
    GEBPLUS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print('Anonymous-panel annotations:')
    complete = status()

    if args.fetch_gebd:
        cmd = ['gdown', '--folder', GEBD_DRIVE_FOLDER, '-O', str(KINETICS_DIR)]
        print(f'\n$ {" ".join(cmd)}')
        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f'download failed ({e}); follow the notes below instead')
        print('\nAfter downloading, move the two k400_*.pkl files into '
              f'{GEBD_RAW_DIR}')
        complete = status()

    if complete:
        print('\nAll annotation files present -- '
              'scripts/run_kinetics.py is ready to run.')
    else:
        print(NOTES)
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
