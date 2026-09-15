"""
File: check-source-estimation.py
Author: Chuncheng Zhang
Date: 2026-09-11
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Check the source estimation results for MODE.

    It reads the stc files of all the subjects, z-scores every subject over
    the vertices and the times, averages them into a group map, reports the
    peak latency, and opens the interactive surface plot.

    With --png it renders the group map at the peak time into a PNG file
    instead of opening the interactive window, useful on a headless run.

Usage:
    python python/check-source-estimation.py -m MEG -t ave -e epochs-1-notch-epo.fif
    python python/check-source-estimation.py -m MEG -t ave -e epochs-1-notch-removal-artificial-epo.fif --png

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-09-11 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Require SUBJ and MODE parameters')
parser.add_argument('-m', '--mode', default='MEG', help='Mode name EEG | MEG')
parser.add_argument('-e', '--epochs_fname', default='epochs-1-notch-epo.fif',
                    help='Epochs fname, it should be inside the $DATA_DIR')
parser.add_argument('-t', '--tag', default='ave',
                    help='ave | ssvep10-evoked | ssvep10-power')
parser.add_argument('-s', '--subject', default='*',
                    help='Subject name like S02, all subjects by default')
parser.add_argument('--png', action='store_true',
                    help='Render the peak time into a PNG file instead of '
                         'the interactive window')

args = parser.parse_args()
MODE = args.mode
EPOCHS_FNAME = args.epochs_fname
TAG = args.tag
SUBJ = args.subject

logger.info(f'Start with {args=}')

# %%
DATA_DIR = Path('./output/source-estimation')
OUTPUT_DIR = DATA_DIR

# %% ---- 2026-09-11 ------------------------
# Function and class


# %% ---- 2026-09-11 ------------------------
# Play ground
stc_files = sorted(DATA_DIR.rglob(
    f'{MODE}-{SUBJ}/{EPOCHS_FNAME}.{TAG}.stc-lh.stc'))

assert len(stc_files) > 0, \
    f'No stc file is found, {DATA_DIR=}, {MODE=}, {EPOCHS_FNAME=}, {TAG=}'

stcs = [mne.read_source_estimate(p.parent / p.name.replace('-lh.stc', ''))
        for p in stc_files]

logger.info(f'Loaded {len(stcs)} stc files, {EPOCHS_FNAME=}, {TAG=}')

# data shape is (n_stcs, n_verts, n_times)
data = np.array([stc.data for stc in stcs])

# Z-score every subject over the vertices and the times, so the subjects
# are comparable, then average them into the group map
mean = data.mean()
std = data.std()
zscore = (data - mean) / (std + 1e-8)
group = zscore.mean(axis=0)

stc = stcs[0]
stc.data = group
stc.subject = 'fsaverage'

# The peak latency of the group |activation|, and the peak vertex
times = stc.times
peak_time = times[int(np.argmax(np.abs(group).max(axis=0)))]
vertex = int(np.argmax(np.abs(group[:, int(np.argmin(np.abs(times - peak_time)))])))
logger.info(
    f'Group map of {len(stcs)} subjects, {len(group)} vertices, '
    f'peak time {peak_time:.3f} s, peak vertex {vertex}')

# Also save the group map, so it can be re-plotted without the stc files
if SUBJ == '*':
    fname = OUTPUT_DIR / f'group-{MODE}-{EPOCHS_FNAME}.{TAG}.stc'
    stc.save(fname, overwrite=True)
    logger.info(f'Saved the group map into {fname}')

if args.png:
    import matplotlib
    matplotlib.use('Agg')
    fname = OUTPUT_DIR / f'group-{MODE}-{EPOCHS_FNAME}.{TAG}-peak{peak_time:.3f}s.png'
    brain = stc.plot(hemi='both', initial_time=peak_time, time_viewer=False,
                     show=False, subjects_dir=None)
    brain.save_image(fname)
    logger.info(f'Saved the snapshot into {fname}')
else:
    stc.plot(hemi='both')
    input('Press enter to escape.')


# %% ---- 2026-09-11 ------------------------
# Pending


# %% ---- 2026-09-11 ------------------------
# Pending
