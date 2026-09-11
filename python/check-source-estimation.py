"""
File: check-source-estimation.py
Author: Chuncheng Zhang
Date: 2026-09-11
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Check the source estimation results for MODE

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

args = parser.parse_args()
MODE = args.mode
EPOCHS_FNAME = args.epochs_fname

logger.info(f'Start with {args=}')

# %%
DATA_DIR = Path('./output/source-estimation')

# %% ---- 2026-09-11 ------------------------
# Function and class


# %% ---- 2026-09-11 ------------------------
# Play ground
stc_files = sorted(DATA_DIR.rglob(f'{MODE}-*/{EPOCHS_FNAME}.ave.stc-lh.stc'))

stcs = [mne.read_source_estimate(p.parent / p.name.replace('-lh.stc', ''))
        for p in stc_files]

print(stc_files)
print(stcs)

# data shape is (n_stcs, n_verts, n_times)
data = np.array([stc.data for stc in stcs])
mean_data = np.mean([stc.data for stc in stcs], axis=0)

mean = data.mean(axis=0, keepdims=True)   # (1, n_verts, n_times)
std = data.std(axis=0, keepdims=True)    # (1, n_verts, n_times)

mean = data.mean()
std = data.std()

zscore = (data - mean) / (std + 0*1e-8)
zscore = np.mean(zscore, axis=0)

stc = stcs[0]
stc.data = zscore
# stc.data = mean_data
stc.subject = 'fsaverage'

stc.plot(hemi='both')
input('Press enter to escape.')
exit(0)


# %% ---- 2026-09-11 ------------------------
# Pending


# %% ---- 2026-09-11 ------------------------
# Pending
