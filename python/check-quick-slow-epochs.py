"""
File: check-quick-slow-epochs.py
Author: Chuncheng Zhang
Date: 2026-08-28
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Separate the quick and slow epochs.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-08-28 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Require SUBJ and MODE parameters')
parser.add_argument('-s', '--subject', default='S02',
                    help='Subject name like S02')
parser.add_argument('-m', '--mode', default='EEG', help='Mode name EEG | MEG')
args = parser.parse_args()
SUBJ = args.subject
MODE = args.mode

logger.info(f'Start with {SUBJ=}, {MODE=}')

# %%
DATA_DIR = Path(f'output/epochs/{MODE}-{SUBJ}')

TD_DIR = Path('./output/timeDelays')

# %% ---- 2026-08-28 ------------------------
# Function and class


# %% ---- 2026-08-28 ------------------------
# Play ground
fname = DATA_DIR / 'epochs-1-notch-epo.fif'
epochs = mne.read_epochs(fname)
print(epochs)

df = pd.read_csv(TD_DIR / 'timeDelays-target.csv')
queries = [f'mode=="{MODE}"', f'subject=="{SUBJ}"']
df = df.query(' & '.join(queries))
df.index = range(len(df))
display(df)

# %% ---- 2026-08-28 ------------------------
# Pending


# %% ---- 2026-08-28 ------------------------
# Pending
