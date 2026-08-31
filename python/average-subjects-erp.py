"""
File: average-subjects-erp.py
Author: Chuncheng Zhang
Date: 2026-08-26
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    ERP average across subjects.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-08-26 ------------------------
# Requirements and constants
from itertools import product
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Require SUBJ and MODE parameters')
parser.add_argument('-m', '--mode', default='EEG', help='Mode name EEG | MEG')
args = parser.parse_args()
MODE = args.mode

logger.info(f'Start with {MODE=}')

# %%
DATA_DIR = Path(f'output/epochs')
TD_DIR = Path('./output/timeDelays')

OUTPUT_DIR = Path(f'output/ERP/{MODE}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% ---- 2026-08-26 ------------------------
# Function and class


# %% ---- 2026-08-26 ------------------------
# Play ground
target_df = pd.read_csv(TD_DIR / 'timeDelays-target.csv')
keypress_df = pd.read_csv(TD_DIR / 'timeDelays-keypress.csv')

for evt, flag_removal_artificial, td in product(
    ['1', '2', '3'],
    [True, False],
    ['quick', 'slow', 'all']
):

    if evt == '2':
        td = 'all'

    if flag_removal_artificial:
        pattern = f'{MODE}-S*/epochs-{evt}-notch-removal-artificial-ave.fif'
        img_fname = OUTPUT_DIR / f'{td}-{evt}-notch-removal-artificial-ave.png'
    else:
        pattern = f'{MODE}-S*/epochs-{evt}-notch-ave.fif'
        img_fname = OUTPUT_DIR / f'{td}-{evt}-notch-ave.png'

    if img_fname.exists():
        continue

    logger.debug(f'Find files by {pattern=}')
    files = sorted(DATA_DIR.rglob(pattern))
    logger.debug(f'{files=}')

    evokeds = []
    numbers = []
    for fname in files:
        mode, subj = fname.parent.name.split('-')
        logger.debug(f'{mode=}, {subj=}')

        if flag_removal_artificial:
            _fname = f'{mode}-{subj}/epochs-{evt}-notch-removal-artificial-epo.fif'
        else:
            _fname = f'{mode}-{subj}/epochs-{evt}-notch-epo.fif'

        epochs = mne.read_epochs(DATA_DIR / _fname)
        print(epochs)

        # evokeds += mne.read_evokeds(fname)
        # evoked = evokeds[0]

        # Fetch the table
        td_threshold = 0.38
        queries = [f'mode=="{mode}"', f'subject=="{subj}"']
        if evt == '1':
            df = target_df.query(' & '.join(queries))
        elif evt == '3':
            df = keypress_df.query(' & '.join(queries))

        # Separate the quick and slow epochs
        if evt != '2':
            if td == 'quick':
                df = df.query(f'delay < {td_threshold}')
            elif td == 'slow':
                df = df.query(f'delay >= {td_threshold}')

        # Apply the mask
        if evt != '2':
            epochs = epochs[df['index']]
        else:
            epochs = epochs

        evoked = epochs.average()
        evokeds.append(evoked)

        numbers.append(evoked.nave)

    total = np.sum(numbers)

    data = []
    for n, evoked in zip(numbers, evokeds):
        data.append(evoked.data * n / total)

    evoked.data = np.sum(data, axis=0)

    fig = evoked.plot_joint(title=f'{evt=}', show=False)
    fig.savefig(img_fname)
    plt.close(fig)
    logger.debug(f'Saved into {img_fname}')

# %% ---- 2026-08-26 ------------------------
# Pending


# %% ---- 2026-08-26 ------------------------
# Pending
