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

OUTPUT_DIR = Path(f'output/ERP/{MODE}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% ---- 2026-08-26 ------------------------
# Function and class


# %% ---- 2026-08-26 ------------------------
# Play ground

for evt, flag_removal_artificial in product(['1', '2', '3'], [True, False]):

    if flag_removal_artificial:
        pattern = f'{MODE}-S*/epochs-{evt}-notch-removal-artificial-ave.fif'
        fname = OUTPUT_DIR / f'{evt}-notch-removal-artificial-ave.png'
    else:
        pattern = f'{MODE}-S*/epochs-{evt}-notch-ave.fif'
        fname = OUTPUT_DIR / f'{evt}-notch-ave.png'

    logger.debug(f'Find files by {pattern=}')
    files = sorted(DATA_DIR.rglob(pattern))
    logger.debug(f'{files=}')

    evokeds = []
    numbers = []
    for _fname in files:
        evokeds += mne.read_evokeds(_fname)
        evoked = evokeds[0]
        numbers.append(evoked.nave)
    total = np.sum(numbers)

    data = []
    for n, evoked in zip(numbers, evokeds):
        data.append(evoked.data * n / total)

    evoked.data = np.sum(data, axis=0)

    fig = evoked.plot_joint(title=f'{evt=}', show=False)
    fig.savefig(fname)
    plt.close(fig)
    logger.debug(f'Saved into {fname}')

# %% ---- 2026-08-26 ------------------------
# Pending


# %% ---- 2026-08-26 ------------------------
# Pending
