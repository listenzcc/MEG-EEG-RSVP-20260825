"""
File: notch-epochs.py
Author: Chuncheng Zhang
Date: 2026-08-26
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot ERP.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-08-26 ------------------------
# Requirements and constants
from scipy.signal import butter, filtfilt
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
OUTPUT_DIR = DATA_DIR

# %% ---- 2026-08-26 ------------------------
# Function and class


def notch_output(notch_freq: float, data: np.array, fs: int, band_width: float = 2):
    low = notch_freq - band_width
    high = notch_freq + band_width

    b, a = butter(
        N=4,
        Wn=[low, high],
        btype='bandstop',
        fs=fs
    )

    data_filtered = filtfilt(b, a, data, axis=-1)

    return data_filtered


# %% ---- 2026-08-26 ------------------------
# Play ground

for evt in ['1', '2', '3']:
    fname = DATA_DIR / f'epochs-{evt}-epo.fif'
    epochs = mne.read_epochs(fname)
    fs = epochs.info['sfreq']

    # data shape is (n_trials, n_channels, n_times)
    data = epochs.get_data()

    # Notch out 10 and 20 Hz signal
    data = notch_output(10, data, fs)
    data = notch_output(20, data, fs)

    epochs = mne.EpochsArray(data, epochs.info,
                             epochs.events, epochs.times[0], epochs.event_id)
    evoked = epochs.average()

    fname = OUTPUT_DIR / f'epochs-{evt}-notch-epo.fif'
    epochs.save(fname, overwrite=True)
    logger.debug(f'Saved into {fname}')

    fname = OUTPUT_DIR / f'epochs-{evt}-notch-ave.fif'
    evoked.save(fname, overwrite=True)

    fig = evoked.plot_joint(title=f'{evt=}', show=False)
    fig.savefig(fname.with_suffix('.png'))
    plt.close(fig)

    logger.debug(f'Saved into {fname}')


# %% ---- 2026-08-26 ------------------------
# Pending


# %% ---- 2026-08-26 ------------------------
# Pending
