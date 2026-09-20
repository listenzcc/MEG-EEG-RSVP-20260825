"""
File: remove-keypress-artificial.py
Author: Chuncheng Zhang
Date: 2026-08-26
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Remove keypress artificial from epochs.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-08-26 ------------------------
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
OUTPUT_DIR = DATA_DIR


# %% ---- 2026-08-26 ------------------------
# Function and class


# %% ---- 2026-08-26 ------------------------
# Play ground
if MODE == 'EEG':
    kwargs = dict(n_grad=0, n_mag=0, n_eeg=2)
elif MODE == 'MEG':
    kwargs = dict(n_grad=0, n_mag=3, n_eeg=0)

evoked1 = mne.read_evokeds(DATA_DIR / 'epochs-1-notch-ave.fif')[0]
evoked2 = mne.read_evokeds(DATA_DIR / 'epochs-2-notch-ave.fif')[0]
evoked3 = mne.read_evokeds(DATA_DIR / 'epochs-3-notch-ave.fif')[0]
proj2 = mne.compute_proj_evoked(evoked2, **kwargs)
proj3 = mne.compute_proj_evoked(evoked3, **kwargs)
proj = proj2 + proj3

# Report how much variance each projector explains in its source evoked
# Note: the projectors built from evoked2 (non-target) will also remove
# the standard visual response from the target epochs - what survives
# is the target-specific part, close to a spatially filtered difference wave.
for name, evk, prj in [('non-target(2)', evoked2, proj2),
                       ('keypress(3)', evoked3, proj3)]:
    # prj, not proj: each source evoked has to be reported on its own,
    # otherwise both lines print the total over all the projectors
    parts = [f"{p['desc']} {p['explained_var']:.3f}" for p in prj]
    v = np.sum([p['explained_var'] for p in prj])
    logger.info(
        f'Projector from {name} explains {v:.3f} of its variance: '
        f'{", ".join(parts)}')


# %% ---- 2026-08-26 ------------------------
# Target minus non-target, before the removal
# The 0.3s target-specific component should be checked here:
# if it exists in both, and gets cleaner after the removal, it is real.
diff = mne.combine_evoked([evoked1, evoked2], weights=[1, -1])
fname = OUTPUT_DIR / 'diff-1-2-notch-ave.fif'
diff.save(fname, overwrite=True)
fig = diff.plot_joint(title='target - non-target (before removal)',
                      show=False)
fig.savefig(fname.with_suffix('.png'))
plt.close(fig)
logger.debug(f'Saved into {fname}')


# %% ---- 2026-08-26 ------------------------
# Pending
for evt in ['1', '2', '3']:
    fname = DATA_DIR / f'epochs-{evt}-notch-epo.fif'
    epochs = mne.read_epochs(fname)

    epochs.add_proj(proj)
    epochs.apply_proj()

    evoked = epochs.average()

    fname = OUTPUT_DIR / f'epochs-{evt}-notch-removal-artificial-epo.fif'
    epochs.save(fname, overwrite=True)
    logger.debug(f'Saved into {fname}')

    fname = OUTPUT_DIR / f'epochs-{evt}-notch-removal-artificial-ave.fif'
    evoked.save(fname, overwrite=True)

    fig = evoked.plot_joint(title=f'{evt=}', show=False)
    fig.savefig(fname.with_suffix('.png'))
    plt.close(fig)

    logger.debug(f'Saved into {fname}')


# %% ---- 2026-08-26 ------------------------
# Target minus non-target, after the removal
evoked1 = mne.read_evokeds(
    DATA_DIR / 'epochs-1-notch-removal-artificial-ave.fif')[0]
evoked2 = mne.read_evokeds(
    DATA_DIR / 'epochs-2-notch-removal-artificial-ave.fif')[0]

diff = mne.combine_evoked([evoked1, evoked2], weights=[1, -1])
fname = OUTPUT_DIR / 'diff-1-2-notch-removal-artificial-ave.fif'
diff.save(fname, overwrite=True)
fig = diff.plot_joint(title='target - non-target (after removal)',
                      show=False)
fig.savefig(fname.with_suffix('.png'))
plt.close(fig)
logger.debug(f'Saved into {fname}')


# %% ---- 2026-08-26 ------------------------
# Pending
