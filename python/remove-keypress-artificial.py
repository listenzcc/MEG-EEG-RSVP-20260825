"""
File: remove-keypress-artificial.py
Author: Chuncheng Zhang
Date: 2026-08-26
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Remove keypress artificial from epochs.

    The projection basis is the union of two independently estimated sets:
      proj2 <- non-target average  (standard visual response subspace)
      proj3 <- keypress average    (movement-related subspace)
    --proj-source selects which of them is applied. 'both' is the original
    behaviour; 'keypress' and 'visual' split the pair so that the two can be
    compared against each other and against the unprojected data. Because both
    sets are estimated with identical kwargs on data from the same sessions,
    the single-source conditions are matched in dimension and differ only in
    whether they carry movement-related activity.

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
parser.add_argument('--proj-source', '--proj_source', dest='proj_source',
                    default='both', choices=['both', 'keypress', 'visual'],
                    help='Which projector set to apply: '
                         'both (default, reproduces the original outputs) | '
                         'keypress (projectors from the keypress average only) | '
                         'visual (projectors from the non-target average only). '
                         'Splitting the two sets gives a control: they come from '
                         'the same data, same number of components and same trials, '
                         'and differ only in carrying movement-related activity.')
args = parser.parse_args()
SUBJ = args.subject
MODE = args.mode
PROJ_SOURCE = args.proj_source

# 'both' keeps the historical 'artificial' tag so that the original file names
# and their contents stay byte-for-byte unchanged.
PROJ_TAG = dict(both='artificial', keypress='keypress', visual='visual')[PROJ_SOURCE]

logger.info(f'Start with {SUBJ=}, {MODE=}, {PROJ_SOURCE=}, {PROJ_TAG=}')

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

# Which set of projectors to actually apply.
#   both     : proj2 + proj3 (the original behaviour)
#   keypress : movement-related subspace only
#   visual   : standard visual response subspace only
# proj2 and proj3 are built with identical kwargs, so the two single-source
# conditions carry the same number of components and the same trials; the
# difference between them is the presence of movement-related activity.
if PROJ_SOURCE == 'both':
    proj = proj2 + proj3
elif PROJ_SOURCE == 'keypress':
    proj = proj3
else:
    proj = proj2

assert len(proj) > 0, f'No projector computed for {PROJ_SOURCE=}'
logger.info(
    f'Applying {len(proj)} projectors from {PROJ_SOURCE=} '
    f'(proj2={len(proj2)}, proj3={len(proj3)})')

# Report how much variance each projector explains in its source evoked
# Note: the projectors built from evoked2 (non-target) will also remove
# the standard visual response from the target epochs - what survives
# is the target-specific part, close to a spatially filtered difference wave.
proj_rows = []
for name, evk, prj in [('non-target(2)', evoked2, proj2),
                       ('keypress(3)', evoked3, proj3)]:
    # prj, not proj: each source evoked has to be reported on its own,
    # otherwise both lines print the total over all the projectors
    parts = [f"{p['desc']} {p['explained_var']:.3f}" for p in prj]
    v = np.sum([p['explained_var'] for p in prj])
    logger.info(
        f'Projector from {name} explains {v:.3f} of its variance: '
        f'{", ".join(parts)}')

    for p in prj:
        proj_rows.append(dict(
            mode=MODE, subject=SUBJ, source=name,
            desc=p['desc'], explained_var=p['explained_var']))
    proj_rows.append(dict(
        mode=MODE, subject=SUBJ, source=name,
        desc='TOTAL', explained_var=v))

# Every source has to report on its own; the TOTAL row is the sum over the
# projectors of that single source, ready for a group-level summary.
fname = OUTPUT_DIR / f'proj-explained-var-{PROJ_TAG}.csv'
pd.DataFrame(proj_rows).to_csv(fname, index=False, float_format='%.6g')
logger.debug(f'Saved into {fname}')


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

    fname = OUTPUT_DIR / f'epochs-{evt}-notch-removal-{PROJ_TAG}-epo.fif'
    epochs.save(fname, overwrite=True)
    logger.debug(f'Saved into {fname}')

    fname = OUTPUT_DIR / f'epochs-{evt}-notch-removal-{PROJ_TAG}-ave.fif'
    evoked.save(fname, overwrite=True)

    fig = evoked.plot_joint(title=f'{evt=}', show=False)
    fig.savefig(fname.with_suffix('.png'))
    plt.close(fig)

    logger.debug(f'Saved into {fname}')


# %% ---- 2026-08-26 ------------------------
# Target minus non-target, after the removal
evoked1 = mne.read_evokeds(
    DATA_DIR / f'epochs-1-notch-removal-{PROJ_TAG}-ave.fif')[0]
evoked2 = mne.read_evokeds(
    DATA_DIR / f'epochs-2-notch-removal-{PROJ_TAG}-ave.fif')[0]

diff = mne.combine_evoked([evoked1, evoked2], weights=[1, -1])
fname = OUTPUT_DIR / f'diff-1-2-notch-removal-{PROJ_TAG}-ave.fif'
diff.save(fname, overwrite=True)
fig = diff.plot_joint(title='target - non-target (after removal)',
                      show=False)
fig.savefig(fname.with_suffix('.png'))
plt.close(fig)
logger.debug(f'Saved into {fname}')


# %% ---- 2026-08-26 ------------------------
# Pending
