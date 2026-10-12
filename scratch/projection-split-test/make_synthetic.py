"""
Build a synthetic sandbox for projection-source-compare.py, with a known answer.

Four conditions are written per subject, all carrying a "0.3 s" component and a
"0.4 s" component. The 0.4 s component is kept in 'none' and 'visual', and
removed in 'keypress' and 'both'. So the group summary must show

    none      ~ largest 0.4 s peak
    visual    slightly below none
    keypress  collapsed
    both      collapsed

Run from the sandbox root:
    python make_synthetic.py
"""
from pathlib import Path
import numpy as np
import mne

SFREQ = 200.0
TMIN, TMAX = -0.5, 1.5
N_EPOCHS = 24
CH_NAMES = [f'EEG{i:03d}' for i in range(1, 9)]

SUBJECTS = ['S01', 'S02', 'S03']
CONDITIONS = [
    ('epochs-1-notch-epo.fif', 1.00),
    ('epochs-1-notch-removal-visual-epo.fif', 0.92),
    ('epochs-1-notch-removal-keypress-epo.fif', 0.08),
    ('epochs-1-notch-removal-artificial-epo.fif', 0.06),
]

rng = np.random.default_rng(20261012)
OUT = Path('output/epochs')

info = mne.create_info(CH_NAMES, SFREQ, 'eeg')
times = np.arange(TMIN, TMAX, 1 / SFREQ)
n_times = len(times)

# Two spatial patterns that do not overlap in space: an early one on the
# posterior channels, and a late one on the anterior half, as a crude stand-in
# for the target response and the movement-related deflection.
#
# Both bumps have to survive the 0.1-5 Hz display filter that the analysis
# applies, so keep them narrow in frequency (wide in time) and put the late one
# well inside the 0.30-0.55 s reading window.
pat_early = np.zeros(len(CH_NAMES))
pat_early[4:] = 1.0
pat_late = np.zeros(len(CH_NAMES))
pat_late[:4] = 1.0

def bump(t, center, width):
    return np.exp(-((times - center) / width) ** 2)

early = bump(times, 0.18, 0.060) * 0.3
late = bump(times, 0.45, 0.060)

for subj in SUBJECTS:
    subj_dir = OUT / f'EEG-{subj}'
    subj_dir.mkdir(parents=True, exist_ok=True)

    # Per-subject scaling, so that the group average is not degenerate
    scale = 1.0 + 0.15 * rng.standard_normal()
    noise = 0.25 * rng.standard_normal((N_EPOCHS, len(CH_NAMES), n_times))

    for fname, late_gain in CONDITIONS:
        data = noise.copy()
        data += 1.0 * scale * np.outer(pat_early, early)[None, :, :]
        data += late_gain * scale * np.outer(pat_late, late)[None, :, :]

        epochs = mne.EpochsArray(data, info, tmin=TMIN, verbose='ERROR')
        path = subj_dir / fname
        epochs.save(path, overwrite=True)
        print(f'{path}  nave={len(epochs)}  late_gain={late_gain}')

print('done')
