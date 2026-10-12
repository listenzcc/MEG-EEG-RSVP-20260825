"""
Build a synthetic sandbox for keypress-trial-audit.py, with a known answer.

A 10 Hz image grid runs from 0.2 s to 8.0 s (40 images, four of them targets),
and five keypresses sit inside it:

    1.35  0.35 s after the target at 1.0     -> keypress with a target in window
    1.45  0.45 s after the target at 1.0     -> keypress with a target in window
    3.40  0.40 s after the target at 3.0     -> keypress with a target in window
    4.55  no target in [3.75, 4.40]          -> keypress WITHOUT a target
    7.50  0.50 s after the target at 7.0     -> keypress with a target in window

Expected, with the default --rt-window 0.15,0.8:
    targets 4, nontargets 36, keypresses 5
    kp_with_target_in_window     4
    kp_without_target_in_window  1
    target_with_keypress         3   (the target at 5.0 gets none)
    target_without_keypress      1
    delay_median_to_target_ms    425 (= (350+450+400+500)/4)
    target_ratio                 0.1
    keypress_per_target          1.25

Run from the sandbox root:
    python make_synthetic.py
"""
from pathlib import Path
import numpy as np
import mne

SFREQ = 200.0
CH = [f'EEG{i:03d}' for i in range(1, 9)]
OUT = Path('rawData/RSVP_dataset/processed_data/EEG_S01')
OUT.mkdir(parents=True, exist_ok=True)

def make_block(path, duration, events):
    info = mne.create_info(CH, SFREQ, 'eeg')
    raw = mne.io.RawArray(np.zeros((len(CH), int(duration * SFREQ))), info,
                          verbose='ERROR')
    raw.set_annotations(mne.Annotations(
        [t for t, _ in events], 0.01, [str(c) for _, c in events]))
    raw.save(path, overwrite=True)
    print(f'{path}  {len(events)} events')

# Block 1 is skipped by the reader, so give it a decoy event only
make_block(OUT / 'block_01_ica-raw.fif', 2.0, [(0.5, 2)])

# Block 2 carries the audit scenario
events = []
targets = {1.0, 3.0, 5.0, 7.0}
for t in np.round(np.arange(0.2, 8.001, 0.2), 3):
    events.append((float(t), 1 if float(t) in targets else 2))
for t in [1.35, 1.45, 3.40, 4.55, 7.50]:
    events.append((t, 3))

events.sort()
make_block(OUT / 'block_02_ica-raw.fif', 10.0, events)

print('done')
