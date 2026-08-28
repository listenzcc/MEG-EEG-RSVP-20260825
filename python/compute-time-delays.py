"""
File: compute-time-delays.py
Author: Chuncheng Zhang
Date: 2026-08-28
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis the time delays across conditions.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-06-01 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
RAW_DIR = Path('./rawData/RSVP_dataset/processed_data')
DATA_DIR = Path('./output/epochs')
OUTPUT_DIR = Path('./output/timeDelays')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%
subjects = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
print(subjects)

# %% ---- 2026-06-01 ------------------------
# Function and class


def get_diff_delay(events):
    fs = 1200  # Hz
    delays = []
    for i in range(0, len(events)-1):
        if events[i][2] == 1 and events[i+1][2] == 3:
            delay = (events[i+1][0] - events[i][0]) / fs
            delays.append(delay)
    return delays


def get_diff_delay2(times3, times1):
    fs = 1200  # Hz
    delays = []
    for i, t3 in enumerate(times3):
        times1_before_t3 = [t1 for t1 in times1 if t1 < t3]
        if len(times1_before_t3) == 0:
            continue  # No target event before this keypress, skip
        closest_t1 = min(times1_before_t3, key=lambda t1: t3 - t1)
        delay = (t3 - closest_t1) / fs
        delays.append((i, delay))
    return delays


def get_diff_delay_target(times1, times3):
    fs = 1200  # Hz
    delays = []
    for i, t1 in enumerate(times1):
        times3_after_t1 = [t3 for t3 in times3 if t1 < t3]
        if len(times3_after_t1) == 0:
            continue  # No keypress event after this target, skip
        closest_t3 = min(times3_after_t1, key=lambda t3: t3 - t1)
        delay = (closest_t3 - t1) / fs
        delays.append((i, delay))
    return delays


# %% ---- 2026-06-01 ------------------------
# Play ground
dfs_keypress = []
dfs_target = []
for subject in tqdm(subjects):
    mode, subj = subject.split('-')

    # Find first 1 and 3
    folder = RAW_DIR / subject.replace('-', '_')
    found = sorted(folder.glob('block_*_ica-raw.fif'))[1:]
    raw = mne.io.read_raw_fif(found[0], preload=False)
    raw_events, event_id = mne.events_from_annotations(raw)
    first_1 = next(e for e in raw_events if e[-1] == 1)
    first_3 = next(e for e in raw_events if e[-1] == 3)
    print(subject, first_1, first_3)

    # Target
    epochs1 = mne.read_epochs(DATA_DIR / subject / 'epochs-1-epo.fif')
    # Keypress
    epochs3 = mne.read_epochs(DATA_DIR / subject / 'epochs-3-epo.fif')

    epochs1.events[:, 0] -= epochs1.events[0, 0]
    epochs1.events[:, 0] += first_1[0]

    epochs3.events[:, 0] -= epochs3.events[0, 0]
    epochs3.events[:, 0] += first_3[0]

    # epochs = mne.concatenate_epochs([epochs1, epochs3], add_offset=False)
    # epochs.events = sorted(epochs.events, key=lambda x: x[0])
    # events = epochs.events
    # delays = get_diff_delay(events)

    delays = get_diff_delay2(epochs3.events[:, 0], epochs1.events[:, 0])
    df = pd.DataFrame()
    df['delay'] = [e[1] for e in delays]
    df['index'] = [e[0] for e in delays]
    df['subject'] = subj
    df['mode'] = mode
    dfs_keypress.append(df)

    delays = get_diff_delay_target(epochs1.events[:, 0], epochs3.events[:, 0])
    df = pd.DataFrame()
    df['delay'] = [e[1] for e in delays]
    df['index'] = [e[0] for e in delays]
    df['subject'] = subj
    df['mode'] = mode
    dfs_target.append(df)


df_keypress = pd.concat(dfs_keypress, ignore_index=True)
df_keypress.to_csv(OUTPUT_DIR / 'timeDelays-keypress.csv', index=False)
display(df_keypress)

df_target = pd.concat(dfs_target, ignore_index=True)
df_target.to_csv(OUTPUT_DIR / 'timeDelays-target.csv', index=False)
display(df_target)

# %%
df = df_keypress.copy()
df = df[df['delay'] < 1]  # Filter out delays greater than 1 second
sns.histplot(df, x='delay', hue='mode', element='step',
             stat='density', legend=True)
plt.xlabel('Delay (s)')
plt.title('Distribution of Delays between Target and Keypress Events')
plt.savefig(OUTPUT_DIR / 'timeDelays-keypress.png')
plt.close()
# plt.show()

df = df_target.copy()
df = df[df['delay'] < 1]  # Filter out delays greater than 1 second
sns.histplot(df, x='delay', hue='mode', element='step',
             stat='density', legend=True)
plt.xlabel('Delay (s)')
plt.title('Distribution of Delays between Target and Keypress Events')
plt.savefig(OUTPUT_DIR / 'timeDelays-target.png')
plt.close()
# plt.show()

# %%

# %%

# %% ---- 2026-06-01 ------------------------
# Pending


# %% ---- 2026-06-01 ------------------------
# Pending

# %%

# %%
