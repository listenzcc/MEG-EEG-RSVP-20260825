"""
File: keypress-trial-audit.py
Author: Chuncheng Zhang
Date: 2026-10-12
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Audit the relation between keypress events and stimulus events, at the raw
    annotation level, before any epoching or rejection.

    Why this matters: if a keypress ever follows a non-target image, those
    trials form an internal control that needs no projection at all - a
    condition with a keypress but without a target. A target-specific
    component cannot be present there; a movement-related one can.

    For every keypress, the script finds the nearest preceding image event and
    reports whether that image was a target or a non-target, plus the delay.
    It also counts, for every target image, whether a keypress followed it.

    The output is one row per subject. Run with --summarize afterwards to fold
    them into a single table.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-10-12 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Audit keypress versus stimulus events at the raw level')
parser.add_argument('-s', '--subject', default='S01',
                    help='Subject name like S02, ignored with --summarize')
parser.add_argument('-m', '--mode', default='EEG', help='Mode name EEG | MEG')
parser.add_argument('--summarize', action='store_true',
                    help='Fold the per-subject csv into summary-{MODE}.csv '
                         'instead of reading raw data')
parser.add_argument('--rt-window', '--rt_window', dest='rt_window',
                    default='0.15,0.8',
                    help='Window (s) before a keypress in which its trigger '
                         'image may sit, comma separated lo,hi. Images in a '
                         '10 Hz stream are 100 ms apart while the response '
                         'lands 200-800 ms after the trigger, so the image '
                         'right before a keypress is usually NOT the trigger: '
                         'the whole window has to be searched.')
parser.add_argument('--target-code', '--target_code', dest='target_code',
                    default='1', help='Annotation label of the target image')
parser.add_argument('--nontarget-code', '--nontarget_code', dest='nontarget_code',
                    default='2', help='Annotation label of the non-target image')
parser.add_argument('--keypress-code', '--keypress_code', dest='keypress_code',
                    default='3', help='Annotation label of the keypress')
args = parser.parse_args()
SUBJ = args.subject
MODE = args.mode
RT_LO, RT_HI = [float(e) for e in args.rt_window.split(',')]

# %%
OUTPUT_DIR = Path('output/keypress-trial-audit')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %% ---- 2026-10-12 ------------------------
# Function and class
def read_raw_events(path: Path):
    '''
    Read one raw file and return it without loading the data or picking
    channels. Only the annotations are needed here, and keeping every channel
    avoids any chance of the picking step altering the event stream.

    Args:
        path: Path to the raw data file

    Returns:
        raw: MNE Raw object, not preloaded
    '''
    return mne.io.read_raw_fif(path, preload=False)


def load_raw(RAW_DIR: Path):
    '''
    Concatenate the ICA-cleaned blocks of one subject, skipping the first
    block, exactly as python/raw-to-epochs.py does.

    Args:
        RAW_DIR: directory holding the block_*_ica-raw.fif files

    Returns:
        raw: MNE Raw object with the annotations of all blocks
    '''
    files = sorted(RAW_DIR.glob('block_*_ica-raw.fif'))
    assert len(files) > 1, f'Not enough blocks in {RAW_DIR}: {files}'
    logger.debug(f'{len(files)} blocks, skipping the first one')

    raws = [read_raw_events(e) for e in files[1:]]
    raw = mne.concatenate_raws(raws, on_mismatch='ignore', verbose='ERROR')
    return raw


def classify_events(events, codes, sfreq, rt_window, logger_=logger):
    '''
    Relate keypress events to the image events that could have triggered them.

    In a 10 Hz stream the images are 100 ms apart while a keypress lands
    200-800 ms after its trigger, so the image immediately before a keypress
    is usually not the trigger. Attribution therefore searches the whole
    [lo, hi] window: if any target image sits inside it the keypress may
    belong to that target, and if no target was on screen in that window the
    keypress cannot be a target response. The latter count is a conservative
    lower bound, because a stray keypress can accidentally land inside the
    window of a target.

    Args:
        events: numpy array of shape (n_events, 3), [sample, 0, code]
        codes: dict with target / nontarget / keypress integer codes
        sfreq: sampling rate of the raw data, in Hz
        rt_window: (lo, hi) in seconds, the window searched before a keypress

    Returns:
        dict of counts, plus delay distributions in ms
    '''
    known = sorted(set(codes.values()))
    counts = dict(
        targets=int((events[:, 2] == codes['target']).sum()),
        nontargets=int((events[:, 2] == codes['nontarget']).sum()),
        keypresses=int((events[:, 2] == codes['keypress']).sum()),
        other_events=int((~np.isin(events[:, 2], known)).sum()))

    if counts['other_events']:
        logger_.info(
            f'{counts["other_events"]} events carry a code outside {known}')

    tg = np.sort(events[events[:, 2] == codes['target']][:, 0])
    nt = np.sort(events[events[:, 2] == codes['nontarget']][:, 0])
    kp = np.sort(events[events[:, 2] == codes['keypress']][:, 0])

    out = dict(counts)

    if len(kp) == 0 or len(tg) + len(nt) == 0:
        logger_.warning('No keypress or no image event, nothing to relate')
        return out

    rt_lo, rt_hi = rt_window
    n_lo, n_hi = int(round(rt_lo * sfreq)), int(round(rt_hi * sfreq))

    # keypress -> is there a target in [kp - hi, kp - lo]?
    left = np.searchsorted(tg, kp - n_hi, side='left')
    right = np.searchsorted(tg, kp - n_lo, side='right')
    window_has_tg = right > left

    out.update(
        kp_with_target_in_window=int(window_has_tg.sum()),
        kp_without_target_in_window=int((~window_has_tg).sum()))

    # Delay to the target inside the window closest to the keypress
    if window_has_tg.any():
        nearest = np.array([(k - tg[r - 1]) / sfreq
                            for k, r in zip(kp[window_has_tg],
                                            right[window_has_tg])])
        q1, q3 = np.percentile(nearest, [25, 75])
        out['delay_median_to_target_ms'] = float(np.median(nearest) * 1e3)
        out['delay_iqr_to_target_ms'] = float((q3 - q1) * 1e3)

    # Delay to the nearest preceding image of any kind, as a marker sanity check
    pic_t = np.sort(np.concatenate([tg, nt]))
    idx = np.searchsorted(pic_t, kp, side='right') - 1
    if (idx >= 0).any():
        m = idx >= 0
        d = (kp[m] - pic_t[np.clip(idx, 0, len(pic_t) - 1)][m]) / sfreq
        out['delay_median_to_nearest_image_ms'] = float(np.median(d) * 1e3)

    # target -> does a keypress follow inside the window?
    l2 = np.searchsorted(kp, tg + n_lo, side='left')
    r2 = np.searchsorted(kp, tg + n_hi, side='right')
    got_kp = r2 > l2

    out.update(
        target_with_keypress=int(got_kp.sum()),
        target_without_keypress=int((~got_kp).sum()))

    n_pic = len(tg) + len(nt)
    if n_pic:
        out['target_ratio'] = float(len(tg) / n_pic)

    # More than one keypress per target means repeated presses, which is worth
    # knowing before those trials enter any analysis.
    if len(tg):
        out['keypress_per_target'] = float(len(kp) / len(tg))

    return out


# %% ---- 2026-10-12 ------------------------
# Play ground
if args.summarize:
    files = sorted(OUTPUT_DIR.glob(f'{MODE}-S*.csv'))
    assert len(files) > 0, f'No per-subject csv found in {OUTPUT_DIR} for {MODE=}'
    logger.info(f'Summarizing {len(files)} subjects: {[e.stem for e in files]}')

    df = pd.concat([pd.read_csv(e) for e in files], ignore_index=True)
    df.to_csv(OUTPUT_DIR / f'summary-{MODE}.csv',
              index=False, float_format='%.6g')

    print(df.to_string(index=False))
    print()
    print('-' * 72)
    for col in ['targets', 'nontargets', 'keypresses', 'other_events',
                'kp_with_target_in_window', 'kp_without_target_in_window',
                'target_with_keypress', 'target_without_keypress']:
        if col in df.columns:
            print(f'{col:<30} total {int(df[col].fillna(0).sum()):>8}')

    sys.exit(0)

# %%
RAW_DIR = Path('rawData/RSVP_dataset/processed_data') / f'{MODE}_{SUBJ}'
assert RAW_DIR.exists(), f'{RAW_DIR} does not exist'

logger.info(f'Reading {RAW_DIR}')
raw = load_raw(RAW_DIR)
events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
logger.info(f'{event_id=}')
logger.info(f'{raw.info["sfreq"]=}, n_events={len(events)}')

# The annotation labels are whatever the acquisition used; map them onto the
# three roles. Anything missing shows up in the log rather than silently
# producing zeros.
codes = {}
for role, wanted in [('target', args.target_code),
                     ('nontarget', args.nontarget_code),
                     ('keypress', args.keypress_code)]:
    if wanted not in event_id:
        logger.warning(f'No annotation {wanted!r} for {role}; '
                       f'available: {sorted(event_id)}')
        continue
    codes[role] = event_id[wanted]

assert len(codes) == 3, f'Could not map all three roles, got {codes}'

row = dict(mode=MODE, subject=SUBJ,
           sfreq=float(raw.info['sfreq']), n_events=len(events),
           **classify_events(events, codes, raw.info['sfreq'],
                             (RT_LO, RT_HI), logger))

fname = OUTPUT_DIR / f'{MODE}-{SUBJ}.csv'
pd.DataFrame([row]).to_csv(fname, index=False, float_format='%.6g')
logger.info(f'Saved into {fname}')
print(pd.DataFrame([row]).T.to_string())


# %% ---- 2026-10-12 ------------------------
# Pending
