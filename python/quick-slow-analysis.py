"""
File: quick-slow-analysis.py
Author: Chuncheng Zhang
Date: 2026-09-18
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Split the TARGET trials of one subject into quick and slow groups by their
    own reaction time, and compare the target ERPs of the two groups.
    The comparison is done on the epochs whose keypress artifact has been
    projected out (the 'removal-artificial' epochs), so the difference between
    the two groups reflects the target specific activity rather than the
    response locked artifact.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending

Note:
    The reaction times come from output/timeDelays/timeDelays-target.csv,
    which is computed by compute-time-delays.py. The 'index' column of that
    csv points at the trial index inside epochs-1, which is exactly the trial
    axis of the epochs file used here.

    Three splitting rules are tried in order, the first one that works wins:
      median    Split at the subject's own median RT. Accepted when both
                groups have enough trials AND the two group medians differ by
                at least --min_gap.
      threshold Split at a fixed RT (--threshold, default 0.4 s). Accepted
                when both groups have enough trials.
      tertile   Keep the fastest and the slowest third of the trials.
    Which rule was used is written into the summary csv, so the group level
    analysis can be run with or without the degenerate subjects.
"""


# %% ---- 2026-09-18 ------------------------
# Requirements and constants
from util.easy_imports import *
from scipy.stats import spearmanr

# %%
parser = argparse.ArgumentParser(
    description='Split the target trials by reaction time and compare ERPs')
parser.add_argument('-s', '--subject', default='S02',
                    help='Subject name like S02')
parser.add_argument('-m', '--mode', default='EEG', help='Mode name EEG | MEG')
parser.add_argument('-d', '--data-dir', default='output/epochs',
                    help='Folder containing the {MODE}-{SUBJ} folders')
parser.add_argument('-o', '--output-dir', default='output/quick-slow',
                    help='Where the quick-slow results are written')
parser.add_argument('-t', '--tag', default='rma',
                    help='Short name of the input epochs, used in filenames')
parser.add_argument('--epochs-fname',
                    default='epochs-1-notch-removal-artificial-epo.fif',
                    help='The target epochs to split')
parser.add_argument('--reference-fname',
                    default='epochs-2-notch-removal-artificial-ave.fif',
                    help='Projected non-target evoked, drawn as reference')
parser.add_argument('--rt-csv',
                    default='output/timeDelays/timeDelays-target.csv',
                    help='The reaction times computed by compute-time-delays')
parser.add_argument('--rt-min', type=float, default=0.15,
                    help='Shortest accepted reaction time in seconds')
parser.add_argument('--rt-max', type=float, default=1.0,
                    help='Longest accepted reaction time in seconds')
parser.add_argument('--threshold', type=float, default=0.4,
                    help='The fixed RT threshold used by the second rule')
parser.add_argument('--min-gap', type=float, default=0.03,
                    help='Minimum RT difference between the group medians')
parser.add_argument('--min-trials', type=int, default=60,
                    help='Minimum number of trials in each group')
parser.add_argument('--split', default='auto',
                    choices=['auto', 'median', 'threshold', 'tertile'],
                    help='Force one splitting rule instead of trying in order')
parser.add_argument('--win', type=float, nargs=2, default=[0.2, 0.45],
                    help='Time window in seconds for the trial wise amplitude')
args = parser.parse_args()
SUBJ = args.subject
MODE = args.mode
TAG = args.tag

logger.info(f'Start with {SUBJ=}, {MODE=}, {TAG=}')

# %%
DATA_DIR = Path(args.data_dir) / f'{MODE}-{SUBJ}'
OUTPUT_DIR = Path(args.output_dir) / f'{MODE}-{SUBJ}'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
RT_WINDOW = (args.win[0], args.win[1])


# %% ---- 2026-09-18 ------------------------
# Function and class
def load_rt(fpath: Path, mode: str, subject: str, rt_min: float,
            rt_max: float):
    '''
    Read the reaction times of one subject and drop the implausible trials.

    The extremely short ones are usually a keypress that belongs to the
    previous target, the extremely long ones are missed targets whose pairing
    key belongs to a later image. Both would blur the quick and slow groups.

    Args:
        fpath: The timeDelays-target.csv written by compute-time-delays.py
        mode: EEG | MEG
        subject: Subject name like S02
        rt_min: The shortest accepted reaction time in seconds
        rt_max: The longest accepted reaction time in seconds

    Returns:
        df: The valid rows, sorted by the trial index, or None when empty
    '''
    if not fpath.exists():
        logger.warning(f'There is no such file {fpath}')
        return None

    df = pd.read_csv(fpath)
    keep = (df['mode'] == mode) & (df['subject'] == subject)
    df = df[keep][['delay', 'index']].copy()
    logger.debug(f'{len(df)} target trials of {mode}-{subject}')

    n_before = len(df)
    df = df[(df['delay'] >= rt_min) & (df['delay'] <= rt_max)]
    df = df.sort_values('index').reset_index(drop=True)
    logger.info(
        f'RT window [{rt_min:g}, {rt_max:g}] s keeps '
        f'{len(df)} / {n_before} trials')

    if len(df) < 2 * args.min_trials:
        logger.warning(
            f'Only {len(df)} trials are left, it is too few to split')
        return None
    return df


def split_rt(delays: np.ndarray, rule: str, threshold: float,
             min_trials: int, min_gap: float):
    '''
    Split the trials of one subject into a quick and a slow group.

    Args:
        delays: The reaction times of the trials, (n_trials,)
        rule: auto | median | threshold | tertile
        threshold: The fixed reaction time used by the threshold rule
        min_trials: The minimum number of trials in each group
        min_gap: The minimum difference between the two group medians

    Returns:
        quick, slow: Boolean masks over the trials
        info: The rule actually used and the resulting group statistics
    '''
    rules = ['median', 'threshold', 'tertile'] if rule == 'auto' else [rule]
    order = np.argsort(delays)
    last_info = None

    for name in rules:
        if name == 'median':
            boundary = np.median(delays)
            quick = delays <= boundary
        elif name == 'threshold':
            boundary = threshold
            quick = delays < boundary
        elif name == 'tertile':
            # Split by rank, not by value. The reaction times are quantized by
            # the 10 Hz presentation and a large part of the trials may share
            # the same value, in which case comparing with the quantiles would
            # put one trial into both groups.
            order_idx = np.argsort(delays, kind='stable')
            k = len(delays) // 3
            quick = np.zeros(len(delays), bool)
            slow = np.zeros(len(delays), bool)
            quick[order_idx[:k]] = True
            slow[order_idx[-k:]] = True
            boundary = float(delays[order_idx[k - 1]])
            cut = float(delays[order_idx[-k]])
            n_q, n_s = int(quick.sum()), int(slow.sum())
            if min(n_q, n_s) >= min_trials:
                info = dict(method=name, threshold=boundary,
                            n_quick=n_q, n_slow=n_s,
                            rt_quick=round(float(np.median(delays[quick])), 4),
                            rt_slow=round(float(np.median(delays[slow])), 4),
                            gap=round(float(np.median(delays[slow]) -
                                            np.median(delays[quick])), 4))
                return quick, slow, info
            last_info = dict(method=name, threshold=boundary,
                             n_quick=n_q, n_slow=n_s)
            continue
        else:
            raise ValueError(f'Unknown {rule=}')

        slow = ~quick
        n_q, n_s = int(quick.sum()), int(slow.sum())
        if n_q == 0 or n_s == 0:
            logger.info(
                f'{MODE}-{SUBJ}: the {name} split gives '
                f'{n_q} vs {n_s} trials, try the next rule')
            last_info = dict(method=name, threshold=boundary,
                             n_quick=n_q, n_slow=n_s, rt_quick=np.nan,
                             rt_slow=np.nan, gap=np.nan)
            continue
        gap = float(np.median(delays[slow]) - np.median(delays[quick]))
        info = dict(method=name, threshold=boundary,
                    n_quick=n_q, n_slow=n_s,
                    rt_quick=round(float(np.median(delays[quick])), 4),
                    rt_slow=round(float(np.median(delays[slow])), 4),
                    gap=round(gap, 4))

        enough = min(n_q, n_s) >= min_trials
        if name == 'median' and not (enough and gap >= min_gap):
            logger.info(
                f'{MODE}-{SUBJ}: the median split is too weak, '
                f'{n_q} vs {n_s} trials, gap {gap:.3f} s')
            last_info = info
            continue
        if not enough:
            logger.info(
                f'{MODE}-{SUBJ}: the {name} split gives '
                f'{n_q} vs {n_s} trials, try the next rule')
            last_info = info
            continue
        return quick, slow, info

    return None, None, last_info


def pick_data(info: mne.Info):
    '''
    Return the indices of the gradiometer / magnetometer / EEG channels.

    Args:
        info: MNE Info

    Returns:
        picks: Array of channel indices
    '''
    return mne.pick_types(info, meg=True, eeg=True, exclude='bads')


def global_field_power(data: np.ndarray):
    '''
    The root mean square deviation of the data from its instantaneous spatial
    average, computed over channels.

    Args:
        data: (..., n_channels, n_times)

    Returns:
        gfp: (..., n_times)
    '''
    return np.sqrt(((data - data.mean(axis=-2, keepdims=True)) ** 2)
                   .mean(axis=-2))


def gfp_peak(evoked: mne.Evoked, picks: np.ndarray, window: tuple):
    '''
    The latency and the amplitude of the strongest peak of the global field
    power inside a time window.

    Args:
        evoked: MNE Evoked
        picks: Channel indices to use
        window: (tmin, tmax) in seconds

    Returns:
        latency: The time of the peak in seconds
        amplitude: The global field power at the peak
    '''
    times = evoked.times
    mask = (times >= window[0]) & (times <= window[1])
    gfp = global_field_power(evoked.data[picks])
    i = int(np.argmax(gfp[mask]))
    return float(times[mask][i]), float(gfp[mask][i])


def trial_amplitude(epochs: mne.Epochs, window: tuple,
                    weights: np.ndarray = None):
    '''
    Score every single trial with a spatial filter built from the evoked.

    The filter is the scalp pattern averaged over the time window, so a trial
    gets a high score when its own activity shares that pattern. It is a one
    dimensional version of the spatial filtering used in single trial ERP
    analysis, and it keeps the sign of the response.

    Args:
        epochs: MNE Epochs
        window: (tmin, tmax) in seconds
        weights: The spatial filter. If None it is estimated from the evoked

    Returns:
        score: The trial wise score, (n_epochs,)
        weights: The spatial filter that was used
    '''
    times = epochs.times
    mask = (times >= window[0]) & (times <= window[1])
    data = epochs.get_data(copy=True)[:, :, mask].mean(axis=2)

    if weights is None:
        weights = epochs.average().data[:, mask].mean(axis=1)
    weights = weights / np.linalg.norm(weights)
    score = data @ weights
    return score, weights


# %% ---- 2026-09-18 ------------------------
# Play ground
SUMMARY_CSV = Path(args.output_dir) / f'summary-{TAG}.csv'


def upsert_summary(row: dict, fpath: Path, keys=('mode', 'subject')):
    '''
    Write one row into the summary csv, replacing the older row of the same
    subject if there is one, so rerunning the script never duplicates rows.

    Args:
        row: The new row
        fpath: The csv to write
        keys: The columns identifying a row
    '''
    if fpath.exists():
        df = pd.read_csv(fpath)
        same = np.ones(len(df), bool)
        for k in keys:
            same &= (df[k] == row[k]).values
        df = df[~same]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(list(keys)).reset_index(drop=True)
    df.to_csv(fpath, index=False)
    logger.debug(f'Saved into {fpath}, {len(df)} rows')


def main():
    '''
    Split the trials of one subject and compare the quick and slow ERPs.
    '''
    epochs_fname = DATA_DIR / args.epochs_fname
    if not epochs_fname.exists():
        logger.error(
            f'{epochs_fname} does not exist. Run '
            f'3.remove-keypress-artificial.sh for {MODE}-{SUBJ} first.')
        return 1

    df = load_rt(Path(args.rt_csv), MODE, SUBJ, args.rt_min, args.rt_max)
    if df is None or len(df) == 0:
        return 1

    epochs = mne.read_epochs(epochs_fname, preload=True, verbose='ERROR')
    logger.debug(f'Loaded {epochs_fname}, {epochs}')

    n_trials = len(epochs)
    df = df[df['index'] < n_trials]
    if len(df) < 2 * args.min_trials:
        logger.warning(f'Only {len(df)} trials have a usable reaction time')
        return 1

    delays = df['delay'].to_numpy()
    index = df['index'].to_numpy()
    quick, slow, info = split_rt(
        delays, args.split, args.threshold, args.min_trials, args.min_gap)
    if quick is None:
        logger.warning(f'{MODE}-{SUBJ}: no rule could split the trials')
        return 1
    logger.info(f'{MODE}-{SUBJ}: split by {info}, sizes '
                f'{int(quick.sum())} / {int(slow.sum())}')

    # Mark the subjects whose two groups share almost the same RT. Their
    # contrast is too small to support any conclusion, the group analysis can
    # leave them out with --exclude-weak.
    info['threshold'] = round(float(info['threshold']), 4)
    info['weak'] = int(float(info.get('gap', 0.)) < args.min_gap)
    if info['weak']:
        logger.warning(
            f'{MODE}-{SUBJ}: the two groups differ by less than '
            f'{args.min_gap:g} s, the contrast is weak')

    picks = pick_data(epochs.info)
    idx_quick = index[quick]
    idx_slow = index[slow]
    evoked_quick = epochs[idx_quick].average()
    evoked_slow = epochs[idx_slow].average()

    # Save the evoked of both groups for the group level analysis
    for name, evoked in [('quick', evoked_quick), ('slow', evoked_slow)]:
        fname = OUTPUT_DIR / f'target-{TAG}-{name}-ave.fif'
        evoked.save(fname, overwrite=True)
        logger.debug(f'Saved into {fname}')

    # Peak latency and amplitude of each group
    win_peak = (0.1, 0.8)
    t_q, a_q = gfp_peak(evoked_quick, picks, win_peak)
    t_s, a_s = gfp_peak(evoked_slow, picks, win_peak)
    logger.info(f'GFP peak quick: {t_q:.3f} s {a_q:.3g}, '
                f'slow: {t_s:.3f} s {a_s:.3g}')

    # Trial wise amplitude, scored with one common spatial filter
    score_all, weights = trial_amplitude(epochs, RT_WINDOW)
    score_df = pd.DataFrame(
        dict(index=index, delay=delays,
             group=np.where(quick, 'quick', np.where(slow, 'slow', 'drop')),
             amplitude=score_all[index]))
    fname = OUTPUT_DIR / f'quick-slow-trials-{TAG}.csv'
    score_df.to_csv(fname, index=False)
    logger.debug(f'Saved into {fname}')

    rho, pval = spearmanr(delays, score_all[index])
    logger.info(f'Spearman between RT and amplitude: rho={rho:+.3f}, p={pval:.3g}')

    sq = score_df[score_df['group'] == 'quick']['amplitude']
    ss = score_df[score_df['group'] == 'slow']['amplitude']

    row = dict(mode=MODE, subject=SUBJ, tag=TAG,
               n_total=n_trials, n_valid=len(df), **info,
               gfp_peak_t_quick=round(t_q, 4), gfp_peak_a_quick=round(a_q, 6),
               gfp_peak_t_slow=round(t_s, 4), gfp_peak_a_slow=round(a_s, 6),
               amp_quick=round(float(sq.mean()), 6),
               amp_slow=round(float(ss.mean()), 6),
               spearman_rho=round(float(rho), 4),
               spearman_p=float(f'{pval:.3g}'))
    upsert_summary(row, SUMMARY_CSV)

    # ---- figures ----
    # 1. The global field power of both groups, with the projected
    # non-target as a reference
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    gfp_q = global_field_power(evoked_quick.data[picks])
    gfp_s = global_field_power(evoked_slow.data[picks])
    scale = 1e6 if MODE == 'EEG' else 1e15
    unit = 'uV' if MODE == 'EEG' else 'fT'
    axes[0].plot(evoked_quick.times, gfp_q * scale, label='quick', lw=1.8)
    axes[0].plot(evoked_slow.times, gfp_s * scale, label='slow', lw=1.8)

    reference = DATA_DIR / args.reference_fname
    if reference.exists():
        evk_ref = mne.read_evokeds(reference, verbose='ERROR')[0]
        picks_ref = pick_data(evk_ref.info)
        axes[0].plot(evk_ref.times,
                     global_field_power(evk_ref.data[picks_ref]) * scale,
                     ls=':', color='0.4', label='non-target')
    axes[0].axvline(0, color='k', ls=':', lw=0.8)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel(f'Global field power ({unit})')
    axes[0].set_title(f'{MODE}-{SUBJ} target after removal, '
                      f'split by {info["method"]}')
    axes[0].legend()

    # 2. Topography of the difference between the two groups at their peaks
    diff = mne.combine_evoked([evoked_quick, evoked_slow], weights=[1, -1])
    t_diff, _ = gfp_peak(diff, picks, win_peak)
    _, tmax = evoked_quick.times[0], evoked_quick.times[-1]
    t_show = float(np.clip(t_diff, 0.05, tmax - 0.05))
    try:
        mne.viz.plot_topomap(
            diff.data[picks][:, np.argmin(np.abs(diff.times - t_show))],
            diff.info, axes=axes[1], show=False, cmap='RdBu_r')
        axes[1].set_title(f'quick - slow at {t_show:.3f} s')
    except Exception as e:
        logger.warning(f'Failed to draw the topography: {e}')

    # 3. Single trial amplitude against the reaction time
    axes[2].scatter(delays, score_all[index] * scale, s=6, alpha=0.35,
                    c=np.where(quick, 'tab:blue', 'tab:orange'))
    axes[2].set_xlabel('Reaction time (s)')
    axes[2].set_ylabel(f'Trial amplitude ({unit})')
    axes[2].set_title(f'Spearman rho = {rho:+.3f}, p = {pval:.3g}')

    fig.tight_layout()
    fname = OUTPUT_DIR / f'quick-slow-{TAG}.png'
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    logger.debug(f'Saved into {fname}')
    return 0


main()


# %% ---- 2026-09-18 ------------------------
# Pending


# %% ---- 2026-09-18 ------------------------
# Pending
