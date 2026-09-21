"""
File: peak-structure.py
Author: Chuncheng Zhang
Date: 2026-09-20
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Describe the shape of the target response instead of only its maximum.

    The group figure of the quick-slow analysis shows two posterior peaks in
    MEG (around 0.29 s and 0.47 s, with a trough near 0.41 s) but a single
    peak in EEG. Picking one maximum per subject, as quick-slow-analysis.py
    does, is then not a stable readout: it returns the first peak for the
    subjects whose first peak is taller and the second peak for the others,
    which makes the group latency distribution bimodal and the paired test
    lose power. This script detects the peaks properly so that the first and
    the second peak can be followed separately.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending

Note:
    For each condition the global field power is measured the same way as in
    quick-slow-analysis.py, the root mean square deviation from the
    instantaneous spatial average, so the numbers of the two scripts are
    comparable.

    The second peak is only reported when a peak that passes the prominence
    threshold is found after the first one. A single peak response, which is
    what EEG looks like, leaves the second peak columns empty.

    The detection runs on a lightly smoothed global field power. A group
    average is smooth because it holds hundreds of trials, a single subject is
    not, and on the raw curve of one subject find_peaks locks onto the
    residual high frequency wiggle instead of the response. The figure shows
    both curves, the thin one is what the smoothing started from. Set
    --smooth 0 to switch the smoothing off and see the raw peaks.
"""


# %% ---- 2026-09-20 ------------------------
# Requirements and constants
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Describe the peak structure of the target response')
parser.add_argument('-s', '--subject', default='S02',
                    help='Subject name like S02')
parser.add_argument('-m', '--mode', default='EEG', help='Mode name EEG | MEG')
parser.add_argument('-d', '--data-dir', default='output/epochs',
                    help='Folder containing the {MODE}-{SUBJ} folders')
parser.add_argument('--qs-dir', default='output/quick-slow',
                    help='Folder written by quick-slow-analysis.py')
parser.add_argument('-o', '--output-dir', default='output/peak-structure',
                    help='Where the peak structure results are written')
parser.add_argument('-t', '--tag', default='rma',
                    help='Short name of the input epochs, used in filenames')
parser.add_argument('--target-fname',
                    default='epochs-1-notch-removal-artificial-ave.fif',
                    help='The target evoked')
parser.add_argument('--nontarget-fname',
                    default='epochs-2-notch-removal-artificial-ave.fif',
                    help='The projected non-target evoked, used as reference')
parser.add_argument('--win', type=float, nargs=2, default=[0.05, 0.8],
                    help='Time window in seconds to look for peaks')
parser.add_argument('--prominence', type=float, default=0.08,
                    help='Prominence to accept a peak, as a fraction of the '
                         'global field power range inside the window')
parser.add_argument('--min-distance', type=float, default=0.06,
                    help='Minimum distance between two peaks in seconds')
parser.add_argument('--min-ratio', type=float, default=0.35,
                    help='A peak counts only when its global field power is at '
                         'least this fraction of the tallest peak. Without it '
                         'a bump of noise far from the response is reported as '
                         'a second peak')
parser.add_argument('--smooth', type=float, default=0.015,
                    help='The standard deviation in seconds of the gaussian '
                         'that smooths the global field power before the peaks '
                         'are looked for, 0 switches it off')
parser.add_argument('--save-curves', action='store_true',
                    help='Also write the global field power of every condition '
                         'into a npz, which the group summary does not need but '
                         'hand made figures do')
args = parser.parse_args()
SUBJ = args.subject
MODE = args.mode
TAG = args.tag

logger.info(f'Start with {SUBJ=}, {MODE=}, {TAG=}')

# %%
DATA_DIR = Path(args.data_dir) / f'{MODE}-{SUBJ}'
QS_DIR = Path(args.qs_dir) / f'{MODE}-{SUBJ}'
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PEAK_CSV = OUTPUT_DIR / f'peak-structure-{TAG}.csv'


# %% ---- 2026-09-20 ------------------------
# Function and class
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


def detect_peaks(evoked: mne.Evoked, window: tuple, prominence: float,
                 min_distance: float, min_ratio: float, smooth: float = 0.,
                 top_n: int = 2):
    '''
    Detect the peaks of the global field power of one evoked.

    A peak has to be separated from the other peaks by at least min_distance
    seconds, rise at least prominence times the range above its surroundings,
    and reach at least min_ratio times the height of the tallest peak.

    Args:
        evoked: MNE Evoked
        window: (tmin, tmax) in seconds, where the peaks are looked for
        prominence: The threshold as a fraction of the global field power
            range inside the window
        min_distance: The minimum distance between two peaks in seconds
        min_ratio: The minimum height of a peak as a fraction of the tallest
            peak, which keeps a bump of noise far away from being reported as
            a second peak
        smooth: The standard deviation in seconds of the gaussian that smooths
            the global field power before the peaks are looked for, 0 keeps
            the raw curve
        top_n: How many peaks to keep

    Returns:
        peaks: List of dicts, sorted by latency, at most top_n long. Each has
            latency, gfp and prominence. Empty when nothing passes the
            threshold.
        gfp: The smoothed global field power of the whole epoch, it is the
            curve the peaks were read from
        gfp_raw: The global field power before the smoothing, for the figure
    '''
    times = evoked.times
    gfp_raw = global_field_power(evoked.data)
    sigma = smooth * evoked.info['sfreq']
    gfp = gaussian_filter1d(gfp_raw, sigma) if sigma > 0 else gfp_raw.copy()
    mask = (times >= window[0]) & (times <= window[1])
    segment = gfp[mask]
    segment_times = times[mask]

    height = prominence * (segment.max() - segment.min())
    distance = max(int(round(min_distance * evoked.info['sfreq'])), 1)
    index, _ = find_peaks(segment, height=height, distance=distance)
    # prominence is asked for separately, find_peaks only returns it when it
    # is also used as a threshold and the threshold here is on the height
    proms = peak_prominences(segment, index)[0] if index.size else np.array([])

    found = [dict(latency=float(segment_times[i]),
                  gfp=float(segment[i]),
                  prominence=float(proms[j]))
             for j, i in enumerate(index)]
    if found:
        tallest = max(p['gfp'] for p in found)
        found = [p for p in found if p['gfp'] >= min_ratio * tallest]
    found.sort(key=lambda p: -p['gfp'])
    found = found[:top_n]
    found.sort(key=lambda p: p['latency'])
    return found, gfp, gfp_raw


def describe(evoked: mne.Evoked, condition: str):
    '''
    Turn one evoked into a row of the summary table.

    Args:
        evoked: MNE Evoked
        condition: The name of the condition

    Returns:
        row: The summary row
    '''
    peaks, gfp, _ = detect_peaks(
        evoked, tuple(args.win), args.prominence, args.min_distance,
        args.min_ratio, args.smooth)

    times = evoked.times
    mask = (times >= args.win[0]) & (times <= args.win[1])
    i_argmax = int(np.argmax(gfp[mask]))

    row = dict(mode=MODE, subject=SUBJ, tag=TAG, condition=condition,
               n_peaks=len(peaks),
               argmax_t=round(float(times[mask][i_argmax]), 4),
               argmax_gfp=float(f'{gfp[mask][i_argmax]:.4g}'))

    for k, peak in enumerate(peaks, start=1):
        row[f'peak{k}_t'] = round(peak['latency'], 4)
        row[f'peak{k}_gfp'] = float(f"{peak['gfp']:.4g}")
        row[f'peak{k}_prom'] = float(f"{peak['prominence']:.4g}")

    # The trough between the first two peaks, the depth of the dip is what
    # makes the two peak structure visible at all.
    if len(peaks) >= 2:
        row['peak2_ratio'] = float(f"{peaks[1]['gfp'] / peaks[0]['gfp']:.4g}")
        between = (times > peaks[0]['latency']) & (times < peaks[1]['latency'])
        if between.any():
            i_dip = int(np.argmin(gfp[between]))
            row['dip_t'] = round(float(times[between][i_dip]), 4)
            row['dip_gfp'] = float(f'{gfp[between][i_dip]:.4g}')
            row['dip_ratio'] = float(
                f"{gfp[between][i_dip] / peaks[0]['gfp']:.4g}")
            row['dominant'] = 1 if peaks[0]['gfp'] >= peaks[1]['gfp'] else 2
    return row


def upsert(row: dict, fpath: Path, keys=('mode', 'subject', 'tag',
                                         'condition')):
    '''
    Write one row into the summary csv, replacing the older row of the same
    condition if there is one, so rerunning never duplicates rows.

    Args:
        row: The new row
        fpath: The csv to write
        keys: The columns identifying a row
    '''
    if fpath.exists():
        df = pd.read_csv(fpath)
        # A key column that the old table does not have yet is filled with the
        # value that stands for the default run, otherwise rerunning the
        # default appends a second row next to the old one instead of
        # replacing it.
        for k in keys:
            if k not in df.columns and k in row:
                df[k] = 0. if isinstance(row[k], (int, float)) else row[k]
        same = np.ones(len(df), bool)
        for k in keys:
            if k not in df.columns or k not in row:
                same &= False
                continue
            same &= (df[k] == row[k]).values
        df = df[~same]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(list(keys)).reset_index(drop=True)
    df.to_csv(fpath, index=False)
    logger.debug(f'Saved into {fpath}, {len(df)} rows')


# %% ---- 2026-09-20 ------------------------
# Play ground
def main():
    '''
    Describe the peak structure of the target, quick, slow and the reference
    non-target evoked of one subject.
    '''
    conditions = [
        ('target', DATA_DIR / args.target_fname),
        ('quick', QS_DIR / f'target-{TAG}-quick-ave.fif'),
        ('slow', QS_DIR / f'target-{TAG}-slow-ave.fif'),
        ('non-target', DATA_DIR / args.nontarget_fname),
    ]

    rows, curves = [], dict()
    for condition, fpath in conditions:
        if not fpath.exists():
            logger.warning(f'{fpath} does not exist, {condition} is skipped')
            continue
        evoked = mne.read_evokeds(fpath, verbose='ERROR')[0]
        row = describe(evoked, condition)
        rows.append(row)
        _, gfp, gfp_raw = detect_peaks(
            evoked, tuple(args.win), args.prominence, args.min_distance,
            args.min_ratio, args.smooth)
        curves[condition] = (evoked.times, gfp, gfp_raw, row)
        logger.info(f'{MODE}-{SUBJ} {condition}: {row["n_peaks"]} peaks, '
                    f'argmax {row["argmax_t"]:.3f} s, '
                    f'peaks {[row.get(f"peak{k}_t") for k in (1, 2)]}')

    if not rows:
        logger.error('There is no evoked to describe')
        return 1

    for row in rows:
        upsert(row, PEAK_CSV)

    # ---- figure ----
    scale = 1e6 if MODE == 'EEG' else 1e15
    unit = 'uV' if MODE == 'EEG' else 'fT'
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (condition, (times, gfp, gfp_raw, row)) in enumerate(curves.items()):
        colour = f'C{i}'
        ls = ':' if condition == 'non-target' else '-'
        # The thin line is the curve the smoothing started from, it is there
        # to show how much of the raw wiggle the peak detection had to survive.
        ax.plot(times, gfp_raw * scale, lw=.8, alpha=.3, color=colour)
        ax.plot(times, gfp * scale, ls=ls, lw=1.6, color=colour,
                label=condition)
        for k, marker_colour in [(1, 'k'), (2, 'tab:red')]:
            if f'peak{k}_t' not in row:
                continue
            ax.plot(row[f'peak{k}_t'], row[f'peak{k}_gfp'] * scale,
                    marker='v', ms=8, color=marker_colour)
            ax.annotate(f"p{k} {row[f'peak{k}_t']:.3f}s",
                        (row[f'peak{k}_t'], row[f'peak{k}_gfp'] * scale),
                        textcoords='offset points', xytext=(4, 6),
                        fontsize=8, color=marker_colour)
        if 'dip_t' in row:
            ax.plot(row['dip_t'], row['dip_gfp'] * scale, marker='^', ms=7,
                    color='0.4')
    ax.axvline(0, color='k', ls=':', lw=0.8)
    ax.set_xlim(args.win[0] - 0.05, args.win[1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Global field power ({unit})')
    tail = f' (smooth {args.smooth * 1e3:.0f} ms)' if args.smooth > 0 else ''
    ax.set_title(f'{MODE}-{SUBJ} peak structure after removal{tail}')
    ax.legend()
    fig.tight_layout()
    fname = OUTPUT_DIR / f'{MODE}-{SUBJ}-peak-structure-{TAG}.png'
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    logger.info(f'Saved into {fname}')

    if args.save_curves:
        times = next(iter(curves.values()))[0]
        payload = dict(times=times)
        for condition, (t, gfp, gfp_raw, _) in curves.items():
            payload[f'{condition}_gfp'] = gfp
            payload[f'{condition}_gfp_raw'] = gfp_raw
        fname = OUTPUT_DIR / f'{MODE}-{SUBJ}-peak-curves-{TAG}.npz'
        np.savez(fname, **payload)
        logger.info(f'Saved into {fname}')
    return 0


main()


# %% ---- 2026-09-20 ------------------------
# Pending


# %% ---- 2026-09-20 ------------------------
# Pending
