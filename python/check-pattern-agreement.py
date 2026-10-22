'''
How reliable the decoder topographies of stage 21 are, and when the shape
changes.

`check-decode-pattern.py` draws the maps and reports a split half correlation
for a handful of windows. This script puts a test under the picture:

1. Agreement over time. For every time point it asks whether the ten subjects
   point the same way in channel space. The reading is the length of the
   average unit pattern, which sits near 1 / sqrt(10) even when the subjects
   share nothing, so the null is built by flipping the sign of each subject
   and reading the same number again. Signs are flipped per subject for the
   whole time series, which keeps the temporal structure a subject has. The
   time points above the cluster forming threshold are clustered along time
   and given a p value from the same flips.
2. Whether two windows carry the same topography. Comparing the group averages
   of two windows is not enough: both contain the same subjects, so their
   noise is shared and the cosine is pulled towards one. The comparison here
   is between one half of the subjects in the first window and the other half
   in the second, which removes that path. A negative cosine means the map
   flips, not that it shrinks.
3. Which channels carry the average topography, as a function of time.

Usage:
    python check-pattern-agreement.py --analysis target --tag rma
'''

import argparse
import itertools
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% ---- 2026-10-22 ------------------------
# Constant
parser = argparse.ArgumentParser(
    description='Test how consistent the decoder topographies are across '
                'subjects, and whether they keep their shape over time')
parser.add_argument('--analysis', default='target',
                    choices=['target', 'rt'],
                    help='Which contrast of stage 21 to read')
parser.add_argument('--tag', default='rma',
                    help='rma for the projected epochs, norp for the ones '
                         'before the projection')
parser.add_argument('--modes', default='EEG,MEG',
                    help='The modalities, comma separated')
parser.add_argument('--subjects', default=None,
                    help='The subjects, comma separated. Left out, S01 to '
                         'S10')
parser.add_argument('--input-dir', default='output/decode-pattern',
                    help='Where decode-pattern.py wrote its csv')
parser.add_argument('--output-dir', default=None,
                    help='Where to write, the input directory by default')
parser.add_argument('--win-base', '--win_base', dest='win_base', type=float,
                    nargs=2, default=[-0.2, -0.05],
                    help='The pre stimulus baseline window')
parser.add_argument('--win-preresp', '--win_preresp', dest='win_preresp',
                    type=float, nargs=2, default=[0.10, 0.20],
                    help='The window before any keypress can have happened')
parser.add_argument('--win-early', '--win_early', dest='win_early',
                    type=float, nargs=2, default=[0.24, 0.36],
                    help='The first peak window')
parser.add_argument('--win-late', '--win_late', dest='win_late', type=float,
                    nargs=2, default=[0.41, 0.55],
                    help='The second peak window')
parser.add_argument('--alpha', type=float, default=0.05,
                    help='The cluster forming threshold, as a percentile of '
                         'the permutation null')
parser.add_argument('--n-perm', type=int, default=2000,
                    help='How many sign flips to draw. Every flip is used '
                         'when there are fewer than this')
parser.add_argument('--n-split', '--n_split', dest='n_split', type=int,
                    default=400,
                    help='How many random halves the window comparison is '
                         'averaged over')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed of the sign flips and of the halves')
args = parser.parse_args()

ANALYSIS = args.analysis
TAG = args.tag
INPUT_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir) if args.output_dir else INPUT_DIR
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODES = [m.strip() for m in args.modes.split(',') if m.strip()]
SUBJECTS = ([s.strip() for s in args.subjects.split(',') if s.strip()]
            if args.subjects else [f'S{i:02d}' for i in range(1, 11)])

WINDOWS = dict(baseline=tuple(args.win_base),
               preresp=tuple(args.win_preresp),
               early=tuple(args.win_early),
               late=tuple(args.win_late))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger('agreement')


# %% ---- 2026-10-22 ------------------------
# Function and class
def read_matrix(path: Path):
    '''
    Read one per subject csv of the stage.

    Args:
        path: The csv written by decode-pattern.py

    Returns:
        times: (n_times,)
        names: The channel names in the column order of the file
        mat: (n_times, n_channels)
    '''
    df = pd.read_csv(path)
    names = [c for c in df.columns if c != 'time']
    return df['time'].to_numpy(), names, df[names].to_numpy()


def sign_matrix(n_sub: int, n_perm: int, rng):
    '''
    The sign flips the null is read from.

    Every combination is used when there are few enough subjects, otherwise
    random ones are drawn. The all plus one flip is the observed reading and
    is kept: it belongs in the null the same way every other flip does.

    Args:
        n_sub: The number of subjects
        n_perm: How many flips are wanted
        rng: The random generator

    Returns:
        signs: (n_perm, n_sub) of plus and minus one
    '''
    if n_sub <= 12 and 2 ** n_sub <= n_perm:
        code = np.arange(2 ** n_sub)
        signs = ((code[:, None] >> np.arange(n_sub)[None, :]) & 1) * 2 - 1
        logger.info(f'{2 ** n_sub} sign flips enumerated over {n_sub} '
                    f'subjects')
        return signs.astype(float)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n_sub))
    signs[0] = 1.0
    logger.info(f'{n_perm} sign flips drawn over {n_sub} subjects')
    return signs


def agreement_curve(units: np.ndarray, signs: np.ndarray):
    '''
    Length of the average topography under every sign flip.

    Args:
        units: (n_subjects, n_times, n_channels) unit length patterns
        signs: (n_perm, n_subjects) sign flips

    Returns:
        out: (n_perm, n_times) the length of the average
    '''
    means = np.einsum('ps,stc->ptc', signs, units) / units.shape[0]
    return np.linalg.norm(means, axis=2)


def find_clusters(curve: np.ndarray, thr: float):
    '''
    Contiguous stretches above a threshold.

    Args:
        curve: (n_times,) the values
        thr: The threshold

    Returns:
        out: A list of (start, stop, mass), the stop is exclusive
    '''
    out = []
    above = curve > thr
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i
        while j < len(above) and above[j]:
            j += 1
        out.append((i, j, float(np.sum(curve[i:j] - thr))))
        i = j
    return out


def cluster_test(obs: np.ndarray, null: np.ndarray, alpha: float):
    '''
    One dimensional cluster mass test along time.

    Args:
        obs: (n_times,) the observed agreement
        null: (n_perm, n_times) the agreement under every sign flip
        alpha: The cluster forming percentile of the null

    Returns:
        thr: The cluster forming threshold
        clusters: A list of (start, stop, mass, p), the stop is exclusive
    '''
    thr = float(np.quantile(null, 1 - alpha))
    obs_clusters = find_clusters(obs, thr)
    if not obs_clusters:
        return thr, []
    null_mass = np.array([max([c[2] for c in find_clusters(row, thr)],
                              default=0.0) for row in null])
    out = []
    for start, stop, mass in obs_clusters:
        p = (1 + np.sum(null_mass >= mass)) / (1 + len(null_mass))
        out.append((start, stop, mass, float(p)))
    logger.info(f'{len(out)} cluster(s) above {thr:.3f}')
    return thr, out


def cosine(a: np.ndarray, b: np.ndarray):
    '''
    Cosine between two topographies.

    Args:
        a: One pattern
        b: The other pattern

    Returns:
        out: The cosine, NaN when either side is empty
    '''
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0.0 or nb <= 0.0:
        return np.nan
    return float(a @ b / (na * nb))


def window_mean(units: np.ndarray, times: np.ndarray, win):
    '''
    Average the topographies inside a window.

    Args:
        units: (n_subjects, n_times, n_channels) unit length patterns
        times: (n_times,) the time points
        win: The window, (start, stop)

    Returns:
        out: (n_subjects, n_channels) the mean pattern per subject
    '''
    mask = (times >= win[0]) & (times <= win[1])
    return units[:, mask, :].mean(axis=1)


def split_half_cos(units: np.ndarray, times: np.ndarray, win_a, win_b,
                   i_time: int, n_rep: int, rng):
    '''
    Cosine between one half of the subjects in one place and the other half
    somewhere else.

    Reading the cosine between two group averages of the same subjects pulls
    it towards one, because the noise is shared. Splitting the subjects
    removes that path: what is left is the shape both halves agree on.

    Args:
        units: (n_subjects, n_times, n_channels) unit length patterns
        times: (n_times,) the time points
        win_a: The window of the first half, None for a single time point
        win_b: The window of the second half, None for a single time point
        i_time: Which time point to use where a window is None
        n_rep: How many random halves to average over
        rng: The random generator

    Returns:
        out: The mean cosine over the halves
    '''
    a = (units[:, i_time, :] if win_a is None
         else window_mean(units, times, win_a))
    b = (units[:, i_time, :] if win_b is None
         else window_mean(units, times, win_b))
    n_sub = units.shape[0]
    half = n_sub // 2
    if half < 2:
        return np.nan
    out = []
    for _ in range(n_rep):
        perm = rng.permutation(n_sub)
        out.append(cosine(a[perm[:half]].mean(axis=0),
                          b[perm[half:2 * half]].mean(axis=0)))
    return float(np.nanmean(out))


def extremum_channel(mean_map: np.ndarray, names):
    '''
    The channels that name a topography.

    Args:
        mean_map: (n_channels,) the group average pattern of one time point
        names: The channel names

    Returns:
        out: Three (name, value) pairs, the most negative, the most positive
             and the largest absolute one
    '''
    i_min = int(np.argmin(mean_map))
    i_max = int(np.argmax(mean_map))
    i_abs = int(np.argmax(np.abs(mean_map)))
    return ((names[i_min], float(mean_map[i_min])),
            (names[i_max], float(mean_map[i_max])),
            (names[i_abs], float(mean_map[i_abs])))


# %% ---- 2026-10-22 ------------------------
# Play ground
def main():
    '''
    Read one contrast of stage 21 and test its topographies over time.

    Returns:
        int: 0 on success
    '''
    rng = np.random.default_rng(args.seed)
    for MODE in MODES:
        mats, used, times_ref, names_ref = [], [], None, None
        for subj in SUBJECTS:
            f_pat = (INPUT_DIR / f'{MODE}-{subj}'
                     / f'patterns-unit-{ANALYSIS}-{TAG}.csv')
            f_chn = INPUT_DIR / f'{MODE}-{subj}' / f'channels-{TAG}.csv'
            if not (f_pat.exists() and f_chn.exists()):
                logger.warning(f'{MODE}-{subj}: {f_pat.name} or '
                               f'{f_chn.name} is missing, skipped')
                continue
            times, names, mat = read_matrix(f_pat)
            ch = pd.read_csv(f_chn)
            if len(ch) != mat.shape[1]:
                logger.warning(f'{MODE}-{subj}: {len(ch)} channels against '
                               f'{mat.shape[1]} columns, skipped')
                continue
            if float(np.abs(mat).max()) == 0.0:
                logger.warning(f'{MODE}-{subj}: the patterns are all zero, '
                               f'skipped, see the norm column')
                continue
            if times_ref is None:
                times_ref, names_ref = times, names
            else:
                if (len(times) != len(times_ref)
                        or not np.allclose(times, times_ref)):
                    logger.warning(f'{MODE}-{subj}: the time axis differs, '
                                   f'skipped')
                    continue
                shared = set(names)
                names_ref = [n for n in names_ref if n in shared]
            lookup = {n: i for i, n in enumerate(names)}
            mats.append(mat[:, [lookup[n] for n in names_ref]])
            used.append(subj)

        if len(used) < 3:
            logger.warning(f'{MODE}: only {len(used)} subject(s) left, '
                           f'nothing to test')
            continue
        units = np.stack(mats)         # (n_subjects, n_times, n_channels)
        n_sub, n_times, n_ch = units.shape
        logger.info(f'{MODE}: {n_sub} subjects, {n_ch} channels, '
                    f'{n_times} time points')

        signs = sign_matrix(n_sub, args.n_perm, rng)
        curves = agreement_curve(units, signs)
        obs = curves[0]
        thr, clusters = cluster_test(obs, curves, args.alpha)

        # ---- how much of the map is shared, per time point ----
        p_time = np.array([(1 + np.sum(curves[:, i] >= obs[i]))
                           / (1 + len(curves)) for i in range(n_times)])
        floor = 1 / np.sqrt(n_sub)
        # The squared length of the average, minus what random signs give,
        # over what is left: the share of the map the subjects have in common
        shared = np.clip((obs ** 2 - 1 / n_sub) / (1 - 1 / n_sub), 0, None)

        # ---- the window by window comparison ----
        names_win = list(WINDOWS)
        win_cos = np.full((len(names_win), len(names_win)), np.nan)
        for i, a in enumerate(names_win):
            for j, b in enumerate(names_win):
                if j < i:
                    win_cos[i, j] = win_cos[j, i]
                    continue
                win_cos[i, j] = split_half_cos(units, times_ref, WINDOWS[a],
                                               WINDOWS[b], 0, args.n_split,
                                               rng)
        cos_early = np.array([split_half_cos(units, times_ref, None,
                                             WINDOWS['early'], i,
                                             args.n_split, rng)
                              for i in range(n_times)])
        cos_late = np.array([split_half_cos(units, times_ref, None,
                                            WINDOWS['late'], i,
                                            args.n_split, rng)
                             for i in range(n_times)])

        # ---- the csv, one row per time point ----
        df = pd.DataFrame(dict(time=np.round(times_ref, 6),
                               agreement=np.round(obs, 6),
                               null_median=np.round(
                                   np.median(curves, axis=0), 6),
                               noise_floor=round(floor, 6),
                               shared_share=np.round(shared, 6),
                               p_uncorrected=np.round(p_time, 6),
                               cos_half_early=np.round(cos_early, 6),
                               cos_half_late=np.round(cos_late, 6)))
        mask_cluster = np.zeros(n_times, dtype=int)
        for k, (start, stop, mass, p) in enumerate(clusters, start=1):
            mask_cluster[start:stop] = k
        df['cluster'] = mask_cluster
        path = (OUTPUT_DIR
                / f'group-pattern-agreement-{ANALYSIS}-{TAG}-{MODE}.csv')
        df.to_csv(path, index=False, float_format='%.6g')
        logger.info(f'Saved into {path}')

        # ---- the csv, one row per window pair ----
        wdf = pd.DataFrame(win_cos, index=names_win, columns=names_win)
        path = (OUTPUT_DIR
                / f'group-pattern-window-cos-{ANALYSIS}-{TAG}-{MODE}.csv')
        wdf.round(4).to_csv(path)
        logger.info(f'Saved into {path}')

        # ---- the summary row ----
        row = dict(mode=MODE, analysis=ANALYSIS, tag=TAG,
                   n_subjects=n_sub, n_channels=n_ch,
                   subjects=','.join(used),
                   noise_floor=round(floor, 4),
                   cluster_thr=round(thr, 4), n_clusters=len(clusters))
        for k, (start, stop, mass, p) in enumerate(clusters, start=1):
            row[f'cluster{k}_window'] = (f'{times_ref[start]:.2f} to '
                                        f'{times_ref[stop - 1]:.2f}')
            row[f'cluster{k}_mass'] = round(mass, 4)
            row[f'cluster{k}_p'] = round(p, 4)
            row[f'cluster{k}_peak'] = round(float(obs[start:stop].max()), 4)
        for name, w in WINDOWS.items():
            m = (times_ref >= w[0]) & (times_ref <= w[1])
            row[f'agreement_{name}'] = round(float(np.nanmean(obs[m])), 4)
            row[f'shared_{name}'] = round(float(np.nanmean(shared[m])), 4)
            row[f'p_uncorrected_{name}'] = round(
                float(np.nanmean(p_time[m])), 4)
        for i, a in enumerate(names_win):
            for j, b in enumerate(names_win):
                if j <= i:
                    row[f'cos_{a}_{b}'] = round(float(win_cos[i, j]), 4)
        path = (OUTPUT_DIR / f'group-pattern-agreement-{ANALYSIS}-{TAG}'
                f'-{MODE}-summary.csv')
        pd.DataFrame([row]).to_csv(path, index=False, float_format='%.6g')
        logger.info(f'Saved into {path}')

        # ---- the figure ----
        fig = plt.figure(figsize=(9.4, 10.6))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.25, 1.0],
                              hspace=0.52, wspace=0.45)

        ax = fig.add_subplot(gs[0, :])
        ax.fill_between(times_ref, np.quantile(curves, 0.05, axis=0),
                        np.quantile(curves, 0.95, axis=0), color='0.72',
                        alpha=0.55, label='null, 5 to 95 percent')
        ax.plot(times_ref, obs, lw=1.8, color='k', label='observed')
        ax.axhline(thr, color='tab:red', lw=0.9, ls='--',
                   label=f'cluster threshold ({thr:.2f})')
        ax.axhline(floor, color='tab:blue', lw=0.9, ls='-.',
                   label=f'no shared shape ({floor:.2f})')
        for start, stop, mass, p in clusters:
            ax.axvspan(times_ref[start], times_ref[stop - 1], color='tab:red',
                       alpha=0.12)
            ax.text(np.mean([times_ref[start], times_ref[stop - 1]]), 0.97,
                    f'p = {p:.3f}', fontsize=8, color='tab:red', va='top',
                    ha='center')
        for name, w in WINDOWS.items():
            ax.axvspan(*w, color='0.5', alpha=0.10)
            ax.text(np.mean(w), floor + 0.02, name, fontsize=7, ha='center',
                    color='0.3', rotation=90)
        ax.set_ylabel('Length of the average\ntopography')
        ax.set_ylim(0.2, 1.0)
        ax.legend(fontsize=7, loc='upper left', ncol=2)
        ax.set_title(f'{MODE} {ANALYSIS} ({TAG}, N = {n_sub}): do the '
                     f'topographies agree across subjects?', fontsize=9)

        ax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(win_cos, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(names_win)))
        ax.set_xticklabels(names_win, fontsize=8, rotation=45)
        ax.set_yticks(range(len(names_win)))
        ax.set_yticklabels(names_win, fontsize=8)
        for i, j in itertools.product(range(len(names_win)), repeat=2):
            ax.text(j, i, f'{win_cos[i, j]:+.2f}', ha='center', va='center',
                    fontsize=8,
                    color='white' if abs(win_cos[i, j]) > 0.55 else 'black')
        # Every cell carries its own number, so the colour bar is left out:
        # squeezed in next to the label of the panel beside it, it only
        # collides with it
        ax.set_title('Cosine between windows, half the subjects\nagainst the '
                     'other half, red is the same shape', fontsize=9)

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(times_ref, cos_early, lw=1.6, label='against early window')
        ax.plot(times_ref, cos_late, lw=1.6, label='against late window')
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        for name, w in WINDOWS.items():
            ax.axvspan(*w, color='0.5', alpha=0.10)
        ax.set_ylim(-1, 1)
        ax.set_ylabel('Cosine, half\nagainst half')
        ax.set_xlabel('Time (s)')
        ax.legend(fontsize=7, loc='lower right')
        ax.set_title('Where the shape matches a window', fontsize=9)

        ax = fig.add_subplot(gs[2, :])
        mean_map = units.mean(axis=0)
        scale = float(np.abs(mean_map).max())
        peak = np.abs(mean_map).max(axis=0)
        order = np.argsort(peak)[::-1][:6]
        for rank, i in enumerate(order):
            ax.plot(times_ref, mean_map[:, i] / scale, lw=1.3, alpha=0.9,
                    label=f'{names_ref[i]}')
        for name, w in WINDOWS.items():
            ax.axvspan(*w, color='0.5', alpha=0.10)
        ax.set_xlim(times_ref[0], times_ref[-1])
        ax.legend(fontsize=7, ncol=6, loc='lower center', framealpha=0.9)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel loading\n(scaled to the peak)')
        ax.set_title('The six channels that carry the average topography, '
                     f'out of {n_ch}', fontsize=9)
        fname = (OUTPUT_DIR
                 / f'group-pattern-agreement-{ANALYSIS}-{TAG}-{MODE}.png')
        fig.savefig(fname, dpi=110, bbox_inches='tight')
        plt.close(fig)
        logger.info(f'Saved into {fname}')

        # ---- name the topographies out loud ----
        for name, w in WINDOWS.items():
            m = (times_ref >= w[0]) & (times_ref <= w[1])
            i_peak = int(np.where(m)[0][np.argmax(obs[m])])
            neg, pos, ab = extremum_channel(mean_map[i_peak], names_ref)
            logger.info(f'{MODE} {name} peak at {times_ref[i_peak]:.2f} s '
                        f'(agreement {obs[i_peak]:.2f}, shared '
                        f'{100 * shared[i_peak]:.0f}%): most negative '
                        f'{neg[0]} {neg[1]:+.3f}, most positive {pos[0]} '
                        f'{pos[1]:+.3f}, largest {ab[0]} {ab[1]:+.3f}')
    return 0


main()
