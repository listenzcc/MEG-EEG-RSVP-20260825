"""
File: check-decode-pattern.py
Author: Chuncheng Zhang
Date: 2026-10-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    The group side of stage 21. It reads the per subject patterns written by
    decode-pattern.py, averages them, and draws the time axis as a row of
    maps.

    Three things are computed, and they answer three different questions:

        shape        the average of the unit length patterns, normalised
                     again after averaging. This is the map that gets drawn.
                     It carries the topography and nothing else, because
                     every subject and every time point contributes a vector
                     of length one.
        agreement    the length of that average before it is normalised. Ten
                     unit vectors that point the same way average to a
                     vector of length near one, ten that disagree average to
                     something near zero. It is the same number a
                     split half correlation would give and it costs nothing
                     extra, so it is reported for every time point.
        strength     the mean norm of the un-normalised pattern. It says how
                     big the topography is in data units, which the shape
                     throws away.

    The split half reading is the one that decides whether a map can be read
    at all: the subjects are split in two halves many times, each half is
    averaged, and the two averages are correlated across channels. A time
    point whose maps agree across subjects has a high value there, a time
    point where every subject contributes a different topography does not.

    Maps are drawn per channel type. Magnetometers and gradiometers have
    different layouts and different units, so they never go into one map,
    and the gradiometer pairs are merged by mne the way it always does. A
    gradiometer whose partner is missing in one of the subjects is dropped,
    since a half pair cannot be merged.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-10-21 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Group maps of the per time point decoder')
parser.add_argument('--analysis', default='target',
                    choices=['target', 'rt'],
                    help='Which contrast to summarise')
parser.add_argument('--tag', default='rma',
                    help='Short name of the input epochs, used in filenames')
parser.add_argument('--input-dir', default='output/decode-pattern',
                    help='Folder containing the {MODE}-{SUBJ} folders')
parser.add_argument('--data-dir', default='output/epochs',
                    help='Folder containing the {MODE}-{SUBJ} folders of the '
                         'epochs, read for the sensor layout only')
parser.add_argument('--output-dir', default=None,
                    help='Where the group files are written. Left out, it is '
                         '--input-dir')
parser.add_argument('--modes', default='EEG,MEG',
                    help='Modalities to summarise, comma separated')
parser.add_argument('--subjects', default=None,
                    help='Subject list, comma separated. Left out, every '
                         '{MODE}-S* folder that carries the files is used')
parser.add_argument('--epochs-fname', default=None,
                    help='The epochs file the sensor layout is read from. '
                         'Left out, it follows --tag')
parser.add_argument('--info-subject', default=None,
                    help='Subject whose epochs carry the layout. Left out, '
                         'the first subject of the list')
parser.add_argument('--times', default=None,
                    help='Time points of the map row, in seconds, comma '
                         'separated. Left out, a fixed row covering the '
                         'baseline to the late window is used')
parser.add_argument('--win-base', '--win_base', dest='win_base', type=float,
                    nargs=2, default=[-0.2, -0.05],
                    help='The pre stimulus baseline window')
parser.add_argument('--win-preresp', '--win_preresp', dest='win_preresp',
                    type=float, nargs=2, default=[0.10, 0.20],
                    help='The window before any keypress can have happened')
parser.add_argument('--win-early', '--win_early', dest='win_early',
                    type=float, nargs=2, default=[0.24, 0.36],
                    help='The first peak window of the projected response')
parser.add_argument('--win-late', '--win_late', dest='win_late', type=float,
                    nargs=2, default=[0.41, 0.55],
                    help='The second peak window')
parser.add_argument('--n-split', '--n_split', dest='n_split', type=int,
                    default=200,
                    help='How many random halves the split half reading is '
                         'averaged over')
parser.add_argument('--seed', type=int, default=0,
                    help='Random state of the split half draws')
parser.add_argument('--kind', default='pattern',
                    choices=['pattern', 'diff'],
                    help="'pattern' summarises the decoder pattern, 'diff' "
                         'the plain average difference between the classes')
parser.add_argument('--vlim-quantile', dest='vlim_quantile', type=float,
                    default=0.98,
                    help='The colour limits are the symmetric quantile of '
                         'the absolute value over all drawn maps')
args = parser.parse_args()

ANALYSIS = args.analysis
TAG = args.tag
INPUT_DIR = Path(args.input_dir)
DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_dir) if args.output_dir else INPUT_DIR
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODES = [m.strip() for m in args.modes.split(',') if m.strip()]

if args.epochs_fname is None:
    args.epochs_fname = ('epochs-1-notch-removal-artificial-epo.fif'
                         if TAG == 'rma' else 'epochs-1-notch-epo.fif')

DEFAULT_TIMES = [-0.10, 0.00, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.36,
                 0.42, 0.50, 0.60]
TIMES = ([float(t) for t in args.times.split(',') if t.strip()]
         if args.times else DEFAULT_TIMES)

WINDOWS = dict(baseline=tuple(args.win_base),
               preresp=tuple(args.win_preresp),
               early=tuple(args.win_early),
               late=tuple(args.win_late))

NAME = ('patterns-unit' if args.kind == 'pattern' else 'diff')
KIND_TAG = 'pattern' if args.kind == 'pattern' else 'diff'

logger.info(f'Start with {args=}')

EPS = 1e-12


# %% ---- 2026-10-21 ------------------------
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


def read_channels(path: Path):
    '''
    Read the channel name and type table of one subject.

    Args:
        path: The channels csv written by decode-pattern.py

    Returns:
        names: The channel names
        types: The channel types, mne names like eeg, mag, grad
    '''
    df = pd.read_csv(path)
    return df['ch_name'].tolist(), df['ch_type'].tolist()


def complete_grad_pairs(names, types):
    '''
    Drop gradiometers whose partner is missing.

    A gradiometer pair shares the channel name up to the last character, so
    'MEG0112' and 'MEG0113' are one pair. mne merges a pair when it draws a
    map, and a channel left alone cannot be merged, so both members have to
    be present.

    Args:
        names: The channel names
        types: The channel types

    Returns:
        keep: The indices of the channels that survive
    '''
    groups = {}
    for i, (n, t) in enumerate(zip(names, types)):
        if t == 'grad':
            groups.setdefault(n[:-1], []).append(i)
    drop = set()
    for key, idx in groups.items():
        if len(idx) != 2:
            drop.update(idx)
    return [i for i in range(len(names)) if i not in drop]


def unit_vector(v: np.ndarray, axis: int = -1):
    '''
    Scale a vector to unit length, leaving a zero vector alone.

    Args:
        v: The array to scale
        axis: The axis the norm is taken along

    Returns:
        out: The scaled array, and the norm before scaling
    '''
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.where(n < EPS, 1.0, n)
    return v / n, np.linalg.norm(v, axis=axis)


def split_half(mats: np.ndarray, n_rep: int, rng):
    '''
    Correlate the averages of two random halves of the subjects.

    Args:
        mats: (n_subjects, n_times, n_channels) unit length patterns
        n_rep: How many random splits are averaged
        rng: The random generator

    Returns:
        r: (n_times,) the mean split half correlation
    '''
    n_sub = mats.shape[0]
    half = n_sub // 2
    if half < 2:
        return np.full(mats.shape[1], np.nan)
    r = np.zeros(mats.shape[1])
    for _ in range(n_rep):
        perm = rng.permutation(n_sub)
        a = mats[perm[:half]].mean(axis=0)
        b = mats[perm[half:2 * half]].mean(axis=0)
        a, na = unit_vector(a)
        b, nb = unit_vector(b)
        ok = (na > EPS) & (nb > EPS)
        c = np.zeros(mats.shape[1])
        c[ok] = np.sum(a[ok] * b[ok], axis=1)
        r += c
    return r / n_rep


def pick_type(names, types, ch_type):
    '''
    The indices of one channel type.

    Args:
        names: The channel names
        types: The channel types
        ch_type: eeg, mag or grad

    Returns:
        idx: The indices of that type
    '''
    return [i for i, t in enumerate(types) if t == ch_type]


def subset_info(info, names):
    '''
    An info reduced to a list of channels.

    Args:
        info: The full info
        names: The channel names to keep, in the order of the data

    Returns:
        info_sub: The reduced info
    '''
    where = {n: i for i, n in enumerate(info.ch_names)}
    sel = [where[n] for n in names if n in where]
    return mne.pick_info(info, sel, copy=True)


def draw_maps(fig, gs, row_start, n_cols, data, times_draw, info_sub, ch_type,
              vlim, title_prefix, annotate=None):
    '''
    Draw one block of maps, wrapping onto as many rows as it takes.

    Args:
        fig: The figure
        gs: The gridspec
        row_start: The first row of the block
        n_cols: How many maps fit in one row
        data: (n_times, n_channels) values to draw
        times_draw: The time points to draw
        info_sub: The info reduced to these channels
        ch_type: eeg, mag or grad
        vlim: The symmetric colour limit
        title_prefix: What goes in front of the time point in every title
        annotate: Optional (n_times,) numbers printed under the time point

    Returns:
        axes: The axes that were drawn into
    '''
    axes = []
    for j, t in enumerate(times_draw):
        ax = fig.add_subplot(gs[row_start + j // n_cols, j % n_cols])
        mne.viz.plot_topomap(data[j], pos=info_sub, ch_type=ch_type,
                             axes=ax, cmap='RdBu_r', vlim=(-vlim, vlim),
                             sensors=False, show=False, contours=0,
                             extrapolate='auto')
        label = f'{t:.2f} s'
        if j == 0:
            label = f'{title_prefix}{label}'
        if annotate is not None:
            label += f'\nr={annotate[j]:.2f}'
        ax.set_title(label, fontsize=8.5)
        axes.append(ax)
    return axes


# %% ---- 2026-10-21 ------------------------
# Play ground
def main():
    '''
    Average the patterns across subjects and draw the maps.

    Returns:
        int: 0 on success
    '''
    rng = np.random.default_rng(args.seed)
    summary_rows = []

    for MODE in MODES:
        if args.subjects:
            subjects = [s.strip() for s in args.subjects.split(',')
                        if s.strip()]
        else:
            folders = sorted(INPUT_DIR.glob(f'{MODE}-S*'))
            subjects = [f.name.split('-')[1] for f in folders]

        # ---- load every subject ----
        loaded = []
        for subj in subjects:
            f_pat = INPUT_DIR / f'{MODE}-{subj}' / f'{NAME}-{ANALYSIS}-{TAG}.csv'
            f_chn = INPUT_DIR / f'{MODE}-{subj}' / f'channels-{TAG}.csv'
            f_nrm = INPUT_DIR / f'{MODE}-{subj}' / f'norm-{ANALYSIS}-{TAG}.csv'
            if not (f_pat.exists() and f_chn.exists()):
                logger.warning(f'{MODE}-{subj}: {f_pat.name} or '
                               f'{f_chn.name} is missing, skipped')
                continue
            times, names, mat = read_matrix(f_pat)
            ch_names, ch_types = read_channels(f_chn)
            if len(ch_names) != mat.shape[1]:
                logger.warning(f'{MODE}-{subj}: {len(ch_names)} channels '
                               f'against {mat.shape[1]} columns, skipped')
                continue

            norm = None
            if f_nrm.exists():
                t_n, _, nm = read_matrix(f_nrm)
                if len(t_n) == len(times):
                    norm = nm.ravel()

            loaded.append(dict(subject=subj, times=times, names=ch_names,
                               types=ch_types, mat=mat, norm=norm))

        if not loaded:
            logger.warning(f'{MODE}: no subject left, skipped')
            continue

        # ---- the channels every one of them carries ----
        times_ref = loaded[0]['times']
        loaded = [L for L in loaded
                  if len(L['times']) == len(times_ref)
                  and np.allclose(L['times'], times_ref)]
        common = list(loaded[0]['names'])
        types_common = list(loaded[0]['types'])
        for L in loaded[1:]:
            have = set(L['names'])
            pos = [i for i, n in enumerate(common) if n in have]
            common = [common[i] for i in pos]
            types_common = [types_common[i] for i in pos]

        # A gradiometer whose partner is missing in the shared set cannot be
        # merged into one map, so the pair is dropped as a whole
        keep = complete_grad_pairs(common, types_common)
        if len(keep) != len(common):
            logger.info(f'{MODE}: {len(common) - len(keep)} shared '
                        f'gradiometers without a partner dropped')
        common = [common[i] for i in keep]
        types_common = [types_common[i] for i in keep]

        mats = []
        norms = []
        used = []
        for L in loaded:
            lookup = {n: i for i, n in enumerate(L['names'])}
            mats.append(L['mat'][:, [lookup[n] for n in common]])
            if L['norm'] is not None:
                norms.append(L['norm'])
            used.append(L['subject'])

        mats = np.stack(mats)          # (n_subjects, n_times, n_channels)
        logger.info(f'{MODE}: {len(used)} subjects, {len(common)} shared '
                    f'channels, {mats.shape[1]} time points')

        # ---- the group shape, and how much the subjects agree on it ----
        if args.kind == 'pattern':
            mean = mats.mean(axis=0)                  # already unit length
            agreement = np.linalg.norm(mean, axis=1)
            shape, _ = unit_vector(mean)
        else:
            mean = mats.mean(axis=0)
            agreement = np.full(mean.shape[0], np.nan)
            shape = mean.copy()

        # ---- split half, per channel type ----
        ch_types_here = []
        for t in ['eeg', 'mag', 'grad']:
            if t in types_common:
                ch_types_here.append(t)

        r_by_type = {}
        for t in ch_types_here:
            idx = pick_type(common, types_common, t)
            r_by_type[t] = split_half(mats[:, :, idx], args.n_split, rng)

        # ---- the sensor layout ----
        info_subj = args.info_subject if args.info_subject else used[0]
        f_info = DATA_DIR / f'{MODE}-{info_subj}' / args.epochs_fname
        if not f_info.exists():
            logger.warning(f'{MODE}: {f_info} is missing, no maps drawn')
            continue
        info = mne.read_epochs(f_info, verbose='ERROR').info
        have = set(info.ch_names)
        missing = [n for n in common if n not in have]
        if missing:
            logger.warning(f'{MODE}: {len(missing)} of the shared channels '
                           f'are not in {info_subj}, no maps drawn')
            continue

        # ---- which channel types can actually be drawn ----
        # A magnetometer and a gradiometer layout are not the same object and
        # mne merges a gradiometer pair only when both members are there, so
        # each type is probed once and a type that fails is left out instead
        # of taking the whole figure down with it
        drawable = []
        probe = plt.figure(figsize=(2, 2))
        probe_ax = probe.add_subplot(1, 1, 1)
        for t in ch_types_here:
            idx = pick_type(common, types_common, t)
            try:
                mne.viz.plot_topomap(shape[len(shape) // 2][idx],
                                     pos=subset_info(info,
                                                     [common[i] for i in idx]),
                                     ch_type=t, axes=probe_ax, show=False,
                                     sensors=False)
                drawable.append(t)
            except Exception as exc:
                logger.warning(f'{MODE}: cannot draw {t} maps ({exc}), '
                               f'skipped')
            probe_ax.clear()
        plt.close(probe)
        ch_types_here = drawable
        if not ch_types_here:
            logger.warning(f'{MODE}: no channel type can be drawn, skipped')
            continue

        # ---- the map row ----
        i_draw = [int(np.argmin(np.abs(times_ref - t))) for t in TIMES]
        times_draw = times_ref[i_draw]

        from matplotlib.gridspec import GridSpec
        n_cols = min(6, len(TIMES))
        n_blocks = int(np.ceil(len(TIMES) / n_cols))
        n_rows = n_blocks * len(ch_types_here)
        fig = plt.figure(figsize=(1.45 * n_cols, 1.95 * n_rows + 0.8))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.42, wspace=0.10)

        for block, t in enumerate(ch_types_here):
            idx = pick_type(common, types_common, t)
            info_t = subset_info(info, [common[i] for i in idx])
            data = shape[:, idx]
            vlim = float(np.quantile(np.abs(data[i_draw]),
                                     args.vlim_quantile))
            vlim = max(vlim, EPS)
            draw_maps(fig, gs, block * n_blocks, n_cols, data[i_draw],
                      times_draw, info_t, t, vlim, f'{MODE} {t} ',
                      annotate=r_by_type[t][i_draw])

        label = ('target vs non-target' if ANALYSIS == 'target'
                 else 'quick vs slow')
        kind = ('decoder pattern' if args.kind == 'pattern'
                else 'average difference')
        fig.suptitle(f'{MODE} {label}, {kind} ({TAG}, N = {len(used)}). '
                     f'Maps are unit length, colour is topography only. '
                     f'r = split half across subjects.', fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = OUTPUT_DIR / f'group-decode-{KIND_TAG}-{ANALYSIS}-{TAG}-{MODE}.png'
        fig.savefig(fname, dpi=110)
        plt.close(fig)
        logger.info(f'Saved into {fname}')

        # ---- the window maps ----
        n_types = len(ch_types_here)
        fig, axes = plt.subplots(
            n_types, len(WINDOWS), figsize=(2.0 * len(WINDOWS),
                                            2.35 * n_types))
        axes = np.atleast_2d(axes)
        if n_types == 1:
            axes = axes.reshape(1, -1)
        for row, t in enumerate(ch_types_here):
            idx = pick_type(common, types_common, t)
            info_t = subset_info(info, [common[i] for i in idx])
            win_data = []
            for name, w in WINDOWS.items():
                mask = (times_ref >= w[0]) & (times_ref <= w[1])
                m = shape[mask][:, idx].mean(axis=0)
                if args.kind == 'pattern':
                    # Averaging unit vectors in a window already shrinks the
                    # length by the amount the window disagrees with itself,
                    # and that reading belongs to the stability figure. Here
                    # the shape is wanted, so the window average is scaled
                    # back to unit length the same way a single map is
                    n = np.linalg.norm(m)
                    if n > EPS:
                        m = m / n
                win_data.append(m)
            win_data = np.array(win_data)
            vlim = float(np.quantile(np.abs(win_data),
                                     args.vlim_quantile)) or EPS
            for j, (name, w) in enumerate(WINDOWS.items()):
                ax = axes[row, j]
                mne.viz.plot_topomap(win_data[j], pos=info_t, ch_type=t,
                                     axes=ax, cmap='RdBu_r',
                                     vlim=(-vlim, vlim), sensors=False,
                                     show=False, contours=0,
                                     extrapolate='auto')
                prefix = f'{t} ' if len(ch_types_here) > 1 else ''
                ax.set_title(f'{prefix}{name}\n{w[0]:.2f} to {w[1]:.2f} s',
                             fontsize=8.5)
        fig.suptitle(f'{MODE} {label}, window mean of the {kind} '
                     f'({TAG}, N = {len(used)}). One colour scale for all '
                     f'four windows.', fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fname = (OUTPUT_DIR
                 / f'group-decode-{KIND_TAG}-windows-{ANALYSIS}-{TAG}-{MODE}.png')
        fig.savefig(fname, dpi=110)
        plt.close(fig)
        logger.info(f'Saved into {fname}')

        # ---- the two readings over time ----
        fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
        ax = axes[0]
        for t in ch_types_here:
            ax.plot(times_ref, r_by_type[t], lw=1.6,
                    label=f'{t} split half')
        if args.kind == 'pattern':
            ax.plot(times_ref, agreement, lw=1.6, color='k', ls='--',
                    label='agreement of all subjects')
            # The length of an average of random unit vectors is 1 over the
            # root of their number, so a map with no shared shape still sits
            # on that value and not on zero
            ax.axhline(1 / np.sqrt(len(used)), color='gray', lw=0.9, ls='-.',
                       label=f'no shared shape ({1 / np.sqrt(len(used)):.2f})')
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        for name, w in WINDOWS.items():
            ax.axvspan(*w, color='0.5', alpha=0.12)
        ax.set_ylabel('Correlation across subjects')
        ax.set_ylim(-0.6, 1.0)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_title(f'{MODE} {label}: how stable the topography is '
                     f'({TAG}, {args.n_split} splits)', fontsize=9)

        ax = axes[1]
        if norms:
            nm = np.stack(norms)
            ax.plot(times_ref, nm.mean(axis=0), lw=1.6, color='#1f77b4',
                    label='mean norm of the pattern')
            ax.fill_between(times_ref, nm.mean(axis=0) - nm.std(axis=0)
                            / np.sqrt(nm.shape[0]),
                            nm.mean(axis=0) + nm.std(axis=0)
                            / np.sqrt(nm.shape[0]), color='#1f77b4',
                            alpha=0.18)
        elif args.kind == 'diff':
            ax.plot(times_ref, np.linalg.norm(shape, axis=1), lw=1.6,
                    color='#1f77b4',
                    label='norm of the average difference')
        for name, w in WINDOWS.items():
            ax.axvspan(*w, color='0.5', alpha=0.12)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strength')
        ax.legend(fontsize=8, loc='upper right')
        fig.tight_layout()
        fname = (OUTPUT_DIR
                 / f'group-decode-{KIND_TAG}-stability-{ANALYSIS}-{TAG}-{MODE}.png')
        fig.savefig(fname, dpi=110)
        plt.close(fig)
        logger.info(f'Saved into {fname}')

        # ---- the csv ----
        df = pd.DataFrame(shape, columns=common)
        df.insert(0, 'time', np.round(times_ref, 6))
        if args.kind == 'pattern':
            df.insert(1, 'agreement', np.round(agreement, 6))
        for t in ch_types_here:
            df[f'split_half_{t}'] = np.round(r_by_type[t], 6)
        if norms:
            df['norm_mean'] = np.round(np.stack(norms).mean(axis=0), 8)
        fname = (OUTPUT_DIR
                 / f'group-decode-{KIND_TAG}-{ANALYSIS}-{TAG}-{MODE}.csv')
        df.to_csv(fname, index=False, float_format='%.6g')
        logger.info(f'Saved into {fname}')

        row = dict(mode=MODE, analysis=ANALYSIS, tag=TAG, kind=KIND_TAG,
                   n_subjects=len(used), n_channels=len(common),
                   subjects=','.join(used),
                   channel_types=','.join(ch_types_here))
        for name, w in WINDOWS.items():
            mask = (times_ref >= w[0]) & (times_ref <= w[1])
            if args.kind == 'pattern':
                row[f'agreement_{name}'] = round(
                    float(np.nanmean(agreement[mask])), 4)
            for t in ch_types_here:
                row[f'split_half_{t}_{name}'] = round(
                    float(np.nanmean(r_by_type[t][mask])), 4)
        summary_rows.append(row)

    if summary_rows:
        fname = OUTPUT_DIR / f'group-decode-{KIND_TAG}-{ANALYSIS}-{TAG}-summary.csv'
        pd.DataFrame(summary_rows).to_csv(fname, index=False,
                                          float_format='%.6g')
        logger.info(f'Saved into {fname}')
        print(pd.DataFrame(summary_rows).T.to_string())
    return 0


main()


# %% ---- 2026-10-21 ------------------------
# Pending


# %% ---- 2026-10-21 ------------------------
# Pending
