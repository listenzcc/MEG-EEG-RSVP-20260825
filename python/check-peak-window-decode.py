"""
File: check-peak-window-decode.py
Author: Chuncheng Zhang
Date: 2026-09-20
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Group level summary of the peak window decoding written by
    peak-window-decode.py.

    It answers three questions with statistics instead of a glance at a curve:
      - does each window carry target identity, and is the late window weaker
        than the early one;
      - does the classifier trained in one window transfer to the other;
      - does the epoch carry the response speed before any keypress can exist.

Usage:
    python python/check-peak-window-decode.py
    python python/check-peak-window-decode.py -t rma
"""


# %% ---- 2026-09-20 ------------------------
# Requirements and constants
from scipy.stats import wilcoxon, ttest_1samp
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Group level summary of the peak window decoding')
parser.add_argument('-d', '--data-dir', default='output/peak-window-decode',
                    help='Folder written by peak-window-decode.py')
parser.add_argument('-t', '--tag', default='rma',
                    help='Short name of the input epochs')
parser.add_argument('--win-preresp', type=float, nargs=2, default=[0.1, 0.2],
                    help='The window before the earliest plausible keypress, '
                         'must match the one of peak-window-decode.py')
parser.add_argument('--win-early', type=float, nargs=2, default=[0.24, 0.36],
                    help='The first peak window, for the shading only')
parser.add_argument('--win-late', type=float, nargs=2, default=[0.41, 0.55],
                    help='The second peak window, for the shading only')
parser.add_argument('--min-rt', type=float, default=0.,
                    help='Which variant of the window decoding to summarize, '
                         'it must match the --min-rt of peak-window-decode.py. '
                         '0 is the run on all the target trials')
args = parser.parse_args()
TAG = args.tag

# The control run on the late keypress trials is a second table, it must not
# overwrite the main one, so every output of it carries the latency
MINRT_SUFFIX = '' if args.min_rt <= 0 else f'-minrt{args.min_rt:g}'

DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = DATA_DIR
logger.info(f'Summarizing {DATA_DIR=}, {TAG=}')

WINDOW_CSV = DATA_DIR / f'window-decode-{TAG}.csv'
TRANSFER_CSV = DATA_DIR / f'transfer-decode-{TAG}.csv'
RT_CSV = DATA_DIR / f'rt-decode-{TAG}.csv'


# %% ---- 2026-09-20 ------------------------
# Function and class
def against_chance(values: np.ndarray, chance: float = .5):
    '''
    Test a column of AUC values against chance.

    Args:
        values: The AUC values of the subjects
        chance: The chance level

    Returns:
        tstat, pvalue: of the one sample t test, nan when too few subjects
    '''
    values = np.asarray(values, float)
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return np.nan, np.nan
    stat, p = ttest_1samp(values[mask], chance)
    return float(stat), float(p)


def paired(a: np.ndarray, b: np.ndarray):
    '''
    The Wilcoxon signed rank test between two columns of the same subjects.

    Args:
        a: The values of the first condition
        b: The values of the second condition

    Returns:
        statistic, pvalue: nan when there are too few subjects
    '''
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return np.nan, np.nan
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5 or np.allclose(a[mask], b[mask]):
        return np.nan, np.nan
    stat, p = wilcoxon(a[mask], b[mask])
    return float(stat), float(p)


def summarize(values):
    '''
    The mean and the standard error of one column, nan when it is empty.

    Args:
        values: The values of the subjects

    Returns:
        mean, sem
    '''
    v = np.asarray(values, float)
    if v.size == 0:
        return np.nan, np.nan
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1) / np.sqrt(v.size))


def column(df: pd.DataFrame, mode: str, **select):
    '''
    Pull one column of one modality as a float array.

    Args:
        df: The table
        mode: EEG | MEG
        select: The extra column equals value pairs

    Returns:
        values: The values, an empty array when the selection is missing
    '''
    keep = df['mode'] == mode
    for k, v in select.items():
        keep &= df[k] == v
    return df[keep]['auc'].to_numpy(dtype=float)


# %% ---- 2026-09-20 ------------------------
# Play ground
missing = [p.name for p in (WINDOW_CSV, TRANSFER_CSV, RT_CSV) if not p.exists()]
if missing:
    logger.error(f'{missing} do not exist. Run 12.peak-window-decode.sh first.')
    raise SystemExit(1)

win = pd.read_csv(WINDOW_CSV)
tra = pd.read_csv(TRANSFER_CSV)
rt = pd.read_csv(RT_CSV)

# The window table can hold several trial selections next to each other, the
# min rt column keeps them apart. A table written before that column existed
# has only the run on all the trials, and summarizing it as the control run
# would put the main result into a file named after the control.
if 'min_rt' in win.columns:
    n_all = len(win)
    min_rt = win['min_rt'].fillna(0.).to_numpy(dtype=float)
    win = win[np.isclose(min_rt, args.min_rt)]
    logger.info(f'--min-rt {args.min_rt:g} keeps {len(win)} of {n_all} rows '
                f'of the window table')
elif args.min_rt > 0:
    logger.error(f'{WINDOW_CSV} has no min_rt column, it was written before '
                 f'--min-rt existed. Run peak-window-decode.py --min-rt '
                 f'{args.min_rt:g} first.')
    raise SystemExit(1)
if win.empty:
    logger.error(f'There is no window row with min rt {args.min_rt:g}. Run '
                 f'peak-window-decode.py --min-rt {args.min_rt:g} first.')
    raise SystemExit(1)

# ---- the window decoding ----
rows = []
for mode in ['EEG', 'MEG']:
    for window in ['baseline', 'early', 'late']:
        v = column(win, mode, window=window)
        stat, p = against_chance(v)
        mean, sem = summarize(v)
        rows.append(dict(mode=mode, window=window, n=len(v),
                         auc=mean, sem=sem, tstat=stat, pvalue=p))
    q = column(win, mode, window='early')
    s = column(win, mode, window='late')
    stat, p = paired(q, s)
    mean_q, _ = summarize(q)
    mean_s, _ = summarize(s)
    rows.append(dict(mode=mode, window='early - late', n=min(len(q), len(s)),
                     auc=mean_q - mean_s, sem=np.nan,
                     tstat=stat, pvalue=p))
windows = pd.DataFrame(rows)
display(windows)

# ---- the transfer ----
rows = []
for mode in ['EEG', 'MEG']:
    for train_window in ['early', 'late']:
        for test_window in ['early', 'late']:
            v = column(tra, mode, train_window=train_window,
                       test_window=test_window)
            stat, p = against_chance(v)
            mean, sem = summarize(v)
            rows.append(dict(mode=mode, train_window=train_window,
                             test_window=test_window, n=len(v),
                             auc=mean, sem=sem, tstat=stat, pvalue=p))
transfer = pd.DataFrame(rows)
display(transfer)

# ---- the quick against slow decoding ----
rows = []
for mode in ['EEG', 'MEG']:
    v = rt[rt['mode'] == mode]['auc_preresp'].to_numpy(dtype=float)
    stat, p = against_chance(v)
    w = rt[rt['mode'] == mode]['auc_peak'].to_numpy(dtype=float)
    wstat, wp = against_chance(w)
    mean_v, sem_v = summarize(v)
    mean_w, sem_w = summarize(w)
    rows.append(dict(mode=mode, measure='pre response window', n=len(v),
                     auc=mean_v, sem=sem_v, tstat=stat, pvalue=p))
    rows.append(dict(mode=mode, measure='peak of the curve', n=len(w),
                     auc=mean_w, sem=sem_w, tstat=wstat, pvalue=wp))
rt_stats = pd.DataFrame(rows)
display(rt_stats)

# ---- figure ----
# The per subject rt curves live in rt-scores-{TAG}.txt of every subject folder.
# An exported result set often ships only the png and the csv, in which case the
# third column can not be drawn; the figure is then saved next to the complete
# one instead of on top of it.
partial = []
fig, axes = plt.subplots(2, 3, figsize=(21, 10))
for r, mode in enumerate(['EEG', 'MEG']):
    rng = np.random.default_rng(0)

    # Panel 1: the window decoding, with the subjects behind the mean
    ax = axes[r, 0]
    names = ['baseline', 'early', 'late']
    for i, window in enumerate(names):
        v = column(win, mode, window=window)
        mean, sem = summarize(v)
        ax.bar(i, mean, color=f'C{i}', alpha=.65, width=.6)
        ax.errorbar(i, mean, sem, color='k', capsize=4)
        ax.scatter(i + rng.uniform(-.12, .12, len(v)), v, s=14, color='k',
                   alpha=.6)
    ax.axhline(.5, color='gray', ls=':', lw=.9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel('AUC target versus non-target')
    ax.set_title(f'{mode}: target identity inside each window')
    ax.set_ylim(.4, 1.)

    # Panel 2: the transfer matrix
    ax = axes[r, 1]
    grid = np.full((2, 2), np.nan)
    for i, train_window in enumerate(['early', 'late']):
        for j, test_window in enumerate(['early', 'late']):
            v = column(tra, mode, train_window=train_window,
                       test_window=test_window)
            grid[i, j] = np.nanmean(v)
    im = ax.imshow(grid, vmin=.5, vmax=max(.8, np.nanmax(grid)), cmap='Reds')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{grid[i, j]:.3f}', ha='center', va='center',
                    fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['test early', 'test late'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['train early', 'train late'])
    ax.set_title(f'{mode}: cross window transfer')
    fig.colorbar(im, ax=ax, fraction=.046)

    # Panel 3: the group mean of the sliding quick against slow decoding
    ax = axes[r, 2]
    curves, times = [], None
    for folder in sorted(DATA_DIR.glob(f'{mode}-S*')):
        f_scores = folder / f'rt-scores-{TAG}.txt'
        f_times = folder / f'rt-times-{TAG}.txt'
        if not (f_scores.exists() and f_times.exists()):
            continue
        mean = np.loadtxt(f_scores).mean(axis=0)
        t = np.loadtxt(f_times)
        if times is None:
            times = t
        elif not np.array_equal(times, t):
            logger.warning(f'{folder.name} has another time axis')
            continue
        curves.append(mean)
    if curves:
        curves = np.array(curves)
        m = curves.mean(axis=0)
        sem = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
        ax.plot(times, m, lw=1.7, label='quick versus slow')
        ax.fill_between(times, m - sem, m + sem, alpha=.15)
        ax.axvspan(*args.win_preresp, color='tab:green', alpha=.15,
                   label='before any keypress can be')
        v = rt[rt['mode'] == mode]['auc_preresp'].to_numpy(dtype=float)
        mean_v, _ = summarize(v)
        ax.axhline(mean_v, color='tab:green', ls='--', lw=1.,
                   label=f'pre response mean {mean_v:.3f}')
        ax.axvspan(args.win_early[0], args.win_early[1], color='tab:red',
                   alpha=.08)
        ax.axvspan(args.win_late[0], args.win_late[1], color='tab:blue',
                   alpha=.08)
        ax.legend(fontsize=8)
    else:
        partial.append(mode)
        logger.warning(f'{mode}: no rt-scores-{TAG}.txt was found, the panel '
                       f'stays empty')
    ax.axhline(.5, color='gray', ls=':', lw=.9)
    ax.axvline(0, color='k', ls=':', lw=.8)
    ax.set_xlim(-.2, .8)
    ax.set_ylim(.4, .75)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('AUC')
    ax.set_title(f'{mode}: when does the epoch know the reaction time')

if args.min_rt > 0:
    fig.suptitle(f'The late window control, only the target trials whose '
                 f'keypress came after {args.min_rt:g} s\n'
                 f'the middle and the right panel keep every trial, '
                 f'--min-rt only reaches the window decoding')
    for r, mode in enumerate(['EEG', 'MEG']):
        n_mode = int((win['mode'] == mode).sum() // 3)
        if n_mode < 5:
            logger.warning(f'{mode}: only {n_mode} subject has a trial with a '
                           f'keypress after {args.min_rt:g} s, the top left '
                           f'panel is not a group result')

fig.tight_layout()
suffix = ('-no-rt-curve' if partial else '') + MINRT_SUFFIX
fname = OUTPUT_DIR / f'group-peak-window-decode-{TAG}{suffix}.png'
fig.savefig(fname, dpi=120)
plt.close(fig)
if partial:
    logger.warning(f'{partial} have no rt curve, the figure is saved as '
                   f'{fname.name} so that the complete one is left alone')
logger.info(f'Saved into {fname}')

tables = [('windows', windows)]
if args.min_rt > 0:
    # The transfer and the rt decoding read their own csv, which carry no min
    # rt column, so they would be written under the name of a control while
    # still holding every trial. That is worse than not writing them.
    logger.warning(f'--min-rt {args.min_rt:g} only filters the window '
                   f'decoding, the transfer and the rt tables of this run are '
                   f'left out on purpose. Rerun peak-window-decode.py with '
                   f'--min-rt if those two are needed as well.')
else:
    tables += [('transfer', transfer), ('rt', rt_stats)]

for name, table in tables:
    fname = OUTPUT_DIR / f'group-peak-window-{name}-{TAG}{MINRT_SUFFIX}.csv'
    table.to_csv(fname, index=False)
    logger.info(f'Saved into {fname}')


# %% ---- 2026-09-20 ------------------------
# Pending


# %% ---- 2026-09-20 ------------------------
# Pending
