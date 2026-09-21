"""
File: check-peak-structure.py
Author: Chuncheng Zhang
Date: 2026-09-20
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Group level summary of the peak structure written by peak-structure.py.

    It reports, per modality and per condition, how many subjects show a
    second peak, where the first and the second peak sit, and how deep the
    trough between them is. It also compares the peak latency of the first
    peak with the latency that a plain argmax returns, because the argmax
    readout used by quick-slow-analysis.py jumps between the two peaks and
    that is what makes the paired test of MEG lose power.

Usage:
    python python/check-peak-structure.py
    python python/check-peak-structure.py -t rma --exclude-non-double
"""


# %% ---- 2026-09-20 ------------------------
# Requirements and constants
from scipy.stats import wilcoxon
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Group level summary of the peak structure')
parser.add_argument('-d', '--data-dir', default='output/peak-structure',
                    help='Folder written by peak-structure.py')
parser.add_argument('-t', '--tag', default='rma',
                    help='Short name of the input epochs')
parser.add_argument('--conditions', nargs='*',
                    default=['target', 'quick', 'slow'],
                    help='The conditions to summarize')
parser.add_argument('--exclude-non-double', action='store_true',
                    help='Keep only the rows with two peaks, which makes the '
                         'comparison of the two conditions rest on the same '
                         'set of subjects')
parser.add_argument('--early-peak1', type=float, default=0.22,
                    help='A first peak earlier than this is counted as the '
                         'early visual response rather than the target '
                         'component, it only affects the diagnostic column')
args = parser.parse_args()
TAG = args.tag

DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = DATA_DIR
SUMMARY_CSV = DATA_DIR / f'peak-structure-{TAG}.csv'
logger.info(f'Summarizing {SUMMARY_CSV=}')


# %% ---- 2026-09-20 ------------------------
# Function and class
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
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5 or np.allclose(a[mask], b[mask]):
        return np.nan, np.nan
    stat, p = wilcoxon(a[mask], b[mask])
    return float(stat), float(p)


def get(df: pd.DataFrame, mode: str, condition: str, column: str):
    '''
    Pull one column of one modality and condition as a float array.

    Args:
        df: The peak structure table
        mode: EEG | MEG
        condition: The condition name
        column: The column name

    Returns:
        values: The values, an empty array when the column or the selection is
            missing
    '''
    select = df[(df['mode'] == mode) & (df['condition'] == condition)]
    if column not in df.columns or len(select) == 0:
        return np.array([])
    return select[column].to_numpy(dtype=float)


# %% ---- 2026-09-20 ------------------------
# Play ground
if not SUMMARY_CSV.exists():
    logger.error(f'{SUMMARY_CSV} does not exist. Run 11.peak-structure.sh first.')
    raise SystemExit(1)

df = pd.read_csv(SUMMARY_CSV)
if args.exclude_non_double:
    n_before = len(df)
    df = df[df['n_peaks'] >= 2]
    logger.info(f'--exclude-non-double keeps {len(df)} of {n_before} rows')
display(df)

# ---- how often is the response double peaked ----
rows = []
for mode in ['EEG', 'MEG']:
    for condition in args.conditions:
        select = df[(df['mode'] == mode) & (df['condition'] == condition)]
        if len(select) == 0:
            continue
        n = len(select)
        n_double = int((select['n_peaks'] >= 2).sum())
        peak1 = select.get('peak1_t', pd.Series(dtype=float))
        # The search window starts at 0.05 s, so for a few subjects the first
        # peak lands on the early visual response instead of on the target
        # component. Those rows measure something else than the rest and are
        # counted here so that they can be left out by hand.
        n_early = int((peak1 < args.early_peak1).sum())
        rows.append(dict(
            mode=mode, condition=condition, n=n, n_double=n_double,
            frac_double=n_double / n,
            peak1_t=peak1.mean(),
            peak2_t=select.get('peak2_t', pd.Series(dtype=float)).mean(),
            dip_ratio=select.get('dip_ratio', pd.Series(dtype=float)).mean(),
            argmax_dominates_peak2=int(
                (select.get('dominant', pd.Series(dtype=float)) == 2).sum()),
            n_peak1_early=n_early))
incidence = pd.DataFrame(rows)
display(incidence)

# ---- is the first peak latency shifted by the reaction time ----
rows = []
for mode in ['EEG', 'MEG']:
    for column in ['peak1_t', 'peak2_t', 'argmax_t']:
        q = get(df, mode, 'quick', column)
        s = get(df, mode, 'slow', column)
        stat, p = paired(q, s)
        rows.append(dict(mode=mode, readout=column,
                         n=int((np.isfinite(q) & np.isfinite(s)).sum()),
                         quick=np.nanmean(q), slow=np.nanmean(s),
                         shift=np.nanmean(s) - np.nanmean(q),
                         statistic=stat, pvalue=p))
latency = pd.DataFrame(rows)
display(latency)

# ---- does the argmax agree with the first peak ----
rows = []
for mode in ['EEG', 'MEG']:
    for condition in ['quick', 'slow']:
        a = get(df, mode, condition, 'argmax_t')
        p1 = get(df, mode, condition, 'peak1_t')
        mask = np.isfinite(a) & np.isfinite(p1)
        rows.append(dict(
            mode=mode, condition=condition,
            n=int(mask.sum()),
            mean_abs_diff=float(np.mean(np.abs(a[mask] - p1[mask])))
            if mask.any() else np.nan,
            n_disagree=int((np.abs(a[mask] - p1[mask]) > 0.05).sum())))
agreement = pd.DataFrame(rows)
display(agreement)

# ---- figure ----
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for r, mode in enumerate(['EEG', 'MEG']):
    scale = 1e6 if mode == 'EEG' else 1e15

    # Panel 1: where the two peaks sit, quick versus slow
    ax = axes[r, 0]
    rng = np.random.default_rng(0)
    for i, condition in enumerate(['quick', 'slow']):
        for k, marker in [(1, 'o'), (2, 's')]:
            v = get(df, mode, condition, f'peak{k}_t')
            v = v[np.isfinite(v)]
            ax.scatter(i + rng.uniform(-.08, .08, v.size), v, marker=marker,
                       s=42, alpha=.8, label=f'{condition} peak{k}'
                       if r == 0 else None)
            if v.size:
                ax.hlines(v.mean(), i - .18, i + .18, color='k', lw=1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['quick', 'slow'])
    ax.set_ylabel('Peak latency (s)')
    ax.set_title(f'{mode}: first and second peak latency', fontsize=11)
    ax.legend(fontsize=8)

    # Panel 2: the argmax readout against the first peak
    ax = axes[r, 1]
    for i, condition in enumerate(['quick', 'slow']):
        a = get(df, mode, condition, 'argmax_t')
        p1 = get(df, mode, condition, 'peak1_t')
        p2 = get(df, mode, condition, 'peak2_t')
        n = min(a.size, p1.size)
        for j in range(n):
            if not (np.isfinite(a[j]) and np.isfinite(p1[j])):
                continue
            colour = 'tab:red' if abs(a[j] - p1[j]) > 0.05 else '0.7'
            ax.plot([i * 2, i * 2 + 1], [a[j], p1[j]], color=colour, lw=1.,
                    marker='o', ms=4)
        if p2.size:
            ax.scatter(i * 2 + 1 + rng.uniform(-.03, .03, p2.size), p2, s=10,
                       marker='s', color='0.5')
    ax.set_xticks([.5, 2.5])
    ax.set_xticklabels(['quick', 'slow'])
    ax.set_ylabel('Latency (s)')
    ax.set_title(f'{mode}: argmax vs peak1, red = argmax took peak2',
                 fontsize=11)
    ax.legend(handles=[plt.Line2D([], [], color='tab:red', lw=2,
                                  label='argmax != peak1')], fontsize=8)

    # Panel 3: how deep the trough between the two peaks is
    ax = axes[r, 2]
    for condition, colour in [('target', 'tab:blue'), ('quick', 'tab:cyan'),
                              ('slow', 'tab:orange'),
                              ('non-target', '0.5')]:
        v = get(df, mode, condition, 'dip_ratio')
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        ax.hist(v, bins=np.linspace(0, 1, 11), alpha=.5, label=condition,
                color=colour)
    ax.set_xlabel('Trough depth, dip / first peak')
    ax.set_ylabel('Subjects')
    ax.set_title(f'{mode}: dip depth, double peaked subjects only',
                 fontsize=11)
    ax.legend(fontsize=8)

fig.tight_layout()
fname = OUTPUT_DIR / f'group-peak-structure-{TAG}.png'
fig.savefig(fname, dpi=120)
plt.close(fig)
logger.info(f'Saved into {fname}')

for name, table in [('incidence', incidence), ('latency', latency),
                    ('agreement', agreement)]:
    fname = OUTPUT_DIR / f'group-peak-{name}-{TAG}.csv'
    table.to_csv(fname, index=False)
    logger.info(f'Saved into {fname}')


# %% ---- 2026-09-20 ------------------------
# Pending


# %% ---- 2026-09-20 ------------------------
# Pending
