"""
File: check-quick-slow-analysis.py
Author: Chuncheng Zhang
Date: 2026-09-18
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Group level summary of the quick versus slow target ERPs produced by
    quick-slow-analysis.py. For each modality it draws the mean +/- SEM of the
    global field power of the two groups, and reports paired statistics on the
    peak latency, the peak amplitude and the trial wise amplitude, plus the
    Spearman correlation between the single trial amplitude and the reaction
    time.

Usage:
    python python/check-quick-slow-analysis.py
    python python/check-quick-slow-analysis.py -t rma
"""


# %% ---- 2026-09-18 ------------------------
# Requirements and constants
from util.easy_imports import *
from scipy.stats import wilcoxon, ttest_1samp

# %%
parser = argparse.ArgumentParser(
    description='Group level summary of the quick versus slow analysis')
parser.add_argument('-d', '--data-dir', default='output/quick-slow',
                    help='Folder written by quick-slow-analysis.py')
parser.add_argument('-t', '--tag', default='rma',
                    help='Short name of the input epochs')
parser.add_argument('-w', '--win', type=float, nargs=2, default=[0.1, 0.8],
                    help='Time window in seconds to plot')
parser.add_argument('--exclude-weak', action='store_true',
                    help='Drop the subjects whose quick and slow groups have '
                         'almost the same reaction time')
args = parser.parse_args()
TAG = args.tag

DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = DATA_DIR
logger.info(f'Summarizing {DATA_DIR=}, {TAG=}')

SUMMARY_CSV = DATA_DIR / f'summary-{TAG}.csv'


# %% ---- 2026-09-18 ------------------------
# Function and class
def pick_data(info: mne.Info):
    '''
    The indices of the gradiometer / magnetometer / EEG channels.

    Args:
        info: MNE Info

    Returns:
        picks: Array of channel indices
    '''
    return mne.pick_types(info, meg=True, eeg=True, exclude='bads')


def global_field_power(data: np.ndarray):
    '''
    The root mean square deviation from the instantaneous spatial average.

    Args:
        data: (..., n_channels, n_times)

    Returns:
        gfp: (..., n_times)
    '''
    return np.sqrt(((data - data.mean(axis=-2, keepdims=True)) ** 2)
                   .mean(axis=-2))


def gfp_peak(evoked: mne.Evoked, picks: np.ndarray, window: tuple):
    '''
    The strongest global field power peak inside a time window.

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


def paired(a: np.ndarray, b: np.ndarray):
    '''
    The Wilcoxon signed rank test between two columns of the same subjects.

    Args:
        a: The values of the first condition
        b: The values of the second condition

    Returns:
        statistic, pvalue: nan when there are too few subjects
    '''
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5 or np.allclose(a[mask], b[mask]):
        return np.nan, np.nan
    stat, p = wilcoxon(a[mask], b[mask])
    return float(stat), float(p)


# %% ---- 2026-09-18 ------------------------
# Play ground
if not SUMMARY_CSV.exists():
    logger.error(
        f'{SUMMARY_CSV} does not exist. Run '
        f'10.quick-slow-analysis.sh first.')
    raise SystemExit(1)

df = pd.read_csv(SUMMARY_CSV)
if args.exclude_weak and 'weak' in df.columns:
    n_before = len(df)
    df = df[df['weak'] != 1].reset_index(drop=True)
    logger.info(f'Dropped {n_before - len(df)} subjects whose quick and slow '
                f'reaction times are too close, {len(df)} left')
display(df)

# The global field power curves are read back from the evoked of each subject
curves = dict()
for mode in ['EEG', 'MEG']:
    for group in ['quick', 'slow']:
        values, times = [], None
        for _, row in df[df['mode'] == mode].iterrows():
            fpath = (DATA_DIR / f'{mode}-{row["subject"]}'
                     / f'target-{TAG}-{group}-ave.fif')
            if not fpath.exists():
                logger.warning(f'{fpath} does not exist')
                continue
            evoked = mne.read_evokeds(fpath, verbose='ERROR')[0]
            picks = pick_data(evoked.info)
            gfp = global_field_power(evoked.data[picks])
            if times is None:
                times = evoked.times
            elif not np.array_equal(times, evoked.times):
                logger.warning(f'{mode}-{row["subject"]} has another time axis')
                continue
            values.append(gfp)
        if values:
            curves[(mode, group)] = (times, np.array(values))
            logger.debug(f'{mode} {group}: {len(values)} subjects')

if not curves:
    logger.error('There is no evoked to summarize')
    raise SystemExit(1)


# %% ---- 2026-09-18 ------------------------
# Figure
fig, axes = plt.subplots(2, 3, figsize=(21, 10))
stats_rows = []

for r, mode in enumerate(['EEG', 'MEG']):
    row_df = df[df['mode'] == mode].copy()
    scale = 1e6 if mode == 'EEG' else 1e15
    unit = 'uV' if mode == 'EEG' else 'fT'

    # Panel 1: the group mean of the global field power
    ax = axes[r, 0]
    for group, color in [('quick', 'tab:blue'), ('slow', 'tab:orange')]:
        if (mode, group) not in curves:
            continue
        times, values = curves[(mode, group)]
        mean = values.mean(axis=0) * scale
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(values)) * scale
        ax.plot(times, mean, color=color, label=group, lw=1.8)
        ax.fill_between(times, mean - sem, mean + sem,
                        color=color, alpha=0.15)
    ax.axvline(0, color='k', ls=':', lw=0.8)
    ax.axvspan(args.win[0], args.win[1], color='0.9', alpha=0.5, zorder=-1)
    ax.set_xlim(args.win[0] - 0.05, args.win[1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Global field power ({unit})')
    ax.set_title(f'{mode}: target after removal, quick versus slow')
    ax.legend()

    # Panel 2: paired peak latency
    ax = axes[r, 1]
    t_q = row_df['gfp_peak_t_quick'].to_numpy(dtype=float)
    t_s = row_df['gfp_peak_t_slow'].to_numpy(dtype=float)
    for i, subj in enumerate(row_df['subject']):
        ax.plot([1, 2], [t_q[i], t_s[i]], color='0.7', lw=0.8, marker='o',
                ms=3)
    # The tick labels are set below, so no labels kwarg here.
    # matplotlib >= 3.9 renamed labels to tick_labels, and >= 3.11 dropped
    # labels altogether, passing it breaks on the newer releases.
    ax.boxplot([t_q[np.isfinite(t_q)], t_s[np.isfinite(t_s)]],
               positions=[1, 2], showfliers=False, widths=0.4)
    stat, p = paired(t_q, t_s)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['quick', 'slow'])
    ax.set_ylabel('GFP peak latency (s)')
    ax.set_title(f'{mode}: peak latency, Wilcoxon p = {p:.3f}')
    stats_rows.append(dict(mode=mode, measure='gfp_peak_latency',
                           n=int(np.isfinite(t_q + t_s).sum()),
                           quick=np.nanmean(t_q), slow=np.nanmean(t_s),
                           statistic=stat, pvalue=p))

    # Panel 3: the correlation between trial amplitude and reaction time
    ax = axes[r, 2]
    rho = row_df['spearman_rho'].to_numpy(dtype=float)
    colors = np.where(rho < 0, 'tab:blue', 'tab:orange')
    ax.bar(range(len(rho)), rho, color=colors)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(len(rho)))
    ax.set_xticklabels(row_df['subject'].to_numpy(), rotation=90)
    mask = np.isfinite(rho)
    if mask.sum() >= 3:
        tstat, p_t = ttest_1samp(rho[mask], 0.)
    else:
        tstat, p_t = np.nan, np.nan
    ax.set_ylabel('Spearman rho (RT, amplitude)')
    ax.set_title(f'{mode}: rho across subjects, t-test p = {p_t:.3g}')
    stats_rows.append(dict(mode=mode, measure='spearman_rho',
                           n=int(mask.sum()), quick=np.nanmean(rho[mask]),
                           slow=np.nan, statistic=tstat, pvalue=p_t))

    # The peak amplitude comparison, reported but not plotted
    a_q = row_df['gfp_peak_a_quick'].to_numpy(dtype=float)
    a_s = row_df['gfp_peak_a_slow'].to_numpy(dtype=float)
    stat, p = paired(a_q, a_s)
    stats_rows.append(dict(mode=mode, measure='gfp_peak_amplitude',
                           n=int(np.isfinite(a_q + a_s).sum()),
                           quick=np.nanmean(a_q), slow=np.nanmean(a_s),
                           statistic=stat, pvalue=p))
    logger.info(f'{mode}: peak amplitude quick {np.nanmean(a_q):.3g} '
                f'versus slow {np.nanmean(a_s):.3g}, {p=:.3g}')

fig.tight_layout()
fname = OUTPUT_DIR / f'group-quick-slow-{TAG}.png'
fig.savefig(fname, dpi=120)
plt.close(fig)
logger.info(f'Saved into {fname}')

stats = pd.DataFrame(stats_rows)
fname = OUTPUT_DIR / f'group-summary-{TAG}.csv'
stats.to_csv(fname, index=False)
display(stats)
logger.info(f'Saved into {fname}')


# %% ---- 2026-09-18 ------------------------
# Pending


# %% ---- 2026-09-18 ------------------------
# Pending
