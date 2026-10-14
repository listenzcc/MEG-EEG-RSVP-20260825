"""
File: check-rt-sliding-decode.py
Author: Chuncheng Zhang
Date: 2026-10-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Group level summary of the quick against slow sliding decoding written by
    rt-sliding-decode.py.

    Stage 12 reports this contrast as a single number per subject, the AUC at
    the peak of the curve, and that number carries no test. Here the whole
    time course is averaged over subjects and tested, because the shape is the
    point: when does the epoch start to know how fast the subject will answer.

    Two tests are run over time. Pointwise one sample tests against 0.5 with a
    Benjamini-Hochberg correction answer "which single latencies are above
    chance". The cluster test is the one to quote: it flips the sign of every
    subject curve at random (exhaustively, 2**10 draws for ten subjects),
    builds clusters of neighbouring time points that pass the critical t, and
    compares the largest observed cluster mass with the permutation
    distribution. It is corrected for the number of time points by
    construction, and it does not need an assumption about which latency to
    look at, which matters here because the peak of the curve sits at the
    keypress latency and would be picked out by any window that was chosen
    after seeing it.

    Windows reported next to the peak of the 0 - 0.8 s stretch:

        pre-motor  0.10 - 0.20 s   before any keypress can have happened, so
                                   nothing here can come from the button press
        early      0.24 - 0.36 s   the first peak of the projected response
        late       0.41 - 0.55 s   the second peak, it overlaps the keypress

Usage:
    python python/check-rt-sliding-decode.py
    python python/check-rt-sliding-decode.py -t rma -d LR
"""


# %% ---- 2026-10-14 ------------------------
# Requirements and constants
import itertools

from scipy import stats
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Group level summary of the quick against slow sliding '
                'decoding')
parser.add_argument('-d', '--data-dir', default='output/rt-sliding-decode',
                    help='Folder containing the {MODE}-S*-{DECOD} folders')
parser.add_argument('-t', '--tag', default='rma',
                    help='Short name of the input epochs, used in filenames')
parser.add_argument('--decoding', default='LR',
                    help='Which classifier sub folder to read, LR | SVC')
parser.add_argument('-o', '--output-dir', default=None,
                    help='Where the group tables and figures go, default is '
                         'the data dir')
parser.add_argument('--win-p2', '--win_p2', dest='win_p2', type=float,
                    nargs=2, default=[0.10, 0.20],
                    help='Pre motor window, must match rt-sliding-decode.py. '
                         'If any subject answers earlier than win_p2[1] the '
                         'window is no longer motor free and the script says '
                         'so')
parser.add_argument('--win-early', '--win_early', dest='win_early',
                    type=float, nargs=2, default=[0.24, 0.36])
parser.add_argument('--win-late', '--win_late', dest='win_late',
                    type=float, nargs=2, default=[0.41, 0.55])
parser.add_argument('--peak-window', '--peak_window', dest='peak_window',
                    type=float, nargs=2, default=[0.0, 0.8],
                    help='The stretch the peak reading is taken over, must '
                         'match rt-sliding-decode.py')
parser.add_argument('--alpha', type=float, default=0.05,
                    help='Cluster forming threshold, the critical t of this '
                         'one sided level')
parser.add_argument('--n-perm', type=int, default=5000,
                    help='Permutation draws, used only when the exhaustive '
                         'sign flip is too large (more than 15 subjects)')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed of the permutation draw')
parser.add_argument('--crit-min', default='0.211',
                    help='Informative: the smallest reaction time of the '
                         'batch, used to check that the pre motor window '
                         'really precedes every keypress')
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
TAG = args.tag
DECODING = args.decoding
OUTPUT_DIR = Path(args.output_dir) if args.output_dir else DATA_DIR

# IPython compatibility
try:
    __IPYTHON__
    DATA_DIR = Path('../output/rt-sliding-decode')
    OUTPUT_DIR = DATA_DIR
except NameError:
    pass

logger.info(f'Summarizing {DATA_DIR=}, {TAG=}, {DECODING=}')

MODES = ['EEG', 'MEG']
WINDOWS = [('pre-motor', *args.win_p2), ('early', *args.win_early),
           ('late', *args.win_late)]
PEAK_WINDOW = tuple(args.peak_window)


# %% ---- 2026-10-14 ------------------------
# Function and class
def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    '''
    Benjamini-Hochberg adjusted p values.

    Args:
        pvalues: (n,) raw p values

    Returns:
        (n,) adjusted p values, in the order of the input
    '''
    p = np.asarray(pvalues, float)
    ok = np.isfinite(p)
    adj = np.full(p.shape, np.nan)
    if not ok.any():
        return adj
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj[order] = np.clip(ranked, 0, 1)
    return adj


def window_mean(curve: np.ndarray, times: np.ndarray,
                window: tuple) -> np.ndarray:
    '''
    Mean of a curve, or of a stack of curves, inside one window.

    Args:
        curve: (n_times,) or (n_subjects, n_times)
        times: (n_times,)
        window: (lo, hi) in seconds

    Returns:
        The mean over the window, (...,)
    '''
    mask = (times >= window[0]) & (times <= window[1])
    if not mask.any():
        return np.full(curve.shape[:-1], np.nan)
    return curve[..., mask].mean(axis=-1)


def cluster_permutation(curves: np.ndarray, times: np.ndarray,
                       alpha: float, n_perm: int,
                       seed: int = 0):
    '''
    One sample cluster test over time, by exhaustive or Monte Carlo sign
    flipping.

    Under the null that the response speed is not decodable, the sign of the
    deviance of one subject curve from chance is exchangeable, so every
    assignment of signs to the subject curves is equally likely. Time points
    whose t statistic passes the critical level are gathered into clusters of
    neighbouring time points and the mass of a cluster is the sum of its t
    values. A cluster is significant when its mass beats 95% of the largest
    masses seen in the permuted data sets, which corrects for the number of
    time points without ever choosing a latency by hand.

    Args:
        curves: (n_subjects, n_times) AUC values of one modality
        times: (n_times,)
        alpha: One sided level of the cluster forming threshold
        n_perm: Number of random draws, used when the exhaustive set is large
        seed: Seed of the random draws

    Returns:
        clusters: A list of dicts, one per observed cluster, with the p value
        curve: A dict with the observed t, the critical t and the pointwise p
    '''
    d = curves - 0.5
    n, n_times = d.shape
    df = n - 1
    t_crit = float(stats.t.ppf(1 - alpha, df))

    t_obs = d.mean(axis=0) / (d.std(axis=0, ddof=1) / np.sqrt(n))
    p_point = stats.t.sf(t_obs, df)
    p_adj = bh_fdr(p_point)

    def cluster_mass(t: np.ndarray):
        out = []
        above = t > t_crit
        i = 0
        while i < n_times:
            if above[i]:
                j = i
                while j + 1 < n_times and above[j + 1]:
                    j += 1
                out.append((i, j, float(t[i:j + 1].sum())))
                i = j + 1
            else:
                i += 1
        return out

    observed = cluster_mass(t_obs)
    max_obs = max([c[2] for c in observed], default=0.0)

    # 2**n sign assignments, exhaustively when that is affordable
    if 2 ** n <= 20000:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)),
                         dtype=float)
        method = f'exhaustive {2 ** n} sign flips'
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        method = f'{n_perm} random sign flips'
        logger.warning(f'{n} subjects, the exhaustive sign flip would need '
                       f'{2 ** n} draws, using {method}')

    null = np.empty(len(signs))
    for i, s in enumerate(signs):
        ds = d * s[:, None]
        t_p = ds.mean(axis=0) / (ds.std(axis=0, ddof=1) / np.sqrt(n))
        null[i] = max([c[2] for c in cluster_mass(t_p)], default=0.0)

    clusters = []
    for i, j, mass in observed:
        p = float(((null >= mass).sum() + 1) / (len(null) + 1))
        clusters.append(dict(
            t_start=round(float(times[i]), 4),
            t_end=round(float(times[j]), 4),
            peak_t=round(float(times[i + int(np.argmax(t_obs[i:j + 1]))]), 4),
            peak_t_value=round(float(t_obs[i:j + 1].max()), 3),
            mass=round(mass, 3),
            n_times=int(j - i + 1),
            cluster_p=p))
    logger.info(f'cluster test by {method}, critical t {t_crit:.3f}')

    return clusters, dict(t=t_obs, t_crit=t_crit, p_point=p_point,
                          p_adj=p_adj, null=null, n_sign_flip=len(null))


# %% ---- 2026-10-14 ------------------------
# Play ground
# ---- read the per subject curves ----
curves = dict()
times = None
subjects = dict()

for mode in MODES:
    stack, names = [], []
    for folder in sorted(DATA_DIR.glob(f'{mode}-S*-{DECODING}')):
        f = folder / f'curves-{TAG}.csv'
        if not f.exists():
            logger.warning(f'{f} does not exist, skipped')
            continue
        df_c = pd.read_csv(f)
        t = df_c['time'].to_numpy(dtype=float)
        if times is None:
            times = t
        elif t.size != times.size or not np.allclose(t, times, atol=1e-6):
            logger.warning(f'{f} has another time axis, skipped')
            continue
        stack.append(df_c['auc_mean'].to_numpy(dtype=float))
        names.append(folder.name)
    if not stack:
        logger.warning(f'{mode}: no subject curve was found')
        continue
    curves[mode] = np.array(stack)
    subjects[mode] = names
    logger.info(f'{mode}: {len(stack)} subjects, {times.size} time points')
    # A window that is not motor free would make the pre-motor reading moot
    for folder in sorted(DATA_DIR.glob(f'{mode}-S*-{DECODING}')):
        f = folder / f'trials-{TAG}.csv'
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if 'delay' in d.columns and len(d):
            rt = float(d['delay'].min())
            if rt <= args.win_p2[1]:
                logger.warning(f'{folder.name}: the fastest reaction time is '
                               f'{rt:.3f} s, which is inside the pre-motor '
                               f'window {args.win_p2}')

if not curves:
    raise FileNotFoundError(f'No curves-{TAG}.csv found under {DATA_DIR}')

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%
# ---- readings and tests ----
summary, stats_rows, cluster_rows = [], [], []
perm = dict()

for mode in MODES:
    if mode not in curves:
        continue
    data = curves[mode]
    n = data.shape[0]
    m = data.mean(axis=0)
    sem = data.std(axis=0, ddof=1) / np.sqrt(n)

    mask_peak = ((times >= PEAK_WINDOW[0]) & (times <= PEAK_WINDOW[1]))
    i_peak = int(np.argmax(m[mask_peak]))
    row = dict(mode=mode, n_subjects=n, decoding=DECODING,
               peak_t=round(float(times[mask_peak][i_peak]), 4),
               peak_auc=round(float(m[mask_peak][i_peak]), 4),
               peak_auc_sem=round(float(sem[mask_peak][i_peak]), 4),
               auc_mean=round(float(m.mean()), 4))
    for name, lo, hi in WINDOWS:
        v = window_mean(m, times, (lo, hi))
        row[f'auc_{name}'] = round(float(v), 4)
    summary.append(row)

    for name, lo, hi in WINDOWS + [('peak', *PEAK_WINDOW)]:
        v = window_mean(data, times, (lo, hi))
        if np.all(v == v[0]):
            logger.warning(f'{mode} {name}: every subject curve is flat')
            continue
        t = stats.ttest_1samp(v, 0.5)
        w = stats.wilcoxon(v - 0.5)
        stats_rows.append(dict(
            mode=mode, window=name, lo=lo, hi=hi, n_subjects=n,
            mean_auc=round(float(np.mean(v)), 4),
            sd_auc=round(float(np.std(v, ddof=1)), 4),
            median_auc=round(float(np.median(v)), 4),
            n_subject_above_chance=int((v > 0.5).sum()),
            ttest_p=float(f'{t.pvalue:.3g}'),
            wilcoxon_p=float(f'{w.pvalue:.3g}')))

    clusters, prm = cluster_permutation(
        data, times, args.alpha, args.n_perm, args.seed)
    perm[mode] = prm
    logger.info(f'{mode}: {len(clusters)} cluster(s) above the critical t')
    for c in clusters:
        cluster_rows.append(dict(mode=mode, n_subjects=n, **c))
        logger.info(f'  {c["t_start"]:.3f} - {c["t_end"]:.3f} s, '
                    f'peak {c["peak_t"]:.3f} s t={c["peak_t_value"]:.2f}, '
                    f'mass {c["mass"]:.1f}, p={c["cluster_p"]:.4f}')

df = pd.DataFrame(summary)
fname = OUTPUT_DIR / f'group-rt-sliding-summary-{TAG}.csv'
df.to_csv(fname, index=False)
logger.info(f'Saved into {fname}')
display(df)

df_stats = pd.DataFrame(stats_rows)
fname = OUTPUT_DIR / f'group-rt-sliding-windows-{TAG}.csv'
df_stats.to_csv(fname, index=False)
logger.info(f'Saved into {fname}')
display(df_stats)

if cluster_rows:
    df_cluster = pd.DataFrame(cluster_rows)
    fname = OUTPUT_DIR / f'group-rt-sliding-clusters-{TAG}.csv'
    df_cluster.to_csv(fname, index=False)
    logger.info(f'Saved into {fname}')
    display(df_cluster)
else:
    logger.warning(f'No time point of any modality passes the cluster forming '
                   f'threshold t={args.alpha}, the curve is flat everywhere')

# %%
# ---- figures ----
fig, axes = plt.subplots(2, len([m for m in MODES if m in curves]),
                         figsize=(7.2 * len([m for m in MODES if m in curves]),
                                  8.4), sharex=True, squeeze=False)

for col, mode in enumerate([m for m in MODES if m in curves]):
    data = curves[mode]
    n = data.shape[0]
    m = data.mean(axis=0)
    sem = data.std(axis=0, ddof=1) / np.sqrt(n)

    ax = axes[0, col]
    ax.axvspan(*args.win_p2, color='tab:green', alpha=0.12,
               label='before any keypress can be')
    ax.axvspan(*args.win_early, color='tab:orange', alpha=0.08)
    ax.axvspan(*args.win_late, color='tab:purple', alpha=0.08)
    # The significant clusters, shaded on top of the windows
    for c in cluster_rows:
        if c['mode'] == mode and c['cluster_p'] < args.alpha:
            ax.axvspan(c['t_start'], c['t_end'], color='#d62728', alpha=0.16,
                       label='cluster p < %.3f' % args.alpha)
    ax.plot(times, m, lw=1.8, color='#1f77b4',
            label=f'quick vs slow (n={n})')
    ax.fill_between(times, m - sem, m + sem, color='#1f77b4', alpha=0.18)
    ax.axhline(0.5, color='gray', lw=0.9, ls=':')
    ax.axvline(0, color='k', lw=0.9, ls='--', alpha=0.5)
    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(0.4, 0.85)
    ax.set_title(f'{mode}: quick versus slow inside the target trials')
    ax.set_ylabel('ROC AUC')
    ax.legend(fontsize=8, loc='upper left')

    ax = axes[1, col]
    prm = perm[mode]
    for c in cluster_rows:
        if c['mode'] == mode and c['cluster_p'] < args.alpha:
            ax.axvspan(c['t_start'], c['t_end'], color='#d62728', alpha=0.16)
    ax.axvspan(*args.win_p2, color='tab:green', alpha=0.12)
    ax.axvspan(*args.win_early, color='tab:orange', alpha=0.08)
    ax.axvspan(*args.win_late, color='tab:purple', alpha=0.08)
    ax.plot(times, prm['t'], lw=1.6, color='#2ca02c')
    ax.axhline(prm['t_crit'], color='#d62728', lw=1.0, ls='--',
               label=f'critical t at alpha {args.alpha:g}')
    ax.axhline(0, color='gray', lw=0.9, ls=':')
    ax.axvline(0, color='k', lw=0.9, ls='--', alpha=0.5)
    ax.set_xlim(-0.2, 1.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('t across subjects')
    ax.legend(fontsize=8, loc='upper left')

fig.tight_layout()
fname = OUTPUT_DIR / f'group-rt-sliding-{TAG}.png'
fig.savefig(fname, dpi=130)
plt.close(fig)
logger.info(f'Saved into {fname}')

# %%
# A last word on how far the conclusion can go, printed rather than written
# into a table, because it is a caveat and not a result.
logger.info(f'Read the peak together with the keypress: the peak of the curve '
            f'sits at the reaction time, and the projected epochs still hold '
            f'the single trial part of the button response. The pre-motor '
            f'window {args.win_p2} is the reading that carries the claim, '
            f'every subject of this batch answers later than {args.crit_min} s')


# %% ---- 2026-10-14 ------------------------
# Pending
