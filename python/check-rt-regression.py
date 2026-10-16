"""
File: check-rt-regression.py
Author: Chuncheng Zhang
Date: 2026-10-15
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Group level summary of the continuous reaction time regression written by
    rt-regression.py.

    rt-regression.py fits, at every time point and across trials, a design that
    holds a constant, the centred reaction time, and a set of bins for the
    response locked to the keypress. The coefficient on reaction time is the
    quantity of interest: how much of the target response scales with how fast
    the subject will answer, once the response locked to the press itself is
    in the model.

    The one structural fact that decides how far this can be read is the
    earliest press of the subject. Before it every button bin is identically
    zero, so the reaction time coefficient is a clean estimate of a stimulus
    locked modulation. After it the button basis is alive, and because reaction
    time is by definition the interval between the target and the press, the
    basis can partly reproduce the reaction time column itself. The two then
    describe the same variance and the coefficient stays unbiased but loses
    precision, which is a property of the design and not of the amount of data.

    Everything here is therefore reported separately on the two sides of that
    line, and the line is taken from the trials of each subject rather than
    assumed:

        clean   times < the smallest reaction time of that subject
        mixed   times after it

    Three group level outputs:

        summary    one row per subject, the earliest press, whether each side
                   was identifiable, and the recovery of the simulated effect
        windows    the pre specified windows, with a one sample test against
                   zero on the clean side of each window only
        clusters   a sign flip cluster permutation over the clean stretch,
                   which is corrected for the number of time points by
                   construction and needs no assumption about which latency
                   to look at

    A window that is not entirely on the clean side of every subject is
    reported with the number of trials that have already pressed inside it,
    and is not tested. That is the rule that keeps the report honest: the
    association after the press exists, it is simply not attributable.

Usage:
    python python/check-rt-regression.py
    python python/check-rt-regression.py -t norp
"""


# %% ---- 2026-10-15 ------------------------
# Requirements and constants
import itertools

from scipy import stats
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Group level summary of the continuous reaction time '
                'regression')
parser.add_argument('-d', '--data-dir', default='output/rt-regression',
                    help='Folder containing the {MODE}-S* folders')
parser.add_argument('-t', '--tag', default='rma',
                    help='Short name of the input epochs, used in filenames')
parser.add_argument('-o', '--output-dir', default=None,
                    help='Where the group tables and figures go, default is '
                         'the data dir')
parser.add_argument('--win-pre', '--win_pre', dest='win_pre', type=float,
                    nargs=2, default=[0.10, 0.20],
                    help='Pre motor window, must match rt-regression.py')
parser.add_argument('--win-early', '--win_early', dest='win_early',
                    type=float, nargs=2, default=[0.24, 0.36])
parser.add_argument('--win-late', '--win_late', dest='win_late', type=float,
                    nargs=2, default=[0.41, 0.55])
parser.add_argument('--clean-from', '--clean_from', dest='clean_from',
                    type=float, default=0.05,
                    help='Lower bound of the stretch used for the cluster '
                         'test, before this there is no target on screen')
parser.add_argument('--alpha', type=float, default=0.05,
                    help='Cluster forming threshold, the critical t of this '
                         'two sided level')
parser.add_argument('--n-perm', type=int, default=5000,
                    help='Permutation draws, used only when the exhaustive '
                         'sign flip is too large (more than 15 subjects)')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed of the permutation draw')
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
TAG = args.tag
OUTPUT_DIR = Path(args.output_dir) if args.output_dir else DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    __IPYTHON__
    DATA_DIR = Path('../output/rt-regression')
    OUTPUT_DIR = DATA_DIR
except NameError:
    pass

logger.info(f'Summarizing {DATA_DIR=}, {TAG=}')

MODES = ['EEG', 'MEG']
WINDOWS = [('pre-motor', *args.win_pre), ('early', *args.win_early),
           ('late', *args.win_late)]


def load_subject(mode: str, subj: str) -> dict:
    '''
    Read one subject of rt-regression.py output.
    '''
    d = DATA_DIR / f'{mode}-{subj}'
    out = dict(mode=mode, subject=subj, found=False)
    f_curve = d / f'beta-curve-{TAG}.csv'
    f_audit = d / f'audit-{TAG}.csv'
    if not f_curve.exists():
        return out
    out['curve'] = pd.read_csv(f_curve)
    if f_audit.exists():
        a = pd.read_csv(f_audit).iloc[0].to_dict()
        out['audit'] = a
        out['clean_until'] = float(a.get('clean_until_s', np.nan))
    out['found'] = True
    return out


def cluster_test(curves: np.ndarray, times: np.ndarray, alpha: float,
                 rng: np.random.Generator) -> dict:
    '''
    Sign flip cluster permutation on the group mean curve.

    Every subject curve is flipped or not at random, the one sample t is taken
    against zero, contiguous runs above the critical value are summed into
    cluster masses, and the largest observed mass is compared with the largest
    mass of each draw. The sign flip is exact for the mean of a symmetric
    statistic and needs no assumption about the distribution of the curves.

    Args:
        curves: (n_subj, n_times) one curve per subject
        times: (n_times,) time axis
        alpha: two sided level of the cluster forming threshold
        rng: random generator

    Returns:
        out: the observed clusters and their permutation p values
    '''
    n_subj = curves.shape[0]
    crit = float(stats.t.ppf(1 - alpha / 2, n_subj - 1))

    def masses(x: np.ndarray):
        t = stats.ttest_1samp(x, 0.0, axis=0).statistic
        t = np.nan_to_num(t, nan=0.0)
        over = np.abs(t) >= crit
        out = []
        i = 0
        while i < len(over):
            if over[i]:
                j = i
                while j + 1 < len(over) and over[j + 1]:
                    j += 1
                out.append((i, j, float(np.nansum(np.abs(t[i:j + 1])))))
                i = j + 1
            else:
                i += 1
        return out, t

    obs, t_obs = masses(curves)
    if not obs:
        return dict(n_cluster=0, crit_t=crit)

    # Exhaustive sign flip while it is cheap, random draws beyond that
    n_flip = 2 ** n_subj
    if n_flip <= 4096:
        flips = np.array(list(itertools.product([-1, 1], repeat=n_subj)))
    else:
        flips = rng.choice([-1.0, 1.0], size=(args.n_perm, n_subj))
    null_max = []
    for f in flips:
        m, _ = masses(curves * f[:, None])
        null_max.append(max([c[2] for c in m]) if m else 0.0)
    null_max = np.array(null_max)

    rows = []
    for i, j, mass in obs:
        p = float((null_max >= mass).mean())
        rows.append(dict(t_lo=float(times[i]), t_hi=float(times[j]),
                         n_points=int(j - i + 1), mass=mass, p_perm=p))
    rows.sort(key=lambda r: r['mass'], reverse=True)
    return dict(n_cluster=len(rows), crit_t=crit, clusters=rows,
                n_perm=len(null_max))


# %% ---- 2026-10-15 ------------------------
# Load and align
SUBJECTS = sorted({p.name.split('-')[1]
                   for p in DATA_DIR.glob('EEG-S*') if p.is_dir()})
if not SUBJECTS:
    raise SystemExit(f'no {DATA_DIR}/EEG-S* folders, run 20.rt-regression.sh')
logger.info(f'{len(SUBJECTS)} subjects: {", ".join(SUBJECTS)}')

loaded = {(m, s): load_subject(m, s) for m in MODES for s in SUBJECTS}
missing = [f'{m}-{s}' for (m, s), v in loaded.items() if not v['found']]
if missing:
    logger.warning(f'{len(missing)} subject folders missing: '
                   f'{", ".join(missing)}')

# The batch line: no subject has pressed before this
clean_until = {k: v.get('clean_until', np.nan) for k, v in loaded.items()
               if v['found']}
if clean_until:
    logger.info('earliest press per subject, s: '
                + ', '.join(f'{m[0]}{s[-2:]} {clean_until[(m, s)]:.3f}'
                            for m in MODES for s in SUBJECTS
                            if (m, s) in clean_until))
    batch_earliest = float(np.nanmin(list(clean_until.values())))
    batch_latest_of_earliest = float(np.nanmax(list(clean_until.values())))
    logger.info(f'batch earliest press {batch_earliest:.3f} s, the slowest '
                f'first press of any subject {batch_latest_of_earliest:.3f} s')
else:
    batch_earliest = batch_latest_of_earliest = np.nan

# %% ---- 2026-10-15 ------------------------
# Per subject summary
rows = []
for (mode, subj), v in loaded.items():
    if not v['found']:
        continue
    a = v.get('audit', {})
    cur = v['curve']
    rows.append(dict(
        mode=mode, subject=subj,
        n_trials=a.get('n_trials', np.nan),
        rt_median_s=a.get('rt_median', np.nan),
        rt_n_unique=a.get('rt_n_unique', np.nan),
        rt_modal_share=a.get('rt_modal_share', np.nan),
        clean_until_s=a.get('clean_until_s', np.nan),
        clean_identifiable=a.get('recovery_clean_identifiable', np.nan),
        mixed_identifiable=a.get('recovery_mixed_identifiable', np.nan),
        gain_clean=a.get('recovery_recovery_gain_clean', np.nan),
        corr_clean=a.get('recovery_recovery_corr_clean', np.nan),
        gain_mixed=a.get('recovery_recovery_gain_mixed', np.nan),
        corr_mixed=a.get('recovery_recovery_corr_mixed', np.nan),
        leak_over_null_clean=a.get('recovery_leak_over_null_clean', np.nan),
        leak_over_null_mixed=a.get('recovery_leak_over_null_mixed', np.nan),
    ))
summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT_DIR / f'group-rt-regression-summary-{TAG}.csv',
               index=False)

# %% ---- 2026-10-15 ------------------------
# Group curves and window tests
times = None
curves = {m: [] for m in MODES}
for mode in MODES:
    for subj in SUBJECTS:
        v = loaded[(mode, subj)]
        if not v['found']:
            continue
        cur = v['curve']
        if times is None:
            times = cur['time'].to_numpy(float)
        curves[mode].append(cur['beta_rt_proj'].to_numpy(float))
curves = {m: np.array(c) for m, c in curves.items() if c}

win_rows = []
for mode in MODES:
    if mode not in curves or not len(curves[mode]):
        continue
    C = curves[mode]
    for wname, lo, hi in WINDOWS:
        m = (times >= lo) & (times <= hi)
        if m.sum() == 0:
            continue
        # How many subjects still have a clean window here
        clean_ok = sum(1 for s in SUBJECTS
                       if np.isfinite(clean_until.get((mode, s), np.nan))
                       and hi < clean_until[(mode, s)])
        vals = np.nanmean(C[:, m], axis=1)
        if clean_ok == len(curves[mode]):
            t_, p_ = stats.ttest_1samp(vals, 0.0)
            w_, pw_ = stats.wilcoxon(vals)
            testable = True
        else:
            t_ = p_ = w_ = pw_ = np.nan
            testable = False
        win_rows.append(dict(
            mode=mode, window=wname, win_lo=lo, win_hi=hi,
            n_subjects=int(np.sum(np.isfinite(vals))),
            n_subjects_clean=clean_ok,
            mean_beta=float(np.nanmean(vals)),
            median_beta=float(np.nanmedian(vals)),
            t=float(t_) if testable else np.nan,
            p_t=float(p_) if testable else np.nan,
            p_wilcoxon=float(pw_) if testable else np.nan,
            n_above_zero=int(np.nansum(vals > 0)),
            testable=testable,
            note=('' if testable else
                  f'{len(curves[mode]) - clean_ok}/{len(curves[mode])} '
                  f'subjects have already pressed before {hi:.2f} s')))
windows = pd.DataFrame(win_rows)
windows.to_csv(OUTPUT_DIR / f'group-rt-regression-windows-{TAG}.csv',
               index=False)

# %% ---- 2026-10-15 ------------------------
# Cluster permutation on the clean stretch only
rng = np.random.default_rng(args.seed)
cluster_rows = []
cluster_top = {}
for mode in MODES:
    if mode not in curves or not len(curves[mode]):
        continue
    limit = float(np.nanmin([clean_until[(mode, s)] for s in SUBJECTS
                             if (mode, s) in clean_until]))
    m = (times >= args.clean_from) & (times < limit)
    if m.sum() < 5:
        logger.warning(f'{mode}: only {m.sum()} clean time points, no '
                       f'cluster test')
        continue
    res = cluster_test(curves[mode][:, m], times[m], args.alpha, rng)
    cluster_top[mode] = res
    logger.info(f'{mode}: clean stretch {args.clean_from}-{limit:.3f} s, '
                f'{m.sum()} points, crit |t| {res["crit_t"]:.2f}, '
                f'{res["n_cluster"]} cluster(s)')
    for c in res.get('clusters', []):
        cluster_rows.append(dict(mode=mode, **c))
    if not res.get('clusters'):
        cluster_rows.append(dict(mode=mode, t_lo=np.nan, t_hi=np.nan,
                                 n_points=0, mass=0.0, p_perm=np.nan))
clusters = pd.DataFrame(cluster_rows)
clusters.to_csv(OUTPUT_DIR / f'group-rt-regression-clusters-{TAG}.csv',
                index=False)

# %% ---- 2026-10-16 ------------------------
# Figure. One curve panel per modality plus, below it, the per subject window
# means, so the sign consistency behind the window tests can be seen directly.
plot_modes = [m for m in MODES if m in curves and len(curves[m])]
fig, axes = plt.subplots(2 * len(plot_modes), 1,
                         figsize=(9, 2.5 * len(plot_modes) + 1.4),
                         squeeze=False)
for row, mode in enumerate(plot_modes):
    ax = axes[2 * row, 0]
    C = curves[mode]
    mean = np.nanmean(C, axis=0)
    sem = stats.sem(C, axis=0, nan_policy='omit')
    ax.fill_between(times, mean - sem, mean + sem, color='C0', alpha=0.25)
    ax.plot(times, mean, color='C0', lw=1.6,
            label=f'{mode} (n={C.shape[0]})')
    ax.axhline(0, color='k', lw=0.6)

    limit = float(np.nanmin([clean_until[(mode, s)] for s in SUBJECTS
                             if (mode, s) in clean_until]))
    ax.axvspan(times[0], limit, color='green', alpha=0.06)
    ax.axvspan(limit, times[-1], color='red', alpha=0.06)
    ax.axvline(limit, color='r', lw=0.9, ls='--')
    ax.annotate(f'earliest press {limit:.2f} s', xy=(limit, ax.get_ylim()
                                                          [1] * 0.9),
                fontsize=7, color='r')

    for wname, lo, hi in WINDOWS:
        ax.axvspan(lo, hi, color='grey', alpha=0.12)
        ax.text((lo + hi) / 2, ax.get_ylim()[0], wname, fontsize=6,
                ha='center', va='bottom', color='grey')

    res = cluster_top.get(mode, {})
    for c in res.get('clusters', []):
        if c['p_perm'] < 0.05:
            ax.plot([c['t_lo'], c['t_hi']],
                    [ax.get_ylim()[1] * 0.95] * 2, color='k', lw=2.5)
            ax.text((c['t_lo'] + c['t_hi']) / 2, ax.get_ylim()[1] * 0.96,
                    f"p={c['p_perm']:.3f}", fontsize=6, ha='center')

    ax.set_ylabel('beta RT along the target topography\n'
                  '(fraction of target average per SD of RT)')
    ax.set_title(f'{mode}: reaction time coefficient, {TAG}\n'
                 f'green: before any press, red: press already possible')
    ax.legend(fontsize=7, loc='upper left')

    # Per subject window means, horizontal bars ordered by value
    axb = axes[2 * row + 1, 0]
    wvals = []
    for s in SUBJECTS:
        cur = loaded[(mode, s)].get('curve')
        if cur is None:
            continue
        m = ((cur['time'] >= args.win_pre[0])
             & (cur['time'] <= args.win_pre[1]))
        wvals.append((s, float(np.nanmean(cur['beta_rt_proj'][m]))))
    wvals.sort(key=lambda x: x[1])
    names = [w[0] for w in wvals]
    vals = np.array([w[1] for w in wvals])
    colors = ['C3' if v < 0 else 'C0' for v in vals]
    axb.barh(range(len(vals)), vals, color=colors, height=0.62)
    axb.set_yticks(range(len(vals)))
    axb.set_yticklabels(names, fontsize=6)
    axb.axvline(0, color='k', lw=0.6)
    med = float(np.nanmedian(vals))
    axb.axvline(med, color='k', lw=0.9, ls=':')
    axb.annotate(f'median {med:+.3f}', xy=(med, len(vals) - 0.3), fontsize=7)
    axb.set_ylabel(f'{mode} pre-motor window mean')
    if row == 0:
        axb.set_title('per subject mean of beta RT in the pre-motor window '
                      '0.10-0.20 s')
axes[-1, 0].set_xlabel('time from target onset (s)')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / f'group-rt-regression-{TAG}.png', dpi=130)
plt.close(fig)

logger.info(f'wrote {OUTPUT_DIR}/group-rt-regression-{{summary,windows,'
            f'clusters}}-{TAG}.csv and group-rt-regression-{TAG}.png')
