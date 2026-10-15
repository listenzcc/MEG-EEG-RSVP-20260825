"""
File: check-behavior.py
Author: Chuncheng Zhang
Date: 2026-10-15
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Group level description of the reaction times, for the behavioural section
    of the manuscript.

    Three things are read out of the trial tables, because all three constrain
    what the neural analysis is allowed to claim:

    How the reaction times are distributed. The median split into quick and
    slow trials throws away the shape of the distribution, and the shape is
    not normal here. If most of the mass sits on a handful of values the split
    is closer to a binary label than to a continuous measure, and any analysis
    that treats reaction time as continuous has to say so.

    How strongly the times are quantised. The most probable reaction time of a
    subject, and the share of trials that carry it, are reported per subject.
    A distribution that piles up on one value points at the event timestamps
    rather than at the subject, and it caps how finely a reaction time
    regression can resolve anything.

    Where the earliest press of each subject falls. That is the line the
    regression stage draws: before it, no response locked activity exists in
    the data at all, so an effect found there cannot be a movement artefact.

Usage:
    python python/check-behavior.py
    python python/check-behavior.py -o output/behavior
"""


# %% ---- 2026-10-15 ------------------------
# Requirements and constants
from scipy import stats
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Group level description of the reaction times')
parser.add_argument('-d', '--data-dir', default='output/quick-slow',
                    help='Folder containing the {MODE}-{SUBJ} folders written '
                         'by 10.quick-slow-analysis.sh')
parser.add_argument('-t', '--tag', default='rma',
                    help='Short name of the input epochs, used in filenames')
parser.add_argument('-o', '--output-dir', default='output/behavior',
                    help='Where the tables and the figure go')
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
TAG = args.tag
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    __IPYTHON__
    DATA_DIR = Path('../output/quick-slow')
    OUTPUT_DIR = Path('../output/behavior')
except NameError:
    pass

logger.info(f'Describing reaction times in {DATA_DIR=}, {TAG=}')

MODES = ['EEG', 'MEG']
SUBJECTS = sorted({p.name.split('-')[1] for p in DATA_DIR.glob('EEG-S*')
                   if p.is_dir()})
if not SUBJECTS:
    raise SystemExit(f'no {DATA_DIR}/EEG-S* folders, run '
                     f'10.quick-slow-analysis.sh')
logger.info(f'{len(SUBJECTS)} subjects')

# %% ---- 2026-10-15 ------------------------
# Per subject readouts
rows = []
curves = {m: {} for m in MODES}
for mode in MODES:
    for subj in SUBJECTS:
        f = DATA_DIR / f'{mode}-{subj}' / f'quick-slow-trials-{TAG}.csv'
        if not f.exists():
            logger.warning(f'{f} missing')
            continue
        t = pd.read_csv(f)
        rt = t['delay'].to_numpy(float)
        rt = rt[np.isfinite(rt)]
        if rt.size < 10:
            continue
        values, counts = np.unique(np.round(rt, 4), return_counts=True)
        modal_share = float(counts.max() / counts.sum())
        modal_rt = float(values[np.argmax(counts)])
        g = t.groupby('group').size() if 'group' in t else {}
        n_quick = int(g.get('quick', 0))
        n_slow = int(g.get('slow', 0))
        # How empty the stretch around the median split is
        med = float(np.median(rt))
        gap = float(np.mean((rt > med - 0.02) & (rt < med + 0.02)))
        rows.append(dict(
            mode=mode, subject=subj, n_trials=int(rt.size),
            n_quick=n_quick, n_slow=n_slow,
            rt_min=float(rt.min()), rt_q05=float(np.quantile(rt, 0.05)),
            rt_median=med, rt_iqr=float(stats.iqr(rt)),
            rt_max=float(rt.max()),
            n_unique=int(values.size), modal_rt=modal_rt,
            modal_share=modal_share,
            share_within_20ms_of_median=gap,
        ))
        curves[mode][subj] = rt

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT_DIR / 'group-behavior-summary.csv', index=False)

per_mode = []
for mode in MODES:
    s = summary[summary['mode'] == mode]
    if s.empty:
        continue
    per_mode.append(dict(
        mode=mode, n_subjects=len(s),
        n_trials_median=float(s['n_trials'].median()),
        rt_min=float(s['rt_min'].min()),
        rt_q05_min=float(s['rt_q05'].min()),
        rt_median=float(s['rt_median'].median()),
        modal_share_median=float(s['modal_share'].median()),
        n_unique_median=float(s['n_unique'].median()),
        quick_slow_ratio=float((s['n_quick'] / s['n_slow']).median()),
        share_within_20ms_of_median=float(
            s['share_within_20ms_of_median'].median()),
    ))
per_mode = pd.DataFrame(per_mode)
per_mode.to_csv(OUTPUT_DIR / 'group-behavior-by-mode.csv', index=False)
print(per_mode.to_string(index=False))

# %% ---- 2026-10-15 ------------------------
# Figure: distribution, quantisation, and the earliest press
n_mode = sum(1 for m in MODES if curves[m])
fig, axes = plt.subplots(n_mode, 3, figsize=(13, 3.1 * n_mode),
                         squeeze=False)
BIN_MS = 10
for r, mode in enumerate([m for m in MODES if curves[m]]):
    c = curves[mode]
    # (a) every subject's distribution
    ax = axes[r, 0]
    bins = np.arange(0.2, 1.0 + 1e-9, BIN_MS / 1000)
    for k, (subj, rt) in enumerate(sorted(c.items())):
        h, _ = np.histogram(rt, bins=bins)
        h = h / h.max()
        ax.plot(bins[:-1], h + k * 0.9, lw=0.8)
        ax.text(bins[0], k * 0.9 + 0.4, subj, fontsize=5)
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel('reaction time (s)')
    ax.set_ylabel('subject (normalised)')
    ax.set_title(f'(a) {mode}: reaction time distribution')

    # (b) how much mass sits on the most probable value
    ax = axes[r, 1]
    s = summary[summary['mode'] == mode].sort_values('modal_share')
    ax.barh(s['subject'], s['modal_share'], color='C0')
    ax.axvline(0.05, color='r', ls=':', lw=0.8)
    ax.set_xlabel('share of trials on the modal value')
    ax.set_title(f'(b) {mode}: quantisation of the timestamps')

    # (c) the line every subject draws: the earliest press
    ax = axes[r, 2]
    s = summary[summary['mode'] == mode]
    ax.bar(s['subject'], s['rt_min'], color='C1')
    ax.bar(s['subject'], s['rt_q05'] - s['rt_min'], bottom=s['rt_min'],
           color='C1', alpha=0.35)
    ax.set_ylabel('s')
    ax.set_title(f'(c) {mode}: earliest and 5th percentile press')
    ax.tick_params(axis='x', labelrotation=90, labelsize=6)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'group-rt-distribution.png', dpi=140)
plt.close(fig)

logger.info(f'wrote {OUTPUT_DIR}/group-behavior-summary.csv, '
            f'{OUTPUT_DIR}/group-behavior-by-mode.csv, '
            f'{OUTPUT_DIR}/group-rt-distribution.png')
