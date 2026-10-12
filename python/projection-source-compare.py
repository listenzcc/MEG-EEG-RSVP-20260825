"""
File: projection-source-compare.py
Author: Chuncheng Zhang
Date: 2026-10-12
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Compare the target ERP under four projection conditions, to test whether
    the ~0.4 s component is removed specifically by the movement-related
    subspace or by any spatial projection at all.

    Conditions (see 17.projection-split.sh):
      none      epochs-{evt}-notch-epo.fif                        (unprojected)
      visual    epochs-{evt}-notch-removal-visual-epo.fif         (proj2 only)
      keypress  epochs-{evt}-notch-removal-keypress-epo.fif       (proj3 only)
      both      epochs-{evt}-notch-removal-artificial-epo.fif     (the current pipeline)

    proj2 and proj3 are estimated with identical kwargs on data from the same
    sessions, so 'visual' and 'keypress' are matched in the number of components.
    The reading that matters: if the 0.4 s peak survives 'visual' but collapses
    under 'keypress', the attribution is supported by a direct contrast instead
    of by the fact that the pre-projection peak sits over sensorimotor areas.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-10-12 ------------------------
# Requirements and constants
from scipy import stats
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Compare projection sources on the target ERP')
parser.add_argument('-m', '--mode', default='EEG', help='Mode name EEG | MEG')
parser.add_argument('--evt', default='1', help='Event to analyse, 1 | 2 | 3')
parser.add_argument('--peak-window', '--peak_window', dest='peak_window',
                    default='0.30,0.55',
                    help='Window (s) for reading the 0.4 s component, '
                         'comma separated lo,hi')
parser.add_argument('--l-freq', '--l_freq', dest='l_freq', type=float,
                    default=0.1, help='Low cut of the display filter')
parser.add_argument('--h-freq', '--h_freq', dest='h_freq', type=float,
                    default=5.0, help='High cut of the display filter')
args = parser.parse_args()
MODE = args.mode
EVT = args.evt
PEAK_LO, PEAK_HI = [float(e) for e in args.peak_window.split(',')]
L_FREQ, H_FREQ = args.l_freq, args.h_freq

logger.info(
    f'Start with {MODE=}, {EVT=}, peak window=({PEAK_LO}, {PEAK_HI}), '
    f'filter=({L_FREQ}, {H_FREQ})')

# %%
DATA_DIR = Path('output/epochs')
OUTPUT_DIR = Path('output/projection-source-compare')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIT = 1e15 if MODE == 'MEG' else 1e6
UNIT_NAME = 'fT' if MODE == 'MEG' else 'uV'

CONDITIONS = [
    dict(tag='none', display='no projection', color='0.35',
         stem=f'epochs-{EVT}-notch'),
    dict(tag='visual', display='visual only', color='#1f77b4',
         stem=f'epochs-{EVT}-notch-removal-visual'),
    dict(tag='keypress', display='keypress only', color='#d62728',
         stem=f'epochs-{EVT}-notch-removal-keypress'),
    dict(tag='both', display='visual + keypress', color='#2ca02c',
         stem=f'epochs-{EVT}-notch-removal-artificial'),
]


# %% ---- 2026-10-12 ------------------------
# Function and class
def gfp(evoked):
    '''
    Global field power of the given evoked, in the unit of the modality.

    Args:
        evoked: mne.Evoked object

    Returns:
        gfp: numpy array of shape (n_times,)
    '''
    return np.sqrt((evoked.data ** 2).mean(axis=0)) * UNIT


def read_peak(evoked):
    '''
    Read the peak of the GFP curve inside the analysis window, and the global
    peak over the whole epoch for reference.

    Args:
        evoked: mne.Evoked object, already filtered and baseline corrected

    Returns:
        dict with peak time and peak amplitude of both the window and the full epoch
    '''
    t = evoked.times
    g = gfp(evoked)

    def _argmax(lo, hi):
        m = (t >= lo) & (t <= hi)
        assert m.any(), f'Empty window ({lo}, {hi}) for t in [{t[0]}, {t[-1]}]'
        idx = np.where(m)[0]
        return idx[np.argmax(g[idx])]

    i_win, i_full = _argmax(PEAK_LO, PEAK_HI), _argmax(t[0], t[-1])
    return dict(
        peak_t_window=t[i_win], peak_gfp_window=g[i_win],
        peak_t_full=t[i_full], peak_gfp_full=g[i_full])


# %% ---- 2026-10-12 ------------------------
# Play ground
group_evokeds = {}
rows = []

for cond in CONDITIONS:
    tag, stem = cond['tag'], cond['stem']

    evokeds, numbers, subjects = [], [], []
    for subj_dir in sorted(DATA_DIR.glob(f'{MODE}-S*')):
        _, subj = subj_dir.name.split('-', 1)
        fname = subj_dir / f'{stem}-epo.fif'

        if not fname.exists():
            logger.warning(f'Missing {fname}, skip it')
            continue

        epochs = mne.read_epochs(fname)
        if len(epochs) == 0:
            logger.warning(f'No epoch in {fname}, skip it')
            continue

        evoked = epochs.average()
        if evoked.nave < 1:
            logger.warning(f'{fname} has nave={evoked.nave}, skip it')
            continue

        evokeds.append(evoked)
        numbers.append(evoked.nave)
        subjects.append(subj)

        # Per-subject reading, on a copy so that the group average below stays
        # on the unfiltered data, exactly as 5.average-subjects-erp.sh does.
        _evoked = evoked.copy().filter(L_FREQ, H_FREQ, n_jobs=n_jobs)
        _evoked.apply_baseline((None, 0))
        rows.append(dict(mode=MODE, subject=subj, tag=tag,
                         nave=evoked.nave, **read_peak(_evoked)))

    assert len(evokeds) > 0, f'No epochs found for {MODE=}, {tag=}'
    logger.info(f'{tag}: {len(evokeds)} subjects, {np.sum(numbers)} trials')

    # Weighted group average, following 5.average-subjects-erp.sh
    total = np.sum(numbers)
    data = np.sum([e.data * n / total for n, e in zip(numbers, evokeds)], axis=0)
    evoked = evokeds[0].copy()
    evoked.data = data

    evoked.filter(L_FREQ, H_FREQ, n_jobs=n_jobs)
    evoked.apply_baseline((None, 0))
    group_evokeds[tag] = evoked

    fname = OUTPUT_DIR / f'{MODE}-group-{tag}-ave.fif'
    evoked.save(fname, overwrite=True)
    logger.debug(f'Saved into {fname}')

df = pd.DataFrame(rows)

# Ratio against the unprojected condition, matched by subject
pivot = df.pivot_table(index='subject', columns='tag',
                       values='peak_gfp_window', aggfunc='first')
for tag in ['visual', 'keypress', 'both']:
    df = df.merge(
        (pivot[tag] / pivot['none']).rename(f'ratio_{tag}_vs_none').reset_index(),
        on='subject', how='left')

fname = OUTPUT_DIR / f'{MODE}-per-subject.csv'
df.to_csv(fname, index=False, float_format='%.6g')
logger.info(f'Saved into {fname}')

# %%
# Group level summary: peak of the group average, plus the subject-level ratios
summary = []
for cond in CONDITIONS:
    tag = cond['tag']
    peak = read_peak(group_evokeds[tag])
    line = dict(mode=MODE, tag=tag, n_subjects=int((df['tag'] == tag).sum()),
                n_trials=int(df.loc[df['tag'] == tag, 'nave'].sum()), **peak)

    if tag != 'none':
        vals = df.loc[df['tag'] == tag, f'ratio_{tag}_vs_none'].dropna().values
        # wilcoxon errors out when every difference is exactly zero
        p = stats.wilcoxon(vals - 1).pvalue if np.any(vals != 1) else np.nan
        line.update(
            median_ratio_vs_none=np.median(vals),
            n_subject_ratio_below_one=int((vals < 1).sum()),
            wilcoxon_p_vs_one=p)
    summary.append(line)

summary_df = pd.DataFrame(summary)
fname = OUTPUT_DIR / f'{MODE}-group-summary.csv'
summary_df.to_csv(fname, index=False, float_format='%.6g')
logger.info(f'Saved into {fname}')
print(summary_df.to_string(index=False))


# %% ---- 2026-10-12 ------------------------
# Figure: group GFP curves, and the subject-level ratios
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
t = group_evokeds['none'].times
for cond in CONDITIONS:
    ax.plot(t, gfp(group_evokeds[cond['tag']]),
            label=cond['display'], color=cond['color'], lw=1.6)
ax.axvspan(PEAK_LO, PEAK_HI, color='0.85', alpha=0.4, zorder=0)
ax.axvline(0.4, color='0.5', ls='--', lw=0.8)
ax.axvline(0, color='0.5', lw=0.8)
ax.set_xlim(-0.2, 0.9)
ax.set_xlabel('Time (s)')
ax.set_ylabel(f'GFP ({UNIT_NAME})')
ax.set_title(f'{MODE} target ERP (evt={EVT}), group average')
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
rng = np.random.default_rng(42)
for i, cond in enumerate(CONDITIONS[1:], start=1):
    tag = cond['tag']
    vals = df.loc[df['tag'] == tag, f'ratio_{tag}_vs_none'].dropna().values
    xs = i + rng.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(xs, vals, color=cond['color'], s=34, alpha=0.85,
               edgecolor='white', linewidth=0.6, zorder=3)
    ax.hlines(np.median(vals), i - 0.25, i + 0.25,
              color=cond['color'], lw=2.0, zorder=4)
ax.axhline(1, color='0.4', ls='--', lw=0.9)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['visual\nonly', 'keypress\nonly', 'visual +\nkeypress'])
ax.set_ylabel(f'ratio of the {PEAK_LO}-{PEAK_HI}s GFP peak vs none')
ax.set_title('subject level, bar = median')
ax.set_xlim(0.5, 3.5)

fig.suptitle(f'{MODE}: what removes the 0.4 s component?')
fig.tight_layout()

fname = OUTPUT_DIR / f'{MODE}-gfp-comparison.png'
fig.savefig(fname, dpi=150)
plt.close(fig)
logger.info(f'Saved into {fname}')


# %% ---- 2026-10-12 ------------------------
# Pending
