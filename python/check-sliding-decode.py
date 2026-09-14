"""
File: check-sliding-decode.py
Author: Chuncheng Zhang
Date: 2026-09-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Group-level summary of the sliding decoding scores.
    Plots mean +/- SEM across subjects for EEG and MEG,
    with and without the keypress-artifact removal (rma),
    and reports the peak latency within 0 - 0.8 s.

Usage:
    python python/check-sliding-decode.py
"""


# %% ---- 2026-09-14 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Summarize the sliding-decode scores across subjects')
parser.add_argument('-d', '--data-dir', default='output/sliding-decode',
                    help='Folder containing the {MODE}-S*-{DECOD} folders')
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = DATA_DIR

# IPython compatibility
try:
    __IPYTHON__
    DATA_DIR = Path('../output/sliding-decode')
except NameError:
    pass

logger.info(f'Summarizing {DATA_DIR=}')


# %% ---- 2026-09-14 ------------------------
# Function and class
def load_subject_scores(folder: Path, fname: str):
    '''Read the scores of one subject, average over folds.

    Args:
        folder: Folder like EEG-S02-LR
        fname: scores.txt or scores-rma.txt

    Returns:
        1D array over times, or None
    '''
    f = folder / fname
    if not f.exists():
        return None
    scores = np.loadtxt(f)
    if scores.ndim == 1:
        scores = scores[None, :]
    return scores.mean(axis=0)


# %% ---- 2026-09-14 ------------------------
# Play ground
MODES = ['EEG', 'MEG']
DECODS = ['LR', 'SVC']
CONDITIONS = [('withoutRMA', 'scores.txt', '--'),
              ('withRMA', 'scores-rma.txt', '-')]

n_times = None
curves = {}  # (mode, decod, tag) -> (n_subjects, n_times)

for mode, decod, tag, fname, ls in [
        (m, d, t, f, ls)
        for m in MODES for d in DECODS for t, f, ls in CONDITIONS]:
    values = []
    for folder in sorted(DATA_DIR.glob(f'{mode}-*-{decod}')):
        v = load_subject_scores(folder, fname)
        if v is not None:
            values.append(v)
    if not values:
        logger.warning(f'No scores found for {mode}-{decod}-{tag}')
        continue
    n_times = values[0].size
    curves[(mode, decod, tag)] = np.array(values)

if not curves:
    raise FileNotFoundError(f'No scores found in {DATA_DIR}')

times = np.linspace(-0.5, 1.5, n_times)
logger.debug(f'{n_times=} time points, {len(curves)} curves')

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

summary = []
for mode, ax in zip(MODES, axes):
    for decod in DECODS:
        for tag, fname, ls in CONDITIONS:
            if (mode, decod, tag) not in curves:
                continue
            data = curves[(mode, decod, tag)]
            m = data.mean(axis=0)
            sem = data.std(axis=0) / np.sqrt(data.shape[0])

            label = f'{decod} {tag}'
            ax.plot(times, m, ls, label=label, lw=1.6, alpha=0.9)
            ax.fill_between(times, m - sem, m + sem, alpha=0.12)

            # Peak latency and AUC within the 0 - 0.8 s window
            window = (times >= 0) & (times <= 0.8)
            i = np.argmax(m[window])
            summary.append((mode, decod, tag,
                            times[window][i], m[window][i],
                            data.shape[0]))

    ax.axhline(0.5, color='gray', lw=0.8, ls=':', label='chance')
    ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.axvline(0.3, color='r', lw=0.8, ls=':', alpha=0.7)
    ax.set_title(f'{mode}: target vs non-target')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(0.4, 0.95)
    ax.legend(loc='lower right', fontsize=8)

axes[0].set_ylabel('ROC AUC')
fig.tight_layout()

img_fname = OUTPUT_DIR / 'group-decode.png'
fig.savefig(img_fname, dpi=130)
plt.close(fig)
logger.info(f'Saved into {img_fname}')

# %%
df = pd.DataFrame(summary, columns=['mode', 'decoding', 'condition',
                                    'peak_time_s', 'peak_auc', 'n_subjects'])
display(df)

csv_fname = OUTPUT_DIR / 'group-decode-summary.csv'
df.to_csv(csv_fname, index=False)
logger.info(f'Saved into {csv_fname}')

# %% ---- 2026-09-14 ------------------------
# Pending
