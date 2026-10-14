"""
File: rt-sliding-decode.py
Author: Chuncheng Zhang
Date: 2026-10-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Sliding decode quick against slow inside the target trials, one subject
    of one modality at a time.

    Both classes are target trials, the non-target pool plays no role. The
    label of a trial is the subject's own reaction time, so the question is
    how early in the epoch the response speed is already decodable.

    Read the result with the keypress in mind. The epochs are the projected
    ones (the removal-artificial files), which takes the response locked part
    of the button artifact out, but the single trial part is only partly gone,
    and quick trials by definition press earlier than slow ones. A decoder
    that reads motor preparation, or the residue of the button response, is
    therefore above chance as well. The one stretch that can not be explained
    that way is the pre motor window [0.10, 0.20] s: every subject answers
    later than 0.211 s (the 5% quantile of the reaction times), so nothing in
    that window can come from the button press itself. It is reported
    separately for exactly that reason, and a peak that sits near the keypress
    latency is not evidence on its own.

    Stage 12 scores the same contrast in its --analysis rt branch, but it
    writes the fold wise curves as .txt into
    output/peak-window-decode/{MODE}-{SUBJ}/, and an exported result set
    carries the png and the csv only, so those curves do not survive the copy
    back and there is nothing to average over subjects with. This stage is the
    dedicated version of that one contrast: the curves are csv, the classifier
    and the cross validation are arguments, and every reading is repeated in a
    one row summary. peak-window-decode.py is left untouched, and the defaults
    here match it, so --win-peak -0.2 0.8 reproduces its auc_peak and peak_t.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-10-14 ------------------------
# Requirements and constants
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from mne.decoding import SlidingEstimator, cross_val_multiscore, Vectorizer
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Sliding decode quick against slow inside the target trials')
parser.add_argument('-s', '--subject', default='S02',
                    help='Subject name like S02')
parser.add_argument('-m', '--mode', default='EEG', help='Mode name EEG | MEG')
parser.add_argument('-t', '--tag', default='rma',
                    help='Short name of the input epochs, used in filenames')
parser.add_argument('-d', '--decoding_method', default='LR',
                    choices=['LR', 'SVC'],
                    help='Decoding method, LR | SVC. LR is what stage 12 uses')
parser.add_argument('-o', '--output-dir', default='output/rt-sliding-decode',
                    help='Where the curves and the summary are written')
parser.add_argument('--data-dir', default='output/epochs',
                    help='Folder containing the {MODE}-{SUBJ} folders')
parser.add_argument('--qs-dir', default='output/quick-slow',
                    help='Folder containing the quick-slow label csv of '
                         'stage 10')
parser.add_argument('--epochs-fname',
                    default='epochs-1-notch-removal-artificial-epo.fif',
                    help='The projected target epochs whose trials are split '
                         'by reaction time')
parser.add_argument('--groups', default='quick,slow',
                    help='The two labels to decode, comma separated. The first '
                         'one is coded as the positive class')
parser.add_argument('--picks', default='all', choices=['all', 'typed'],
                    help="'all' uses every channel exactly like stage 12, so "
                         "the curves stay comparable with its rt branch; "
                         "'typed' keeps the good MEG/EEG channels only")
parser.add_argument('--n-splits', '--n_splits', dest='n_splits', type=int,
                    default=10, help='Number of cross validation folds')
parser.add_argument('--seed', type=int, default=0,
                    help='Random state of the cross validation splitter, and '
                         'of the balancing draw. Stage 12 pinned 0, so the '
                         'defaults of the two stages agree')
parser.add_argument('--balance', '--balanced', dest='balance',
                    action='store_true',
                    help='Draw the larger group down to the smaller one. The '
                         'rule of stage 10 mostly gives two groups of equal '
                         'size, but a threshold split can leave a lopsided '
                         'pair (MEG-S03 is 86 against 282), and AUC is '
                         'insensitive to the prior while the classifier is '
                         'not. Off by default, which is what stage 12 did')
parser.add_argument('--win-preresp', '--win_preresp', dest='win_preresp',
                    type=float, nargs=2, default=[0.10, 0.20],
                    help='The window before any keypress can have happened')
parser.add_argument('--win-early', '--win_early', dest='win_early',
                    type=float, nargs=2, default=[0.24, 0.36],
                    help='The first peak window of the projected response')
parser.add_argument('--win-late', '--win_late', dest='win_late',
                    type=float, nargs=2, default=[0.41, 0.55],
                    help='The second peak window, it overlaps the keypress')
parser.add_argument('--win-peak', '--win_peak', dest='win_peak',
                    type=float, nargs=2, default=[0.0, 0.8],
                    help='The stretch the peak reading is taken over. Stage 12 '
                         'searches -0.2 0.8, pass that to reproduce its '
                         'auc_peak and peak_t to the last digit')
args = parser.parse_args()
SUBJ = args.subject
MODE = args.mode
TAG = args.tag
DECODING_METHOD = args.decoding_method
GROUPS = [g.strip() for g in args.groups.split(',') if g.strip()]

if len(GROUPS) != 2:
    raise SystemExit(f'--groups needs exactly two labels, got {GROUPS}')

logger.info(f'Start with {args=}')

# %%
DATA_DIR = Path(args.data_dir) / f'{MODE}-{SUBJ}'
QS_DIR = Path(args.qs_dir) / f'{MODE}-{SUBJ}'
OUTPUT_DIR = Path(args.output_dir) / f'{MODE}-{SUBJ}-{DECODING_METHOD}'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

WINDOWS = dict(preresp=tuple(args.win_preresp),
               early=tuple(args.win_early),
               late=tuple(args.win_late))


# %% ---- 2026-10-14 ------------------------
# Function and class
def make_clf():
    '''
    Build the per time point classifier.

    Returns:
        clf: A scikit-learn pipeline
    '''
    if DECODING_METHOD == 'SVC':
        return make_pipeline(Vectorizer(), StandardScaler(),
                             SVC(kernel='rbf'))
    return make_pipeline(Vectorizer(), StandardScaler(),
                         LogisticRegression(max_iter=1000))


def load_labels(n_trials: int):
    '''
    Read the quick / slow labels written by stage 10.

    The 'index' column points at the trial axis of the projected target
    epochs, which is the file this script splits. A tertile split leaves the
    middle third as 'drop', those trials are kept out of both classes.

    Args:
        n_trials: The number of target epochs, used to drop stale indices

    Returns:
        df: The two class label table, sorted by trial index
    '''
    trial_csv = QS_DIR / f'quick-slow-trials-{TAG}.csv'
    if not trial_csv.exists():
        raise SystemExit(f'{trial_csv} does not exist. Run '
                         f'10.quick-slow-analysis.sh for {MODE}-{SUBJ} first.')

    df = pd.read_csv(trial_csv)
    n_all = len(df)
    df = df[df['group'].isin(GROUPS)].copy()
    df = df[df['index'] < n_trials]
    df['index'] = df['index'].astype(int)
    df = df.sort_values('index').reset_index(drop=True)

    if len(df) < n_all:
        logger.info(f'{n_all - len(df)} of {n_all} labelled trials are left '
                    f'out: {n_all - len(df)} are not in {GROUPS} or point past '
                    f'the {n_trials} epochs of this subject')

    counts = df['group'].value_counts().to_dict()
    for g in GROUPS:
        if counts.get(g, 0) < args.n_splits:
            raise SystemExit(f'{MODE}-{SUBJ}: only {counts.get(g, 0)} {g} '
                             f'trials, fewer than the {args.n_splits} folds')

    rt = {g: df.loc[df['group'] == g, 'delay'] for g in GROUPS}
    logger.info(f'{len(df)} target trials: '
                f'{counts}, rt median '
                f'{", ".join(f"{g} {v.median():.3f} s" for g, v in rt.items())}')
    return df


def window_mean(curve: np.ndarray, times: np.ndarray,
                window: tuple) -> float:
    '''
    Mean of a curve inside one window.

    Args:
        curve: (n_times,) or (..., n_times)
        times: (n_times,)
        window: (lo, hi) in seconds

    Returns:
        The mean over the window, nan when the window is off the epoch
    '''
    mask = (times >= window[0]) & (times <= window[1])
    if not mask.any():
        return np.nan
    return float(curve[..., mask].mean(axis=-1))


# %% ---- 2026-10-14 ------------------------
# Play ground
def main():
    '''
    Score the quick against slow contrast of one subject and save the curve.

    Returns:
        int: 0 on success
    '''
    fname = DATA_DIR / args.epochs_fname
    if not fname.exists():
        raise SystemExit(f'{fname} does not exist. Run '
                         f'3.remove-keypress-artificial.sh for {MODE}-{SUBJ} '
                         f'first.')

    epochs = mne.read_epochs(fname, preload=True, verbose='ERROR')
    logger.info(f'{len(epochs)} target epochs, {len(epochs.ch_names)} '
                f'channels, {epochs.times[0]:.3f} - {epochs.times[-1]:.3f} s')

    df = load_labels(len(epochs))

    if args.balance:
        n_min = int(df['group'].value_counts().min())
        rng = np.random.default_rng(args.seed)
        keep = []
        for g in GROUPS:
            idx_g = df.index[df['group'] == g].to_numpy()
            keep.append(rng.choice(idx_g, n_min, replace=False))
        df = df.loc[np.sort(np.concatenate(keep))].reset_index(drop=True)
        logger.info(f'--balance keeps {len(df)} trials, {n_min} per group')

    index = df['index'].to_numpy()
    y = (df['group'] == GROUPS[0]).to_numpy().astype(int)

    picks = (mne.pick_types(epochs.info, meg=True, eeg=True, exclude='bads')
             if args.picks == 'typed' else None)
    data = epochs.get_data(copy=True, picks=picks)[index]
    logger.info(f'{data.shape[1]} channels per trial ({args.picks=}), '
                f'{int(y.sum())} {GROUPS[0]} against '
                f'{int((1 - y).sum())} {GROUPS[1]}, decoding ...')

    time_decod = SlidingEstimator(
        make_clf(), scoring='roc_auc', n_jobs=n_jobs)
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True,
                         random_state=args.seed)
    scores = cross_val_multiscore(time_decod, X=data, y=y, cv=cv, n_jobs=n_jobs)
    mean = scores.mean(axis=0)
    logger.info(f'scores shape {scores.shape}')

    # ---- the curves, as csv so that they survive an export of png and csv ----
    curves = pd.DataFrame(dict(time=np.round(epochs.times, 6),
                               auc_mean=np.round(mean, 6)))
    for i in range(scores.shape[0]):
        curves[f'auc_fold{i + 1}'] = np.round(scores[i], 6)
    fname = OUTPUT_DIR / f'curves-{TAG}.csv'
    curves.to_csv(fname, index=False, float_format='%.6g')
    logger.info(f'Saved into {fname}')

    fname = OUTPUT_DIR / f'trials-{TAG}.csv'
    df[['index', 'delay', 'group']].to_csv(fname, index=False,
                                           float_format='%.6g')
    logger.debug(f'Saved into {fname}')

    # ---- the readings of this subject ----
    mask_peak = ((epochs.times >= args.win_peak[0])
                 & (epochs.times <= args.win_peak[1]))
    i_peak = int(np.argmax(mean[mask_peak]))

    row = dict(mode=MODE, subject=SUBJ, tag=TAG, decoding=DECODING_METHOD,
               picks=args.picks, n_splits=args.n_splits, seed=args.seed,
               balanced=int(args.balance),
               n_channels=int(data.shape[1]),
               n_positive=int(y.sum()), n_negative=int((1 - y).sum()),
               positive_label=GROUPS[0], negative_label=GROUPS[1],
               rt_median_positive=round(float(
                   df.loc[df['group'] == GROUPS[0], 'delay'].median()), 4),
               rt_median_negative=round(float(
                   df.loc[df['group'] == GROUPS[1], 'delay'].median()), 4),
               auc_peak=round(float(mean[mask_peak][i_peak]), 4),
               peak_t=round(float(epochs.times[mask_peak][i_peak]), 4),
               auc_mean=round(float(mean.mean()), 4),
               auc_min=round(float(mean.min()), 4),
               auc_max=round(float(mean.max()), 4))
    for name, window in WINDOWS.items():
        row[f'auc_{name}'] = round(window_mean(mean, epochs.times, window), 4)

    fname = OUTPUT_DIR / 'summary.csv'
    row_df = pd.DataFrame([row])
    row_df.to_csv(fname, index=False, float_format='%.6g')
    logger.info(f'Saved into {fname}')
    print(row_df.T.to_string())

    # ---- figure: the curve of this subject ----
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sem = scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
    ax.plot(epochs.times, mean, lw=1.7, color='#1f77b4',
            label=f'{GROUPS[0]} vs {GROUPS[1]}')
    ax.fill_between(epochs.times, mean - sem, mean + sem, color='#1f77b4',
                    alpha=0.15)
    ax.axhline(0.5, color='gray', lw=0.9, ls=':')
    ax.axvline(0, color='k', lw=0.9, ls='--', alpha=0.5)
    ax.axvspan(*args.win_preresp, color='tab:green', alpha=0.10,
               label='before any keypress can be')
    ax.axvspan(*args.win_early, color='tab:orange', alpha=0.08)
    ax.axvspan(*args.win_late, color='tab:purple', alpha=0.08)
    # Where the motor response of each group can enter the epoch
    ax.axvline(row['rt_median_positive'], color='#1f77b4', lw=1.0, ls='-.',
               alpha=0.7, label=f'{GROUPS[0]} median rt')
    ax.axvline(row['rt_median_negative'], color='#d62728', lw=1.0, ls='-.',
               alpha=0.7, label=f'{GROUPS[1]} median rt')
    ax.axvspan(args.win_peak[0], args.win_peak[1], color='0.85', alpha=0.25,
               zorder=0)
    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(0.35, max(0.9, float(mean.max()) + 0.03))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ROC AUC')
    ax.set_title(f'{MODE}-{SUBJ}: {GROUPS[0]} versus {GROUPS[1]} inside the '
                 f'target trials ({DECODING_METHOD}, {TAG})')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fname = OUTPUT_DIR / f'rt-sliding-{TAG}.png'
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    logger.info(f'Saved into {fname}')
    return 0


main()


# %% ---- 2026-10-14 ------------------------
# Pending


# %% ---- 2026-10-14 ------------------------
# Pending
