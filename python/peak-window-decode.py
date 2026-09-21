"""
File: peak-window-decode.py
Author: Chuncheng Zhang
Date: 2026-09-20
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decode on the two peak windows of the target response.

    The response after the keypress removal has two peaks in MEG and one in
    EEG. A single time resolved decoding curve can not tell whether the two
    peaks carry the same information, so the decoding is organised around the
    peaks instead:

    windows   Decode target against non-target separately inside the early
              window, the late window and a pre stimulus baseline window.
              If the late peak carries no target identity, its window sits at
              chance and the peak is not a target component. The late window
              overlaps the keypress, so --min-rt can restrict the target trials
              to the ones whose button came after the window, which removes the
              single trial button residue as an explanation.
    transfer  Train the target against non-target classifier in one window and
              test it in the other. A classifier that does not transfer says
              the two peaks reflect different activity, a classifier that
              transfers says they are two views of the same process.
    rt        Decode quick against slow inside the target trials, sliding over
              time. It asks at which latency the epoch already knows how fast
              the subject will respond. The window before the earliest
              plausible keypress is reported separately, because nothing there
              can be explained by the button press.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending

Note:
    Everything runs on the epochs whose keypress subspace has been projected
    out, so the response locked part of the artifact is gone before the
    decoding starts. The residual single trial part is not, which is why the
    quick against slow decoding has to be read together with the earliest
    keypress latency of the subject.
"""


# %% ---- 2026-09-20 ------------------------
# Requirements and constants
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from mne.decoding import SlidingEstimator, cross_val_multiscore, Vectorizer
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Decode on the peak windows of the target response')
parser.add_argument('-s', '--subject', default='S02',
                    help='Subject name like S02')
parser.add_argument('-m', '--mode', default='EEG', help='Mode name EEG | MEG')
parser.add_argument('-d', '--data-dir', default='output/epochs',
                    help='Folder containing the {MODE}-{SUBJ} folders')
parser.add_argument('--qs-dir', default='output/quick-slow',
                    help='Folder written by quick-slow-analysis.py, it holds '
                         'the quick and slow trial assignment')
parser.add_argument('--min-rt', type=float, default=0.,
                    help='When it is above zero the target against non-target '
                         'decoding keeps only the target trials whose keypress '
                         'came after this latency. The late window overlaps the '
                         'keypress, so a residue of the single trial button '
                         'response can always be suspected there; on the trials '
                         'kept here every window ends before the button was '
                         'pressed. 0 keeps all the trials')
parser.add_argument('-o', '--output-dir', default='output/peak-window-decode',
                    help='Where the decoding results are written')
parser.add_argument('-t', '--tag', default='rma',
                    help='Short name of the input epochs, used in filenames')
parser.add_argument('--analysis', default='all',
                    choices=['all', 'windows', 'transfer', 'rt'],
                    help='Which analysis to run')
parser.add_argument('--classifier', default='LR', choices=['LR', 'SVC'],
                    help='The classifier, both support decision_function')
parser.add_argument('--win-early', type=float, nargs=2, default=[0.24, 0.36],
                    help='The first peak window in seconds')
parser.add_argument('--win-late', type=float, nargs=2, default=[0.41, 0.55],
                    help='The second peak window in seconds')
parser.add_argument('--win-base', type=float, nargs=2, default=[-0.2, -0.05],
                    help='The baseline window in seconds, for reference')
parser.add_argument('--win-preresp', type=float, nargs=2, default=[0.1, 0.2],
                    help='The window before the earliest plausible keypress, '
                         'used to summarize the quick against slow decoding')
parser.add_argument('--win-rt', type=float, nargs=2, default=[-0.2, 0.8],
                    help='The time range plotted and saved for the quick '
                         'against slow decoding')
parser.add_argument('--n-splits', type=int, default=10,
                    help='The number of cross validation folds')
args = parser.parse_args()
SUBJ = args.subject
MODE = args.mode
TAG = args.tag

logger.info(f'Start with {SUBJ=}, {MODE=}, {TAG=}, {args.analysis=}')

# %%
DATA_DIR = Path(args.data_dir) / f'{MODE}-{SUBJ}'
QS_DIR = Path(args.qs_dir) / f'{MODE}-{SUBJ}'
OUTPUT_DIR = Path(args.output_dir) / f'{MODE}-{SUBJ}'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

WINDOW_CSV = Path(args.output_dir) / f'window-decode-{TAG}.csv'
TRANSFER_CSV = Path(args.output_dir) / f'transfer-decode-{TAG}.csv'
RT_CSV = Path(args.output_dir) / f'rt-decode-{TAG}.csv'

WINDOWS = dict(baseline=tuple(args.win_base),
               early=tuple(args.win_early),
               late=tuple(args.win_late))


# %% ---- 2026-09-20 ------------------------
# Function and class
def make_clf():
    '''
    Build the classifier pipeline.

    Returns:
        clf: The pipeline, a scaler followed by the classifier
    '''
    if args.classifier == 'SVC':
        return make_pipeline(StandardScaler(), SVC(kernel='rbf'))
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000))


def window_features(epochs: mne.Epochs, window: tuple, index: np.ndarray = None):
    '''
    Average the epochs over one window, which gives one feature vector per
    trial.

    Args:
        epochs: MNE Epochs
        window: (tmin, tmax) in seconds
        index: The trial indices to keep, all of them when None

    Returns:
        X: (n_trials, n_channels)
    '''
    times = epochs.times
    mask = (times >= window[0]) & (times <= window[1])
    if not mask.any():
        raise ValueError(f'The {window=} does not overlap the epoch')
    data = epochs.get_data(copy=True)[:, :, mask].mean(axis=2)
    if index is not None:
        data = data[index]
    return data


def cross_val_auc(X: np.ndarray, y: np.ndarray, n_splits: int):
    '''
    The out of fold AUC of a classifier on one feature set.

    Args:
        X: (n_trials, n_features)
        y: The labels
        n_splits: The number of cross validation folds

    Returns:
        auc: The mean out of fold AUC
        scores: The per fold AUC
    '''
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(make_clf(), X, y, cv=cv, scoring='roc_auc',
                             n_jobs=n_jobs)
    return float(scores.mean()), scores


def transfer_auc(X_train: np.ndarray, X_test: np.ndarray, y: np.ndarray,
                 n_splits: int):
    '''
    Train on one window and score the trials on another window.

    The folds are the same for both windows, so a trial that is left out of
    the training set is the only one used to test, which keeps the estimate
    honest.

    Args:
        X_train: The features of the training window
        X_test: The features of the testing window
        y: The labels
        n_splits: The number of cross validation folds

    Returns:
        auc: The AUC of the transferred classifier
    '''
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    decision = np.zeros(len(y))
    for train, test in cv.split(X_train, y):
        clf = make_clf()
        clf.fit(X_train[train], y[train])
        decision[test] = clf.decision_function(X_test[test])
    return float(roc_auc_score(y, decision))


def upsert(row: dict, fpath: Path,
           keys=('mode', 'subject', 'tag')):
    '''
    Write one row into a csv, replacing the older row with the same key.

    Args:
        row: The new row
        fpath: The csv to write
        keys: The columns identifying a row
    '''
    if fpath.exists():
        df = pd.read_csv(fpath)
        # A key column that the old table does not have yet is filled with the
        # value that stands for the default run, otherwise rerunning the
        # default appends a second row next to the old one instead of
        # replacing it.
        for k in keys:
            if k not in df.columns and k in row:
                df[k] = 0. if isinstance(row[k], (int, float)) else row[k]
        same = np.ones(len(df), bool)
        for k in keys:
            if k not in df.columns or k not in row:
                same &= False
                continue
            same &= (df[k] == row[k]).values
        df = df[~same]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(list(keys)).reset_index(drop=True)
    df.to_csv(fpath, index=False)
    logger.debug(f'Saved into {fpath}, {len(df)} rows')


def load_epochs(target: bool):
    '''
    Read the epochs of one condition, with the keypress subspace removed.

    Args:
        target: True for the target trials, False for the non-target ones

    Returns:
        epochs: MNE Epochs
    '''
    evt = '1' if target else '2'
    fpath = DATA_DIR / f'epochs-{evt}-notch-removal-artificial-epo.fif'
    if not fpath.exists():
        logger.error(f'{fpath} does not exist. Run '
                     f'3.remove-keypress-artificial.sh for {MODE}-{SUBJ} first.')
        return None
    return mne.read_epochs(fpath, preload=True, verbose='ERROR')


def late_response_trials(n_trials: int):
    '''
    The target trial indices whose keypress came after args.min_rt seconds.

    Args:
        n_trials: The number of target epochs

    Returns:
        index: The sorted trial indices, None when the selection can not be
            made
    '''
    trial_csv = QS_DIR / f'quick-slow-trials-{TAG}.csv'
    if not trial_csv.exists():
        logger.error(f'{trial_csv} does not exist, --min-rt needs it. Run '
                     f'10.quick-slow-analysis.sh for {MODE}-{SUBJ} first.')
        return None
    df = pd.read_csv(trial_csv)
    index = df[df['delay'] >= args.min_rt]['index'].to_numpy()
    index = np.sort(index[index < n_trials])
    if index.size < 20:
        logger.warning(f'{MODE}-{SUBJ}: only {index.size} target trials have '
                       f'a keypress later than {args.min_rt:g} s, '
                       f'--min-rt is skipped')
        return None
    logger.info(f'{MODE}-{SUBJ}: {index.size} of {n_trials} target trials '
                f'have a keypress later than {args.min_rt:g} s')
    return index


def run_windows():
    '''
    Decode target against non-target inside the baseline, the early and the
    late window.
    '''
    epochs_1 = load_epochs(target=True)
    epochs_2 = load_epochs(target=False)
    if epochs_1 is None or epochs_2 is None:
        return 1

    idx_1_all = np.arange(len(epochs_1))
    if args.min_rt > 0:
        idx_1_all = late_response_trials(len(epochs_1))
        if idx_1_all is None:
            return 1

    # Keep the two classes comparable in size, otherwise the AUC of the
    # unbalanced design is hard to compare with the transfer analysis.
    n = min(len(idx_1_all), len(epochs_2))
    rng = np.random.default_rng(0)
    idx_1 = np.sort(rng.choice(idx_1_all, n, replace=False))
    idx_2 = np.sort(rng.choice(len(epochs_2), n, replace=False))
    y = np.r_[np.ones(n), np.zeros(n)]

    for name, window in WINDOWS.items():
        X = np.vstack([window_features(epochs_1, window, idx_1),
                       window_features(epochs_2, window, idx_2)])
        auc, scores = cross_val_auc(X, y, args.n_splits)
        row = dict(mode=MODE, subject=SUBJ, tag=TAG, analysis='windows',
                   window=name, tmin=window[0], tmax=window[1],
                   min_rt=args.min_rt,
                   auc=round(auc, 4), auc_std=round(float(scores.std()), 4),
                   n_per_class=n)
        upsert(row, WINDOW_CSV,
               keys=('mode', 'subject', 'tag', 'window', 'min_rt'))
        logger.info(f'{MODE}-{SUBJ} window {name} [{window[0]:g}, '
                    f'{window[1]:g}] s, min rt {args.min_rt:g} s: '
                    f'AUC {auc:.4f}')
    return 0


def run_transfer():
    '''
    Train the target against non-target classifier in one window and test it
    in the other, in both directions.
    '''
    epochs_1 = load_epochs(target=True)
    epochs_2 = load_epochs(target=False)
    if epochs_1 is None or epochs_2 is None:
        return 1

    n = min(len(epochs_1), len(epochs_2))
    rng = np.random.default_rng(0)
    idx_1 = np.sort(rng.choice(len(epochs_1), n, replace=False))
    idx_2 = np.sort(rng.choice(len(epochs_2), n, replace=False))
    y = np.r_[np.ones(n), np.zeros(n)]

    features = dict()
    for name in ['early', 'late']:
        window = WINDOWS[name]
        features[name] = np.vstack(
            [window_features(epochs_1, window, idx_1),
             window_features(epochs_2, window, idx_2)])

    for train_name in ['early', 'late']:
        for test_name in ['early', 'late']:
            if train_name == test_name:
                auc, _ = cross_val_auc(
                    features[train_name], y, args.n_splits)
            else:
                auc = transfer_auc(
                    features[train_name], features[test_name], y,
                    args.n_splits)
            row = dict(mode=MODE, subject=SUBJ, tag=TAG,
                       train_window=train_name, test_window=test_name,
                       same_window=int(train_name == test_name),
                       auc=round(float(auc), 4), n_per_class=n)
            upsert(row, TRANSFER_CSV,
                   keys=('mode', 'subject', 'tag', 'train_window',
                         'test_window'))
            logger.info(f'{MODE}-{SUBJ} train {train_name} test {test_name}: '
                        f'AUC {auc:.4f}')
    return 0


def run_rt():
    '''
    Decode quick against slow inside the target trials, sliding over time.
    '''
    trial_csv = QS_DIR / f'quick-slow-trials-{TAG}.csv'
    if not trial_csv.exists():
        logger.error(f'{trial_csv} does not exist. Run '
                     f'10.quick-slow-analysis.sh for {MODE}-{SUBJ} first.')
        return 1

    epochs = load_epochs(target=True)
    if epochs is None:
        return 1

    df = pd.read_csv(trial_csv)
    df = df[df['group'].isin(['quick', 'slow'])]
    df = df[df['index'] < len(epochs)]
    index = df['index'].to_numpy()
    y = (df['group'] == 'quick').to_numpy().astype(int)
    logger.info(f'{MODE}-{SUBJ}: {int(y.sum())} quick and '
                f'{int((1 - y).sum())} slow target trials')

    X = epochs.get_data(copy=True)[index]
    clf = make_pipeline(Vectorizer(), StandardScaler(),
                        LogisticRegression(max_iter=1000))
    time_decod = SlidingEstimator(clf, scoring='roc_auc', n_jobs=n_jobs)
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=0)
    scores = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=n_jobs)
    mean = scores.mean(axis=0)

    fname = OUTPUT_DIR / f'rt-scores-{TAG}.txt'
    np.savetxt(fname, scores, fmt='%.6f')
    logger.info(f'Saved into {fname}')

    # The time axis is written next to the scores so that the group level
    # summary can read the curves back without an epoch file.
    fname = OUTPUT_DIR / f'rt-times-{TAG}.txt'
    np.savetxt(fname, epochs.times, fmt='%.6f')
    logger.debug(f'Saved into {fname}')

    # The window before the earliest plausible keypress is the only one that
    # can not be explained by the button press at all.
    times = epochs.times
    mask = (times >= args.win_preresp[0]) & (times <= args.win_preresp[1])
    auc_pre = float(mean[mask].mean()) if mask.any() else np.nan

    mask_rt = (times >= args.win_rt[0]) & (times <= args.win_rt[1])
    i_peak = int(np.argmax(mean[mask_rt]))

    row = dict(mode=MODE, subject=SUBJ, tag=TAG,
               n_quick=int(y.sum()), n_slow=int((1 - y).sum()),
               auc_preresp=round(auc_pre, 4),
               auc_peak=round(float(mean[mask_rt][i_peak]), 4),
               peak_t=round(float(times[mask_rt][i_peak]), 4))
    upsert(row, RT_CSV)
    logger.info(f'{MODE}-{SUBJ} quick versus slow: pre response '
                f'[{args.win_preresp[0]:g}, {args.win_preresp[1]:g}] s '
                f'AUC {auc_pre:.4f}, peak {row["peak_t"]:.3f} s '
                f'AUC {row["auc_peak"]:.4f}')

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, mean, lw=1.6, label='quick versus slow')
    sem = scores.std(axis=0) / np.sqrt(scores.shape[0])
    ax.fill_between(times, mean - sem, mean + sem, alpha=.15)
    ax.axhline(.5, color='gray', ls=':', lw=.9)
    ax.axvline(0, color='k', ls=':', lw=.8)
    ax.axvspan(*args.win_preresp, color='tab:green', alpha=.12,
               label='before any keypress can be')
    ax.axvspan(*args.win_early, color='tab:red', alpha=.08)
    ax.axvspan(*args.win_late, color='tab:blue', alpha=.08)
    ax.set_xlim(args.win_rt[0], args.win_rt[1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('AUC')
    ax.set_title(f'{MODE}-{SUBJ} quick versus slow, target trials after removal')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fname = OUTPUT_DIR / f'rt-decode-{TAG}.png'
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    logger.info(f'Saved into {fname}')
    return 0


# %% ---- 2026-09-20 ------------------------
# Play ground
def main():
    '''
    Dispatch to the requested analysis.
    '''
    status = 0
    if args.analysis in ('all', 'windows'):
        status |= run_windows()
    if args.analysis in ('all', 'transfer'):
        status |= run_transfer()
    if args.analysis in ('all', 'rt'):
        status |= run_rt()
    return status


main()


# %% ---- 2026-09-20 ------------------------
# Pending


# %% ---- 2026-09-20 ------------------------
# Pending
