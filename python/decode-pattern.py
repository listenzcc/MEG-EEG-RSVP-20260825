"""
File: decode-pattern.py
Author: Chuncheng Zhang
Date: 2026-10-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    The spatial side of the two sliding decodes.

    Stage 6 (target against non-target) and stage 19 (quick against slow)
    both end in a curve of AUC over time and say nothing about where on the
    head the information sits. This stage fits the same per time point
    classifier that those two stages score, and keeps the spatial pattern
    behind it instead of throwing it away, so the time axis can be read as a
    sequence of maps.

    Two things come out of every time point:

        pattern      the Haufe transform of the classifier weights,
                     a = cov(X) @ (w / sd), with w the logistic regression
                     coefficients taken back into data units. The weights
                     alone say which channels the classifier listens to; the
                     covariance turns that into the topography that carries
                     the discriminative activity, which is what a map should
                     show.
        diff         the plain average difference between the two classes at
                     the same time point. It is univariate and needs no
                     classifier, and it is saved next to the pattern so the
                     two can be compared. Where they agree the map is doing
                     what the average does; where they disagree the
                     classifier is reading something the average does not
                     show, or the average is dominated by activity the
                     classifier cannot use.

    Two contrasts are supported, both with --analysis:

        target       target epochs against non-target epochs. The two files
                     carry wildly different trial counts, so --balance draws
                     the larger class down to the smaller one and is on by
                     default here.
        rt           quick against slow inside the target trials, with the
                     labels of stage 10. Balanced already by construction, so
                     --balance is off by default here.

    Read the maps with one limit in mind: the pattern of a single time point
    is fitted on its own and is noisy, and the saved version is normalised to
    unit length at every time point, so the colour of a map carries the
    shape of the topography and nothing else. How strong the topography is
    has to be read from the norm column and from the AUC curve of the stage
    that scored the same contrast, and how much of the shape is shared
    across subjects is what the group stage measures.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-10-21 ------------------------
# Requirements and constants
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Spatial pattern of the per time point decoder, one subject')
parser.add_argument('-s', '--subject', default='S02',
                    help='Subject name like S02')
parser.add_argument('-m', '--mode', default='EEG', help='Mode name EEG | MEG')
parser.add_argument('--analysis', default='target',
                    choices=['target', 'rt'],
                    help="'target' decodes target against non-target, 'rt' "
                         'decodes quick against slow inside the target '
                         'trials')
parser.add_argument('--tag', default='rma',
                    help='Short name of the input epochs, used in filenames. '
                         'rma is the projected epochs, norp the raw ones')
parser.add_argument('--epochs-fname', default=None,
                    help='The target epochs. Left out, it follows --tag: '
                         'epochs-1-notch-removal-artificial-epo.fif for rma, '
                         'epochs-1-notch-epo.fif for norp')
parser.add_argument('--epochs-fname-2', default=None,
                    help='The non-target epochs of the target contrast, only '
                         'read by --analysis target. Left out, it follows '
                         '--tag the same way')
parser.add_argument('--data-dir', default='output/epochs',
                    help='Folder containing the {MODE}-{SUBJ} folders')
parser.add_argument('--qs-dir', default='output/quick-slow',
                    help='Folder containing the quick-slow label csv of '
                         'stage 10, only read by --analysis rt')
parser.add_argument('--qs-tag', '--qs_tag', dest='qs_tag', default=None,
                    help='Tag of the quick-slow label csv. Left out, it '
                         'follows --tag')
parser.add_argument('--groups', default='quick,slow',
                    help='The two labels of the rt contrast, comma '
                         'separated. The first one is the positive class')
parser.add_argument('--output-dir', default='output/decode-pattern',
                    help='Where the patterns and the summary are written')
parser.add_argument('--picks', default='typed', choices=['all', 'typed'],
                    help="'typed' keeps the MEG/EEG channels and drops the "
                         "bad ones, which is what a map needs; 'all' keeps "
                         "every channel of the file")
parser.add_argument('--balance', action='store_true', default=None,
                    help='Draw the larger class down to the smaller one')
parser.add_argument('--no-balance', dest='balance', action='store_false',
                    help='Keep every trial of both classes')
parser.add_argument('--no-haufe', dest='haufe', action='store_false',
                    help='Save the classifier weights instead of the Haufe '
                         'pattern')
parser.add_argument('--seed', type=int, default=0,
                    help='Random state of the balancing draw and of the '
                         'trial shuffle')
parser.add_argument('--t-min', type=float, default=-0.2,
                    help='First time point written out, in seconds')
parser.add_argument('--t-max', type=float, default=1.2,
                    help='Last time point written out, in seconds')
parser.add_argument('--step-ms', type=float, default=10.0,
                    help='Spacing of the written time points, in ms. The '
                         'patterns are fitted at every sample of the epoch '
                         'either way, this only thins what is saved')
parser.add_argument('--smooth-ms', type=float, default=40.0,
                    help='Width of the moving average applied along the time '
                         'axis after fitting, in ms. One time point on its '
                         'own is a noisy estimate, the average over a few '
                         'samples of an evoked component that is itself tens '
                         'of ms wide is not. Pass 0 to keep every time point '
                         'as it was fitted')
parser.add_argument('--highpass', type=float, default=None,
                    help='Refilter the epochs with this high pass before '
                         'decoding, in Hz')
parser.add_argument('--lowpass', type=float, default=None,
                    help='Refilter the epochs with this low pass, in Hz')
args = parser.parse_args()

SUBJ = args.subject
MODE = args.mode
ANALYSIS = args.analysis
TAG = args.tag
QS_TAG = args.qs_tag if args.qs_tag else args.tag
GROUPS = [g.strip() for g in args.groups.split(',') if g.strip()]

if ANALYSIS == 'rt' and len(GROUPS) != 2:
    raise SystemExit(f'--groups needs exactly two labels, got {GROUPS}')

# The two contrasts differ in how lopsided they are, so the balancing
# default follows the contrast and not one global value
if args.balance is None:
    BALANCE = (ANALYSIS == 'target')
else:
    BALANCE = args.balance

if args.epochs_fname is None:
    args.epochs_fname = ('epochs-1-notch-removal-artificial-epo.fif'
                         if TAG == 'rma' else 'epochs-1-notch-epo.fif')
if args.epochs_fname_2 is None:
    args.epochs_fname_2 = ('epochs-2-notch-removal-artificial-epo.fif'
                           if TAG == 'rma' else 'epochs-2-notch-epo.fif')

logger.info(f'Start with {args=}')

# %%
DATA_DIR = Path(args.data_dir) / f'{MODE}-{SUBJ}'
QS_DIR = Path(args.qs_dir) / f'{MODE}-{SUBJ}'
OUTPUT_DIR = Path(args.output_dir) / f'{MODE}-{SUBJ}'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

EPS = 1e-12


# %% ---- 2026-10-21 ------------------------
# Function and class
def channel_picks(info):
    '''
    The channels a map can be drawn on.

    Args:
        info: The info of the epochs

    Returns:
        picks: The channel indices, and the names and types behind them
    '''
    if args.picks == 'typed':
        picks = mne.pick_types(info, meg=True, eeg=True, exclude='bads')
    else:
        picks = np.arange(len(info.ch_names))
    names = [info.ch_names[i] for i in picks]
    types = [mne.channel_type(info, i) for i in picks]
    return picks, names, types


def fit_one(X_t: np.ndarray, y: np.ndarray):
    '''
    Fit the decoder of one time point and turn its weights into a pattern.

    Args:
        X_t: (n_trials, n_channels) data of this time point
        y: (n_trials,) class labels, the positive class is the first label

    Returns:
        pattern: (n_channels,), the Haufe pattern or the raw weights
        norm: The L2 norm of what is returned
    '''
    scaler = StandardScaler()
    Z = scaler.fit_transform(X_t)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Z, y)

    # StandardScaler divides by the standard deviation, so the weights come
    # back into data units by the same division
    scale = scaler.scale_.copy()
    scale[scale < EPS] = 1.0
    w_raw = clf.coef_.ravel() / scale

    if args.haufe:
        Xc = X_t - X_t.mean(axis=0, keepdims=True)
        cov = Xc.T @ Xc / max(len(X_t) - 1, 1)
        pattern = cov @ w_raw
    else:
        pattern = w_raw

    norm = float(np.linalg.norm(pattern))
    if norm < EPS:
        return np.zeros_like(pattern), 0.0
    return pattern, norm


def fit_all(X: np.ndarray, y: np.ndarray, n_ch: int):
    '''
    Fit every time point of one subject.

    Args:
        X: (n_trials, n_channels, n_times)
        y: (n_trials,) class labels
        n_ch: The number of channels, for the allocation

    Returns:
        patterns: (n_times, n_channels)
        norms: (n_times,)
    '''
    n_times = X.shape[2]
    patterns = np.zeros((n_times, n_ch))
    norms = np.zeros(n_times)
    for i_t in range(n_times):
        patterns[i_t], norms[i_t] = fit_one(X[:, :, i_t], y)
    return patterns, norms


def smooth_time(mat: np.ndarray, sfreq: float):
    '''
    Moving average along the time axis.

    Args:
        mat: (n_times, n_channels)
        sfreq: The sampling rate, for turning ms into samples

    Returns:
        out: The smoothed array, a copy of mat when the width is below one
             sample
    '''
    width = int(round(args.smooth_ms / 1000.0 * sfreq))
    if width <= 1:
        return mat
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width) / width
    pad = width // 2
    padded = np.pad(mat, ((pad, pad), (0, 0)), mode='edge')
    out = np.empty_like(mat)
    for i in range(mat.shape[1]):
        out[:, i] = np.convolve(padded[:, i], kernel, mode='valid')
    return out


def balanced_pairs(n_pos: int, n_neg: int, rng):
    '''
    Draw both classes down to the same trial count.

    Args:
        n_pos: Trials of the positive class
        n_neg: Trials of the negative class
        rng: The random generator

    Returns:
        idx_pos, idx_neg: The drawn trial indices, both of the same length
    '''
    n = min(n_pos, n_neg)
    return rng.choice(n_pos, n, replace=False), rng.choice(n_neg, n, replace=False)


def load_rt_labels(n_trials: int):
    '''
    Read the quick / slow labels written by stage 10.

    Args:
        n_trials: The number of target epochs, used to drop stale indices

    Returns:
        df: The label table, sorted by trial index
    '''
    trial_csv = QS_DIR / f'quick-slow-trials-{QS_TAG}.csv'
    if not trial_csv.exists():
        raise SystemExit(f'{trial_csv} does not exist. Run '
                         f'10.quick-slow-analysis.sh for {MODE}-{SUBJ} '
                         f'first, or point --qs-tag at the label file of '
                         f'these epochs.')

    df = pd.read_csv(trial_csv)
    df = df[df['group'].isin(GROUPS)].copy()
    df = df[df['index'] < n_trials]
    df['index'] = df['index'].astype(int)
    df = df.sort_values('index').reset_index(drop=True)
    return df


def refilter(epochs):
    '''
    Apply the optional on the fly filter.

    Args:
        epochs: The epochs to refilter

    Returns:
        epochs: The same object, refiltered when asked for
    '''
    if args.highpass is None and args.lowpass is None:
        return epochs
    epochs.filter(args.highpass, args.lowpass, n_jobs=n_jobs,
                  verbose='ERROR')
    logger.info(f'Refiltered the epochs to {args.highpass} - '
                f'{args.lowpass} Hz')
    return epochs


# %% ---- 2026-10-21 ------------------------
# Play ground
def main():
    '''
    Fit the pattern of one subject and save it as csv.

    Returns:
        int: 0 on success
    '''
    fname = DATA_DIR / args.epochs_fname
    if not fname.exists():
        raise SystemExit(f'{fname} does not exist. Run the matching stage '
                         f'first.')

    epochs_pos = refilter(mne.read_epochs(fname, preload=True,
                                          verbose='ERROR'))
    picks, ch_names, ch_types = channel_picks(epochs_pos.info)
    logger.info(f'{MODE}-{SUBJ}: {len(ch_names)} channels ({args.picks=})')

    if ANALYSIS == 'target':
        fname_2 = DATA_DIR / args.epochs_fname_2
        if not fname_2.exists():
            raise SystemExit(f'{fname_2} does not exist.')
        epochs_neg = refilter(mne.read_epochs(fname_2, preload=True,
                                              verbose='ERROR'))
        if (len(epochs_neg.times) != len(epochs_pos.times)
                or not np.allclose(epochs_neg.times, epochs_pos.times)):
            raise SystemExit(f'{MODE}-{SUBJ}: the two epochs files do not '
                             f'share a time axis')

        X_pos = epochs_pos.get_data(picks=picks)
        X_neg = epochs_neg.get_data(picks=picks)
        # The average difference keeps every trial, the balancing below only
        # touches what the classifier sees
        diff = X_pos.mean(axis=0) - X_neg.mean(axis=0)
        classes = ('target', 'nontarget')

        rng = np.random.default_rng(args.seed)
        if BALANCE:
            i_pos, i_neg = balanced_pairs(len(X_pos), len(X_neg), rng)
        else:
            i_pos, i_neg = np.arange(len(X_pos)), np.arange(len(X_neg))
        X = np.concatenate([X_pos[i_pos], X_neg[i_neg]], axis=0)
        y = np.concatenate([np.ones(len(i_pos)), np.zeros(len(i_neg))])
        logger.info(f'{len(i_pos)} {classes[0]} against {len(i_neg)} '
                    f'{classes[1]} trials (balance={BALANCE})')

    else:
        df = load_rt_labels(len(epochs_pos))
        X_all = epochs_pos.get_data(picks=picks)
        y_all = (df['group'] == GROUPS[0]).to_numpy().astype(int)
        index = df['index'].to_numpy()
        X_all = X_all[index]
        diff = X_all[y_all == 1].mean(axis=0) - X_all[y_all == 0].mean(axis=0)
        classes = tuple(GROUPS)
        logger.info(f'{int(y_all.sum())} {GROUPS[0]} against '
                    f'{int((1 - y_all).sum())} {GROUPS[1]} trials')

        rng = np.random.default_rng(args.seed)
        if BALANCE:
            i_pos, i_neg = balanced_pairs(int(y_all.sum()),
                                          int((1 - y_all).sum()), rng)
            i_pos = np.where(y_all == 1)[0][i_pos]
            i_neg = np.where(y_all == 0)[0][i_neg]
            keep = np.concatenate([i_pos, i_neg])
            X, y = X_all[keep], y_all[keep]
        else:
            X, y = X_all, y_all
        if BALANCE:
            logger.info(f'--balance keeps {len(y)} trials, '
                        f'{int(y.sum())} per group')

    # Shuffle so that the class order inside a fold is not the trial order
    order = rng.permutation(len(y))
    X, y = X[order], y[order]

    times = epochs_pos.times
    patterns, _ = fit_all(X, y, len(ch_names))
    logger.info(f'fitted {patterns.shape[0]} time points, '
                f'{patterns.shape[1]} channels')

    # One time point is one fit, so the pattern jumps from sample to sample.
    # The components being decoded are tens of ms wide, so averaging along
    # time costs resolution that is not there anyway. The plain difference is
    # left alone, it is an average over trials and needs no help
    patterns = smooth_time(patterns, epochs_pos.info['sfreq'])
    norms = np.linalg.norm(patterns, axis=1)

    # ---- thin the time axis before writing ----
    step = max(int(round(args.step_ms / 1000.0 * epochs_pos.info['sfreq'])), 1)
    mask = ((times >= args.t_min) & (times <= args.t_max))[::step]
    idx = np.arange(len(times))[::step]
    idx = idx[mask]
    if len(idx) == 0:
        raise SystemExit('No time point inside --t-min / --t-max')

    unit = np.zeros_like(patterns)
    good = norms > EPS
    unit[good] = patterns[good] / norms[good, None]

    def save(mat, name):
        df = pd.DataFrame(mat[idx], columns=ch_names)
        df.insert(0, 'time', np.round(times[idx], 6))
        path = OUTPUT_DIR / f'{name}-{ANALYSIS}-{TAG}.csv'
        df.to_csv(path, index=False, float_format='%.6g')
        logger.info(f'Saved into {path}')

    save(patterns, 'patterns')
    save(unit, 'patterns-unit')
    save(diff.T, 'diff')

    df = pd.DataFrame(dict(time=np.round(times[idx], 6),
                           norm=np.round(norms[idx], 8)))
    path = OUTPUT_DIR / f'norm-{ANALYSIS}-{TAG}.csv'
    df.to_csv(path, index=False, float_format='%.6g')
    logger.info(f'Saved into {path}')

    pd.DataFrame(dict(ch_name=ch_names, ch_type=ch_types)).to_csv(
        OUTPUT_DIR / f'channels-{TAG}.csv', index=False)

    row = dict(mode=MODE, subject=SUBJ, analysis=ANALYSIS, tag=TAG,
               epochs_fname=args.epochs_fname,
               positive_class=classes[0], negative_class=classes[1],
               picks=args.picks, balance=int(BALANCE), haufe=int(args.haufe),
               smooth_ms=args.smooth_ms, step_ms=args.step_ms,
               seed=args.seed, n_channels=len(ch_names),
               n_positive=int(y.sum()), n_negative=int((1 - y).sum()),
               n_times=int(len(idx)),
               refilter_highpass=(args.highpass if args.highpass is not None
                                  else np.nan),
               refilter_lowpass=(args.lowpass if args.lowpass is not None
                                 else np.nan))
    path = OUTPUT_DIR / f'summary-{ANALYSIS}-{TAG}.csv'
    pd.DataFrame([row]).to_csv(path, index=False, float_format='%.6g')
    logger.info(f'Saved into {path}')
    print(pd.DataFrame([row]).T.to_string())
    return 0


main()


# %% ---- 2026-10-21 ------------------------
# Pending


# %% ---- 2026-10-21 ------------------------
# Pending
