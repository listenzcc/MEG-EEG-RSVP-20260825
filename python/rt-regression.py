"""
File: rt-regression.py
Author: Chuncheng Zhang
Date: 2026-10-15
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Fit reaction time as a continuous predictor on the target locked response,
    with the button press modelled as its own event, one subject of one
    modality at a time.

    The existing quick / slow split (stage 10, stage 19) throws the continuous
    reaction time away by cutting it at the median. This stage keeps it. The
    question it answers is whether the target locked EEG / MEG response varies
    with reaction time in a graded way, and whether that association survives
    once the button event is in the model.

    Two models are fitted with the same diagnostics:

        epochs       Every time point of the target locked epochs is regressed
                     across trials, so the design is built per time point and
                     const absorbs everything that is identical on every trial
                     (the whole stimulus locked response, the projection, the
                     subject's average drift). What is left to estimate is what
                     changes with reaction time.
        continuous   A finite impulse response deconvolution on the raw blocks.
                     Image onsets, target onsets, the reaction time modulation
                     of the target onset and the button onsets are separate
                     regressors. Needs the raw files and costs much more.

    Read the result with the identifiability report first. Reaction time is the
    interval between the target onset and the button press, so a target locked
    term scaled by reaction time and a button locked term are not automatically
    separable. Two things make them hard to tell apart in this data set:

        the reaction times are quantised, in MEG the most frequent value is the
        same number in all ten subjects, so a button locked response takes only
        a handful of shapes across trials and can hide inside the constant;
        almost every target trial carries a press, so there is no press free
        subset of target trials to anchor the button term with.

    The script therefore does not silently fit and report. It measures the
    design first (rank, condition number, variance inflation, the canonical
    correlation between the reaction time term and the button basis), it runs a
    simulation on the subject's own event timings to see whether a known
    modulation can be recovered and how much of a known button response leaks
    into the reaction time coefficient, and it writes all of that next to the
    coefficients. A coefficient coming out of a design that fails those checks
    is not evidence and should be reported as a null with a reason.

    This is an observational analysis. Nothing here licenses a causal reading,
    and a reaction time association is not a statement about perceptual speed:
    decision, motor preparation and the press itself all move with it.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-10-15 ------------------------
# Requirements and constants
from util.easy_imports import *

try:
    from scipy import sparse
    from scipy.sparse.linalg import lsqr
    HAVE_SCIPY = True
except ImportError:                                          # pragma: no cover
    HAVE_SCIPY = False


# %%
parser = argparse.ArgumentParser(
    description='Continuous reaction time regression with the button press '
                'modelled as its own event')
parser.add_argument('-s', '--subject', default='S02',
                    help='Subject name like S02')
parser.add_argument('-m', '--mode', default='EEG', help='Mode name EEG | MEG')
parser.add_argument('-t', '--tag', default='rma',
                    help='Short name of this run, used in filenames')
parser.add_argument('--model', default='epochs',
                    choices=['epochs', 'continuous'],
                    help="'epochs' regresses across trials at every time point "
                         "of the target epochs, 'continuous' deconvolves the "
                         "raw blocks. The epochs model needs the stage 3 "
                         "output only, the continuous one needs the raw files")
parser.add_argument('--epochs-fname',
                    default='epochs-1-notch-removal-artificial-epo.fif',
                    help='The target epochs to fit. Point it at '
                         'epochs-1-notch-epo.fif for the unprojected data')
parser.add_argument('--data-dir', default='output/epochs',
                    help='Folder containing the {MODE}-{SUBJ} folders')
parser.add_argument('--qs-dir', default='output/quick-slow',
                    help='Folder containing the trial table of stage 10')
parser.add_argument('--qs-tag', '--qs_tag', dest='qs_tag', default=None,
                    help='Tag of the trial table. Left out it follows --tag')
parser.add_argument('--raw-dir', default='rawData/RSVP_dataset/'
                                         'processed_data',
                    help='Folder containing the {MODE}_{SUBJ} raw folders, '
                         'continuous model only')
parser.add_argument('-o', '--output-dir', default='output/rt-regression',
                    help='Where the coefficients and the diagnostics go')
parser.add_argument('--button-window', '--button_window',
                    dest='button_window', type=float, nargs=2,
                    default=[0.0, 0.6],
                    help='The stretch after the button the finite impulse '
                         'response basis covers, in seconds')
parser.add_argument('--button-bin', '--button_bin', dest='button_bin',
                    type=float, default=0.05,
                    help='Width of one bin of the button basis, in seconds')
parser.add_argument('--rt-window', '--rt_window', dest='rt_window',
                    type=float, nargs=2, default=[-0.1, 0.8],
                    help='The stretch after the target the reaction time '
                         'modulation is estimated over, continuous model only')
parser.add_argument('--rt-bin', '--rt_bin', dest='rt_bin', type=float,
                    default=0.05,
                    help='Width of one bin of the target basis, in seconds')
parser.add_argument('--image-window', '--image_window', dest='image_window',
                    type=float, nargs=2, default=[0.0, 1.0],
                    help='The stretch after a non target image onset, '
                         'continuous model only')
parser.add_argument('--image-bin', '--image_bin', dest='image_bin',
                    type=float, default=0.1,
                    help='Width of one bin of the image basis, in seconds')
parser.add_argument('--add-quadratic', '--add_quadratic', dest='add_quad',
                    action='store_true',
                    help='Add an orthogonalised squared reaction time term, '
                         'which is one way to see whether the association is '
                         'graded rather than a step at the median')
parser.add_argument('--add-order', '--add_order', dest='add_order',
                    action='store_true',
                    help='Add the position of the trial in the run, '
                         'centred, as a slow drift term')
parser.add_argument('--rt-min', '--rt_min', dest='rt_min', type=float,
                    default=0.20,
                    help='Drop trials with a reaction time below this, in '
                         'seconds. Presses that follow the target this '
                         'quickly are anticipations, and the earliest press '
                         'of a subject is the line the whole analysis is read '
                         'against, so a single one of them would set it. '
                         'Use 0 to keep every trial')
parser.add_argument('--rt-outlier', '--rt_outlier', dest='rt_outlier',
                    type=float, default=None,
                    help='Drop trials whose reaction time is this far from the '
                         'subject median, in seconds. None keeps the inclusion '
                         'rules of stage 10 untouched')
parser.add_argument('--groups', default='all',
                    help="Which trials of the stage 10 table to fit, comma "
                         "separated. 'all' takes every labelled trial, which "
                         "is the continuous regression and the point of this "
                         "stage; 'quick,slow' reproduces the split sample")
parser.add_argument('--min-bin-trials', '--min_bin_trials',
                    dest='min_bin_trials', type=int, default=5,
                    help='A button bin whose number of active trials falls '
                         'below this at some time point is left out of the fit '
                         'there. A bin with one or two trials carries no '
                         'information and a least squares solver spreads the '
                         'coefficient over the near collinear columns instead, '
                         'which is how a design that looks fine leaks the '
                         'button response into the reaction time coefficient')
parser.add_argument('--cond-max', '--cond_max', dest='cond_max', type=float,
                    default=1e4,
                    help='Condition number above which the design is reported '
                         'as not identifiable')
parser.add_argument('--vif-max', '--vif_max', dest='vif_max', type=float,
                    default=10.0,
                    help='Variance inflation above which a column is reported '
                         'as collinear')
parser.add_argument('--cca-max', '--cca_max', dest='cca_max', type=float,
                    default=0.95,
                    help='Canonical correlation between the reaction time term '
                         'and the button basis above which the two are '
                         'reported as not separable')
parser.add_argument('--n-sim', '--n_sim', dest='n_sim', type=int, default=20,
                    help='How many simulated data sets the recovery check '
                         'averages over')
parser.add_argument('--seed', type=int, default=0,
                    help='Random state of the simulation')
args = parser.parse_args()
SUBJ = args.subject
MODE = args.mode
TAG = args.tag
QS_TAG = args.qs_tag if args.qs_tag else args.tag

logger.info(f'Start with {args=}')

# %%
DATA_DIR = Path(args.data_dir) / f'{MODE}-{SUBJ}'
QS_DIR = Path(args.qs_dir) / f'{MODE}-{SUBJ}'
RAW_DIR = Path(args.raw_dir) / f'{MODE}_{SUBJ}'
OUTPUT_DIR = Path(args.output_dir) / f'{MODE}-{SUBJ}'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

BUTTON_WINDOW = tuple(args.button_window)
RT_WINDOW = tuple(args.rt_window)
IMAGE_WINDOW = tuple(args.image_window)


# %% ---- 2026-10-15 ------------------------
# Function and class
def load_trials(n_trials: int):
    '''
    Read the trial table written by stage 10 and return the reaction times.

    The 'index' column points at the trial axis of the target epochs, 'delay'
    is the reaction time in seconds. Stage 10 writes one row per target trial
    and marks the middle third of a tertile split as 'drop', so the default
    'all' keeps those too: the continuous regression wants the whole
    distribution, not the two tails of it.

    Args:
        n_trials: The number of target epochs, used to drop stale indices

    Returns:
        df: The trial table, sorted by epoch index
    '''
    trial_csv = QS_DIR / f'quick-slow-trials-{QS_TAG}.csv'
    if not trial_csv.exists():
        raise SystemExit(f'{trial_csv} does not exist. Run '
                         f'10.quick-slow-analysis.sh for {MODE}-{SUBJ} first, '
                         f'or point --qs-tag at the table of these epochs.')

    df = pd.read_csv(trial_csv)
    if 'delay' not in df.columns:
        raise SystemExit(f'{trial_csv} has no delay column, so there is no '
                         f'reaction time to regress on')

    n_all = len(df)
    df = df[df['index'] < n_trials].copy()
    df = df[np.isfinite(df['delay'])]
    df = df[df['delay'] > 0]
    df['index'] = df['index'].astype(int)
    df = df.sort_values('index').reset_index(drop=True)
    logger.info(f'{n_all} rows, {len(df)} usable after index and delay checks')

    if args.groups != 'all':
        keep = [g.strip() for g in args.groups.split(',') if g.strip()]
        df = df[df['group'].isin(keep)].reset_index(drop=True)
        logger.info(f'{len(df)} trials left after keeping {keep}')

    if args.rt_min is not None:
        # A press that follows the target by less than this is an anticipation
        # rather than a response to it. They are few, a third of a percent of
        # the EEG trials and three hundredths of a percent of the MEG trials,
        # but the earliest press of a subject is the line this whole analysis
        # is read against, so one of them would set it. They are dropped here
        # and counted, so the choice can be seen and reversed.
        keep = df['delay'] >= args.rt_min
        logger.info(f'rt floor drops {(~keep).sum()} of {len(df)} trials '
                    f'faster than {args.rt_min} s, '
                    f'{df.loc[keep, "delay"].min():.3f} s is now the '
                    f'earliest press')
        df = df[keep].reset_index(drop=True)

    if args.rt_outlier is not None:
        med = df['delay'].median()
        keep = (df['delay'] - med).abs() <= args.rt_outlier
        logger.info(f'rt outlier rule drops {(~keep).sum()} trials further '
                    f'than {args.rt_outlier} s from the median {med:.3f} s')
        df = df[keep].reset_index(drop=True)

    if len(df) < 30:
        raise SystemExit(f'{MODE}-{SUBJ}: only {len(df)} trials left, too few '
                         f'for a per time point regression')
    return df


def rt_summary(rt: np.ndarray) -> dict:
    '''
    Describe the reaction time distribution, with the quantisation in mind.

    A reaction time that takes a small number of distinct values is the whole
    identifiability problem of this analysis, so the count of unique values and
    the share of the modal value are reported next to the ordinary quantiles.

    Args:
        rt: (n_trials,) reaction times in seconds

    Returns:
        out: The descriptive numbers
    '''
    values, counts = np.unique(np.round(rt, 6), return_counts=True)
    i = int(np.argmax(counts))
    return dict(
        n_trials=int(rt.size),
        rt_min=float(rt.min()),
        rt_q05=float(np.quantile(rt, 0.05)),
        rt_median=float(np.median(rt)),
        rt_q95=float(np.quantile(rt, 0.95)),
        rt_max=float(rt.max()),
        rt_iqr=float(np.subtract(*np.quantile(rt, [0.75, 0.25]))),
        rt_std=float(rt.std(ddof=1)),
        rt_n_unique=int(values.size),
        rt_modal_value=float(values[i]),
        rt_modal_share=float(counts[i] / counts.sum()),
        rt_unique_frac=float(values.size / rt.size),
    )


CONST_COL = 0


def usable_columns(X: np.ndarray, active: np.ndarray = None) -> np.ndarray:
    '''
    Which columns of one time point go into the fit.

    A column with no variation carries no information and, worse, makes the
    normal equations singular, so it is dropped. The constant is the exception:
    it has no variation by construction and it has to stay, because without an
    intercept the button columns end up carrying the whole stimulus locked
    response and the reaction time coefficient is no longer a modulation of
    anything.

    Args:
        X: (n_trials, n_cols) design at one time point
        active: (n_cols,) which columns are switched on, None for all

    Returns:
        keep: (n_cols,) boolean
    '''
    keep = X.std(axis=0) > 1e-12
    keep[CONST_COL] = True
    if active is not None:
        keep = keep & active
        keep[CONST_COL] = True
    return keep


def build_design(times: np.ndarray, rt: np.ndarray, lags: np.ndarray):
    '''
    Build the regressor names, the per time point design, and which columns
    carry information where.

    The design is the same at every time point except for the button basis,
    which depends on where the press of that trial falls relative to the time
    point being fitted. Every column except the constant is centred so that the
    constant keeps the whole stimulus locked response and the columns stay
    interpretable as modulations of it.

    A button bin only carries information while enough trials have a press at
    that lag. Before the earliest press of the subject every button column is
    identically zero, and a time point where only one or two trials sit in a
    bin is worse than useless: the solver spreads the coefficient over the near
    collinear columns, which is the mechanism by which a button response ends
    up reported as a reaction time effect. Bins below --min-bin-trials are
    therefore switched off at that time point and the fit runs without them.

    Args:
        times: (n_times,) epoch time axis in seconds
        rt: (n_trials,) reaction times in seconds
        lags: (n_times, n_trials) times minus reaction time

    Returns:
        names: The column names
        X: (n_times, n_trials, n_cols) the design
        active: (n_times, n_cols) whether the column is used at that time point
    '''
    n_times, n_trials = lags.shape
    rt_c = (rt - rt.mean()) / rt.std(ddof=1)
    n_button = int(round((BUTTON_WINDOW[1] - BUTTON_WINDOW[0])
                         / args.button_bin))
    cols = [np.ones(n_trials), rt_c]
    names = ['const', 'rt']

    if args.add_quad:
        q = rt_c ** 2
        q = q - q.mean()                     # keep it off the constant
        cols.append(q / q.std(ddof=1))
        names.append('rt2')

    if args.add_order:
        o = np.arange(n_trials, dtype=float)
        o = (o - o.mean()) / o.std(ddof=1)
        cols.append(o)
        names.append('order')

    n_cols = len(cols) + n_button
    X = np.zeros((n_times, n_trials, n_cols))
    active = np.ones((n_times, n_cols), dtype=bool)
    for j, c in enumerate(cols):
        X[:, :, j] = c[None, :]

    bin_of = np.floor((lags - BUTTON_WINDOW[0]) / args.button_bin).astype(int)
    inside = (lags >= BUTTON_WINDOW[0]) & (lags < BUTTON_WINDOW[1])
    for k in range(n_button):
        col = len(cols) + k
        hit = inside & (bin_of == k)
        active[:, col] = hit.sum(axis=1) >= args.min_bin_trials
        X[:, :, col] = np.where(hit, 1.0, 0.0)
        names.append(f'button_{BUTTON_WINDOW[0] + k * args.button_bin:+.2f}')

    # Centre the button columns. Without this a bin that is the same on every
    # trial is a copy of the constant and the fit is not defined.
    for col in range(len(cols), n_cols):
        m = X[:, :, col].mean(axis=1, keepdims=True)
        X[:, :, col] = np.where(active[:, col][:, None],
                                X[:, :, col] - m, 0.0)
        if active[:, col].sum() == 0:
            active[:, col] = False
    return names, X, active


def diagnose_one(X: np.ndarray, names: list, rt_col: int,
                 button_cols: list, active: np.ndarray = None) -> dict:
    '''
    Collinearity diagnostics of one time point.

    Three numbers matter and they answer different questions. The condition
    number says whether the design as a whole can be inverted at all. The
    variance inflation of the reaction time column says how much its standard
    error is inflated by the others. The canonical correlation between the
    reaction time column and the button basis says whether the two things this
    analysis is trying to separate are separable in principle, which is the one
    that decides whether the coefficient means anything.

    The last one is a property of the design that no amount of data removes.
    Reaction time is the interval between the target and the press, so a narrow
    reaction time distribution makes the button basis a proxy for reaction time
    itself, and the two terms then describe the same variance. The report is
    explicit about that number rather than assuming the model sorted it out.

    Args:
        X: (n_trials, n_cols) design at one time point
        names: Column names
        rt_col: Index of the reaction time column
        button_cols: Indices of the button columns
        active: (n_cols,) which columns are used; None uses all with variance

    Returns:
        out: rank, condition number, variance inflation, canonical correlation,
             and how many button columns are alive at this time point
    '''
    out = {}
    keep = usable_columns(X, active)
    Xk = X[:, keep]
    out['rank'] = int(np.linalg.matrix_rank(Xk))
    out['n_cols_used'] = int(keep.sum())
    out['n_cols'] = int(X.shape[1])
    out['n_degenerate'] = int((X.std(axis=0) > 1e-12).sum() - keep.sum())
    out['n_button_active'] = int(sum(keep[c] for c in button_cols))

    if Xk.shape[1] < 2:
        out['cond'] = np.inf
        out['vif_rt'] = np.inf
        out['cca_rt_button'] = np.nan
        return out

    s = np.linalg.svd(Xk, compute_uv=False)
    out['cond'] = float(s[0] / s[-1]) if s[-1] > 1e-14 * s[0] else np.inf

    # Variance inflation of every column in use. The constant has no variance
    # of its own and no variance inflation either, so it is left out.
    live = [c for c in range(X.shape[1]) if keep[c]]
    for col in live:
        if col == CONST_COL:
            continue
        others = np.delete(Xk, live.index(col), axis=1)
        b = np.linalg.lstsq(others, X[:, col], rcond=None)[0]
        r2 = 1.0 - np.sum((X[:, col] - others @ b) ** 2) / np.sum(
            (X[:, col] - X[:, col].mean()) ** 2)
        out[f'vif_{names[col]}'] = float(1.0 / max(1.0 - r2, 1e-12))

    B = X[:, [c for c in button_cols if keep[c]]]
    if B.shape[1] > 0 and np.linalg.matrix_rank(B) > 0:
        a = X[:, rt_col] - X[:, rt_col].mean()
        Bq, _ = np.linalg.qr(B)
        fit = Bq @ (Bq.T @ a)
        out['cca_rt_button'] = (float(abs(np.corrcoef(a, fit)[0, 1]))
                                if fit.std() > 1e-12 else 0.0)
    else:
        out['cca_rt_button'] = 0.0
    return out


def fit_epochs(data: np.ndarray, X: np.ndarray,
               active: np.ndarray = None) -> dict:
    '''
    Fit the design at every time point, across trials.

    Standard errors are only filled where the design of that time point can be
    inverted with a sane condition number. Reporting a t value out of a design
    that is rank deficient or nearly collinear would be worse than reporting
    nothing, so those time points keep a nan and the report says so.

    Args:
        data: (n_trials, n_ch, n_times) epochs
        X: (n_times, n_trials, n_cols) design
        active: (n_times, n_cols) which columns are used, None uses all

    Returns:
        out: beta, standard error and t values, all (n_cols, n_ch, n_times),
             plus the per time point condition number
    '''
    n_times, n_trials, n_cols = X.shape
    n_ch = data.shape[1]
    beta = np.full((n_cols, n_ch, n_times), np.nan)
    se = np.full_like(beta, np.nan)
    cond = np.full(n_times, np.nan)

    for i in range(n_times):
        Xi = X[i]
        keep = usable_columns(Xi, None if active is None else active[i])
        Xk = Xi[:, keep]
        if Xk.shape[1] < 2:
            continue
        s = np.linalg.svd(Xk, compute_uv=False)
        c = float(s[0] / s[-1]) if s[-1] > 1e-14 * s[0] else np.inf
        cond[i] = c

        Yi = data[:, :, i]                                   # (n_trials, n_ch)
        b, *_ = np.linalg.lstsq(Xk, Yi, rcond=None)
        beta[keep, :, i] = b
        if not np.isfinite(c) or c > args.cond_max:
            continue                   # the coefficient stands, the t does not

        resid = Yi - Xk @ b
        dof = max(n_trials - Xk.shape[1], 1)
        sigma2 = (resid ** 2).sum(axis=0) / dof
        try:
            XtXi = np.linalg.inv(Xk.T @ Xk)
        except np.linalg.LinAlgError:                        # pragma: no cover
            continue
        var = np.outer(np.diag(XtXi), sigma2)
        se[keep, :, i] = np.sqrt(np.maximum(var, 0))

    with np.errstate(invalid='ignore', divide='ignore'):
        tval = beta / se
    return dict(beta=beta, se=se, tval=tval, cond=cond)


def project_on_evoked(beta: np.ndarray, evoked: np.ndarray) -> np.ndarray:
    '''
    Project the coefficient maps on the subject's own target topography.

    A coefficient lives in sensor space, one number per sensor per time point,
    and averaging over sensors would cancel it. Projecting on the target evoked
    direction turns it into one signed curve per time point, in the units of
    the recording, which is what the group analysis and the figures use.

    Before the response has started the average is nearly flat, so there is no
    direction to project on and the ratio is a division of two noise terms.
    Those time points fall back to the time averaged topography, which keeps
    the curve signed and continuous; the report says which stretch that covers.

    Args:
        beta: (n_ch, n_times) one column of the fit
        evoked: (n_ch, n_times) the subject's target average

    Returns:
        curve: (n_times,) the signed projection
    '''
    den = np.sum(evoked ** 2, axis=0)
    flat = evoked.mean(axis=1, keepdims=True)
    weak = den < 1e-3 * np.nanmax(den)
    w = np.where(weak[None, :], flat, evoked)
    den = np.where(weak, np.sum(flat ** 2, axis=0), den)
    den = np.where(den < 1e-30, np.nan, den)
    return np.sum(beta * w, axis=0) / den


def simulate_recovery(data: np.ndarray, X: np.ndarray, rt: np.ndarray,
                      names: list, rt_col: int, times: np.ndarray,
                      active: np.ndarray,
                      rng: np.random.Generator) -> dict:
    '''
    Ask whether the design can recover what it is supposed to measure.

    Two simulations, both on the subject's own event timings and noise. The
    first puts a known reaction time modulation in and measures how well it
    comes back out. The second puts a known button locked response in and
    measures how much of it leaks into the reaction time coefficient, which is
    the number that decides whether a surviving association can be called
    stimulus locked at all.

    Both are split at the earliest press of the subject, because that is the
    line the design itself draws: before it no button column is alive and the
    coefficient is a clean estimate of the target locked modulation, after it
    the button response is in the model and can only be separated to the extent
    that the reaction time distribution allows.

    Args:
        data: (n_trials, n_ch, n_times) the real epochs, used as noise
        X: (n_times, n_trials, n_cols) the real design
        rt: (n_trials,) reaction times
        names: Column names
        rt_col: Index of the reaction time column
        times: (n_times,) epoch time axis
        active: (n_times, n_cols) which columns are used
        rng: Random generator

    Returns:
        out: Recovery and leakage numbers, clean window and mixed window
    '''
    n_trials, n_ch, n_times = data.shape
    rt_c = (rt - rt.mean()) / rt.std(ddof=1)
    noise = data.std(axis=(0, 2))
    noise = np.where(noise < 1e-30, 1e-30, noise)

    clean_until = float(rt.min())
    clean = times < clean_until
    mixed = ~clean

    # A broad envelope rather than a narrow bump. The question the simulation
    # answers is how the reaction time coefficient behaves on either side of
    # the earliest press, so the injected modulation has to exist on both
    # sides; a bump sitting inside one of the two windows would leave the other
    # with nothing to compare against.
    centre = float(np.clip(clean_until * 0.6, times[0] + 0.15, 0.35))
    shape = np.exp(-0.5 * ((times - centre) / 0.22) ** 2)
    shape = np.where(times > 0.0, shape, 0.0)
    shape = shape / np.max(np.abs(shape))

    # One reference for both windows, so the two leakage numbers are on the
    # same scale and can be read against each other.
    ref_all = float(np.sqrt(np.nanmean((shape[None, :] * (noise * 0.5)[:, None])
                                      ** 2)))

    corr = {'clean': [], 'mixed': []}
    gain = {'clean': [], 'mixed': []}
    leak = {'clean': [], 'mixed': []}
    null = {'clean': [], 'mixed': []}
    for _ in range(args.n_sim):
        eps = rng.standard_normal((n_trials, n_ch, n_times)) * noise[None, :,
                                                                     None]
        amp = noise * 0.5

        # 0: nothing at all, which is the level every other number has to be
        # read against. A leak that sits on the noise level is no leak.
        got0 = fit_epochs(eps, X, active)['beta'][rt_col]
        want = shape[None, :] * amp[:, None]
        for part, mask in [('clean', clean), ('mixed', mixed)]:
            if mask.sum() < 2 or ref_all <= 1e-30:
                continue
            null[part].append(
                float(np.sqrt(np.nanmean(got0[:, mask] ** 2)) / ref_all))

        # 1: a known reaction time modulation, no button response
        truth = want[None, :, :] * rt_c[:, None, None]
        got = fit_epochs(truth + eps, X, active)['beta'][rt_col]
        for part, mask in [('clean', clean), ('mixed', mixed)]:
            if mask.sum() < 2:
                continue
            for c in range(n_ch):
                a, b = got[c][mask], want[c][mask]
                # The threshold has to be relative. MEG is carried in
                # tesla and its standard deviation is far below 1e-12, so an
                # absolute test silently drops every channel and reports an
                # empty correlation.
                floor = 1e-12 * max(np.std(want[c][mask]), 1.0)
                if np.std(a) > floor and np.std(b) > floor:
                    corr[part].append(np.corrcoef(a, b)[0, 1])
            num = np.sum(got[:, mask] * want[:, mask])
            den = np.sum(want[:, mask] ** 2)
            gain[part].append(float(num / den) if den > 1e-30 else np.nan)

        # 2: a known button response, no reaction time modulation
        lags = times[None, :] - rt[:, None]                  # (n_trials, n_t)
        btn = np.exp(-0.5 * ((lags - 0.04) / 0.04) ** 2)
        btn = np.where((lags >= BUTTON_WINDOW[0]) & (lags < BUTTON_WINDOW[1]),
                       btn, 0.0)
        btn = btn[:, None, :] * amp[None, :, None]
        got2 = fit_epochs(btn + eps, X, active)['beta'][rt_col]
        for part, mask in [('clean', clean), ('mixed', mixed)]:
            if mask.sum() < 2 or ref_all <= 1e-30:
                continue
            leak[part].append(
                float(np.sqrt(np.nanmean(got2[:, mask] ** 2)) / ref_all))

    out = dict(clean_until_s=clean_until, injected_centre_s=float(centre),
               n_sim=args.n_sim)
    for part in ['clean', 'mixed']:
        out[f'recovery_corr_{part}'] = (float(np.nanmean(corr[part]))
                                        if corr[part] else np.nan)
        out[f'recovery_gain_{part}'] = (float(np.nanmean(gain[part]))
                                        if gain[part] else np.nan)
        for key, store in [('leak_from_button', leak), ('null_level', null)]:
            out[f'{key}_{part}'] = (float(np.nanmean(store[part]))
                                    if store[part] else np.nan)
        out[f'leak_over_null_{part}'] = (
            out[f'leak_from_button_{part}'] / out[f'null_level_{part}']
            if out.get(f'null_level_{part}', 0) > 1e-30 else np.nan)
    return out


# %% ---- 2026-10-15 ------------------------
# Play ground
if args.model == 'epochs':
    # ------------------------------------------------------------------
    # Epochs model: regress across trials at every time point
    # ------------------------------------------------------------------
    epochs = mne.read_epochs(DATA_DIR / args.epochs_fname, verbose=False)
    epochs.load_data()
    picks = mne.pick_types(epochs.info, meg=True, eeg=True, exclude=[])
    if picks.size == 0:
        picks = np.arange(len(epochs.ch_names))
    data = epochs.get_data(copy=True)                 # (n_tr, n_ch, n_times)
    times = epochs.times.copy()
    ch_names = [epochs.ch_names[p] for p in picks]
    data = data[:, picks, :]
    logger.info(f'{epochs.filename}: {data.shape}, {times[0]} to {times[-1]} s')

    df = load_trials(data.shape[0])
    rt = df['delay'].to_numpy(float)
    data = data[df['index'].to_numpy(int)]
    lags = times[None, :] - rt[:, None]               # (n_trials, n_times)

    names, X, active = build_design(times, rt, lags.T)
    rt_col = names.index('rt')
    button_cols = [i for i, n in enumerate(names) if n.startswith('button_')]

    # The share of trials whose press has already happened, which is what
    # decides whether a time point is a clean readout or a mixture
    button_frac = (lags >= 0).mean(axis=0)
    clean_until = float(rt.min())

    # ---- diagnostics on a few time points ----
    probe_t = [t for t in [-0.20, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60]
               if times[0] <= t <= times[-1]]
    rows = []
    for t in probe_t:
        i = int(np.argmin(np.abs(times - t)))
        d = diagnose_one(X[i], names, rt_col, button_cols, active[i])
        d['time'] = float(times[i])
        d['button_active_frac'] = float(button_frac[i])
        d['phase'] = 'clean' if button_frac[i] <= 0 else 'mixed'
        rows.append(d)
    diag = pd.DataFrame(rows)
    diag.to_csv(OUTPUT_DIR / f'identifiability-{TAG}.csv', index=False)

    def verdict(sub: pd.DataFrame):
        '''
        Judge a set of time points, and say which rule they failed.
        '''
        if len(sub) == 0:
            return True, 'no such time point'
        c = sub['cond'].replace(np.inf, np.nan).fillna(np.inf)
        v = sub['vif_rt'].replace(np.inf, np.nan).fillna(np.inf)
        g = sub['cca_rt_button'].fillna(1.0)
        why = []
        if (c >= args.cond_max).any():
            why.append(f'cond up to {c.max():.3g}')
        if (v >= args.vif_max).any():
            why.append(f'vif(rt) up to {v.max():.3g}')
        if (g >= args.cca_max).any():
            why.append(f'cca(rt,button) up to {g.max():.3f}')
        return len(why) == 0, '; '.join(why) or 'ok'

    ok_clean, why_clean = verdict(diag[diag['phase'] == 'clean'])
    ok_mixed, why_mixed = verdict(diag[diag['phase'] == 'mixed'])
    logger.info(f'clean up to {clean_until:.3f} s: '
                f'{"identifiable" if ok_clean else "NOT identifiable"} '
                f'({why_clean})')
    logger.info(f'mixed after that: '
                f'{"identifiable" if ok_mixed else "NOT identifiable"} '
                f'({why_mixed})')

    # ---- fit ----
    fit = fit_epochs(data, X, active)
    beta_rt = fit['beta'][rt_col]                     # (n_ch, n_times)
    tval_rt = fit['tval'][rt_col]

    evoked = data.mean(axis=0)                        # (n_ch, n_times)
    curve = project_on_evoked(beta_rt, evoked)
    curve_t = project_on_evoked(tval_rt, evoked)
    rms = np.sqrt(np.nanmean(beta_rt ** 2, axis=0))

    # ---- recovery ----
    rng = np.random.default_rng(args.seed)
    rec = simulate_recovery(data, X, rt, names, rt_col, times, active, rng)
    rec['clean_identifiable'] = bool(ok_clean)
    rec['mixed_identifiable'] = bool(ok_mixed)
    rec['clean_why'] = why_clean
    rec['mixed_why'] = why_mixed

    # ---- write ----
    audit = rt_summary(rt)
    audit.update(dict(mode=MODE, subject=SUBJ, tag=TAG, model=args.model,
                      n_channels=len(ch_names), n_times=len(times),
                      epochs_file=args.epochs_fname,
                      button_window=str(BUTTON_WINDOW),
                      button_bin_s=args.button_bin,
                      min_bin_trials=args.min_bin_trials,
                      n_button_cols=len(button_cols),
                      n_button_cols_ever_active=int(
                          active[:, button_cols].any(axis=0).sum()),
                      clean_until_s=clean_until,
                      add_quadratic=bool(args.add_quad),
                      add_order=bool(args.add_order)))
    audit.update({f'recovery_{k}': v for k, v in rec.items()})
    pd.DataFrame([audit]).to_csv(OUTPUT_DIR / f'audit-{TAG}.csv', index=False)

    curve_df = pd.DataFrame(dict(time=times, beta_rt_proj=curve,
                                 tval_rt_proj=curve_t, beta_rt_rms=rms,
                                 button_active_frac=button_frac))
    curve_df.to_csv(OUTPUT_DIR / f'beta-curve-{TAG}.csv', index=False)

    win_rows = []
    for wname, w in [('premotor', (0.10, 0.20)), ('early', (0.24, 0.36)),
                     ('late', (0.41, 0.55))]:
        m = (times >= w[0]) & (times <= w[1])
        win_rows.append(dict(
            mode=MODE, subject=SUBJ, tag=TAG, window=wname,
            win_lo=w[0], win_hi=w[1],
            beta=float(np.nanmean(curve[m])),
            tval=float(np.nanmean(curve_t[m])),
            rms=float(np.nanmean(rms[m])),
            button_active_frac_max=float(button_frac[m].max()),
            n_button_active_max=int(round(button_frac[m].max() * len(rt))),
            clean=bool(button_frac[m].max() == 0)))
    pd.DataFrame(win_rows).to_csv(OUTPUT_DIR / f'beta-windows-{TAG}.csv',
                                 index=False)
    logger.info('windows: ' + ', '.join(
        f'{r["window"]} beta {r["beta"]:.3g} '
        f'(button active up to {r["n_button_active_max"]})' for r in win_rows))

    # ---- figure ----
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax = axes[0]
    ax.plot(times, curve * (1e6 if MODE == 'EEG' else 1e15), color='C0',
            label='beta RT, projected')
    ax.axhline(0, color='k', lw=0.6)
    for wname, w in [('premotor', (0.10, 0.20)), ('early', (0.24, 0.36)),
                     ('late', (0.41, 0.55))]:
        ax.axvspan(w[0], w[1], color='grey', alpha=0.15)
    ax.axvspan(clean_until, times[-1], color='red', alpha=0.07)
    ax.axvline(clean_until, color='r', lw=0.8, ls='--')
    ax.annotate(f'earliest press {clean_until:.2f} s\n'
                f'right of this line the coefficient is a mixture',
                xy=(clean_until, np.nanmax(curve * 1e6 if MODE == 'EEG'
                                           else 1e15) * 0.8),
                fontsize=7, color='r')
    ax.set_ylabel('beta RT along the target topography\n'
                  '(fraction of the target average per SD of RT)')
    ax.set_title(f'{MODE}-{SUBJ} reaction time coefficient, {TAG} '
                 f'(projected on the target topography)\n'
                 f'clean: {"identifiable" if ok_clean else why_clean} | '
                 f'mixed: {"identifiable" if ok_mixed else why_mixed}')

    ax = axes[1]
    ax.plot(times, curve_t, color='C1')
    ax.axhline(0, color='k', lw=0.6)
    ax.axhline(1.96, color='r', lw=0.6, ls=':')
    ax.axhline(-1.96, color='r', lw=0.6, ls=':')
    for wname, w in [('premotor', (0.10, 0.20)), ('early', (0.24, 0.36)),
                     ('late', (0.41, 0.55))]:
        ax.axvspan(w[0], w[1], color='grey', alpha=0.15)
    ax.axvspan(clean_until, times[-1], color='red', alpha=0.07)
    ax.axvline(clean_until, color='r', lw=0.8, ls='--')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('t value, blank where the design fails the check')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'rt-regression-{TAG}.png', dpi=120)
    plt.close(fig)

    logger.info(f'wrote {OUTPUT_DIR}/{{audit,identifiability,beta-curve,'
                f'beta-windows}}-{TAG}.csv and rt-regression-{TAG}.png')

else:
    # ------------------------------------------------------------------
    # Continuous model: finite impulse response deconvolution on the raw
    # ------------------------------------------------------------------
    if not HAVE_SCIPY:
        raise SystemExit('the continuous model needs scipy for the sparse '
                         'design matrix')
    if not RAW_DIR.exists():
        raise SystemExit(f'{RAW_DIR} does not exist')

    files = sorted(RAW_DIR.glob('block_*_ica-raw.fif'))
    if len(files) < 2:
        raise SystemExit(f'{RAW_DIR} has {len(files)} block files, expected '
                         f'more than one')
    files = files[1:]                        # the first block is dropped,
    logger.info(f'{len(files)} block files')  # same rule as stage 1

    raws = [mne.io.read_raw_fif(f, preload=False, verbose=False)
            for f in files]
    for r in raws:
        if MODE == 'MEG':
            r.pick('mag')
            r.rename_channels({e: e.replace('-4503', '') for e in r.ch_names})
        else:
            r.pick([e for e in r.ch_names if e not in ['CB1', 'CB2']])
    raw = mne.concatenate_raws(raws, on_mismatch='ignore')
    raw.load_data(verbose=False)
    sfreq = raw.info['sfreq']
    decim = max(int(round(sfreq / 200.0)), 1)
    raw.filter(0.1, 40, n_jobs=n_jobs, verbose=False)
    ch_names = raw.ch_names
    data = raw.get_data()                                    # (n_ch, n_times)
    n_times = data.shape[1]

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    logger.info(f'{event_id}, {len(events)} events')

    def onsets(codes):
        out = []
        for c in codes:
            if str(c) in event_id:
                out.append(events[events[:, 2] == event_id[str(c)], 0])
        return (np.concatenate(out) if out
                else np.zeros(0, dtype=int)) / sfreq

    t_target = onsets([1])
    t_button = onsets([3])
    t_image = onsets([2, 4])
    logger.info(f'{t_target.size} targets, {t_button.size} presses, '
                f'{t_image.size} images')

    # Pair every target with the first press after it
    rt = np.full(t_target.size, np.nan)
    for i, t in enumerate(t_target):
        after = t_button[t_button >= t]
        if after.size:
            rt[i] = after[0] - t
    good = np.isfinite(rt) & (rt > 0) & (rt < 2.0)
    t_target, rt = t_target[good], rt[good]
    if args.rt_outlier is not None:
        keep = np.abs(rt - np.median(rt)) <= args.rt_outlier
        t_target, rt = t_target[keep], rt[keep]
    logger.info(f'{t_target.size} targets with a usable reaction time')


    def fir_block(onsets_s: np.ndarray, window: tuple, bin_s: float,
                  weight: np.ndarray = None):
        '''
        Sparse finite impulse response basis for one event type.
        '''
        n_bin = int(round((window[1] - window[0]) / bin_s))
        rows, cols, vals = [], [], []
        for j, t0 in enumerate(onsets_s):
            for k in range(n_bin):
                lo = t0 + window[0] + k * bin_s
                hi = lo + bin_s
                i0 = int(round(lo * sfreq))
                i1 = int(round(hi * sfreq))
                i0 = max(i0, 0)
                i1 = min(i1, n_times)
                if i1 <= i0:
                    continue
                w = 1.0 if weight is None else weight[j]
                rows.append(np.arange(i0, i1))
                cols.append(np.full(i1 - i0, k))
                vals.append(np.full(i1 - i0, w))
        if not rows:
            return sparse.csr_matrix((n_times, n_bin)), []
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        v = np.concatenate(vals)
        return (sparse.csr_matrix((v, (r, c)), shape=(n_times, n_bin)),
                [f'{window[0] + k * bin_s:+.2f}' for k in range(n_bin)])

    rt_c = (rt - rt.mean()) / rt.std(ddof=1)
    blocks, names = [], []
    Xi, nm = fir_block(t_image, IMAGE_WINDOW, args.image_bin)
    if Xi.shape[1]:
        blocks.append(Xi)
        names += [f'image_{n}' for n in nm]
    Xt, nm = fir_block(t_target, RT_WINDOW, args.rt_bin)
    blocks.append(Xt)
    names += [f'target_{n}' for n in nm]
    Xr, nm = fir_block(t_target, RT_WINDOW, args.rt_bin, weight=rt_c)
    blocks.append(Xr)
    names += [f'target_rt_{n}' for n in nm]
    Xb, nm = fir_block(t_button, BUTTON_WINDOW, args.button_bin)
    blocks.append(Xb)
    names += [f'button_{n}' for n in nm]
    Xo = sparse.csr_matrix(np.ones((n_times, 1)))
    blocks.insert(0, Xo)
    names.insert(0, 'const')
    X = sparse.hstack(blocks).tocsr()
    logger.info(f'design {X.shape}, {len(names)} columns')

    # Diagnostics from the normal equations, the full matrix never fits
    XtX = (X.T @ X).toarray()
    s = np.linalg.svd(XtX, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    rank = int(np.linalg.matrix_rank(XtX))
    rt_cols = [i for i, n in enumerate(names) if n.startswith('target_rt_')]
    btn_cols = [i for i, n in enumerate(names) if n.startswith('button_')]

    diag = dict(rank=rank, n_cols=len(names), n_degenerate=len(names) - rank,
                cond=cond)
    Xd = np.asarray(X.todense())
    keep = Xd.std(axis=0) > 1e-12
    sub = Xd[:, keep][:200000] if keep.sum() else None
    if sub is not None and sub.shape[1] > 1:
        d0 = diagnose_one(sub, [names[i] for i in range(len(names)) if keep[i]],
                          0, [])
        diag['worst_vif'] = float(np.nanmax(
            [v for k, v in d0.items() if k.startswith('vif_')]))
    diag_df = pd.DataFrame([diag])
    diag_df.to_csv(OUTPUT_DIR / f'identifiability-{TAG}.csv', index=False)
    identifiable = cond < args.cond_max and rank == len(names)

    # Fit by the normal equations, one channel at a time is the same thing
    XtY = X.T @ data.T                                       # (p, n_ch)
    pinv_XtX = np.linalg.pinv(XtX)
    beta = pinv_XtX @ XtY                                    # (p, n_ch)

    # Turn the reaction time bins back into a time course
    n_rt = len(rt_cols)
    rt_times = RT_WINDOW[0] + (np.arange(n_rt) + 0.5) * args.rt_bin
    beta_rt = beta[rt_cols]                                  # (n_rt, n_ch)
    scale = 1e6 if MODE == 'EEG' else 1e15
    curve = np.sqrt(np.nanmean((beta_rt * scale) ** 2, axis=1))

    audit = rt_summary(rt)
    audit.update(dict(mode=MODE, subject=SUBJ, tag=TAG, model=args.model,
                      n_channels=len(ch_names), n_times=int(n_times),
                      sfreq=float(sfreq), decim=decim,
                      n_target=len(t_target), n_button=len(t_button),
                      n_image=len(t_image), design_cond=cond,
                      design_rank=rank, n_cols=len(names),
                      identifiable=bool(identifiable)))
    pd.DataFrame([audit]).to_csv(OUTPUT_DIR / f'audit-{TAG}.csv', index=False)

    pd.DataFrame(dict(time=rt_times, beta_rt_rms=curve)).to_csv(
        OUTPUT_DIR / f'beta-curve-{TAG}.csv', index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rt_times, curve, color='C0')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('time from target onset (s)')
    ax.set_ylabel(f'|beta RT| rms ({"uV" if MODE == "EEG" else "fT"})')
    ax.set_title(f'{MODE}-{SUBJ} continuous deconvolution, {TAG}')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'rt-regression-{TAG}.png', dpi=120)
    plt.close(fig)

    logger.info(f'wrote {OUTPUT_DIR}/{{audit,identifiability,beta-curve}}-'
                f'{TAG}.csv and rt-regression-{TAG}.png')

# %% ---- 2026-10-15 ------------------------
# Pending
