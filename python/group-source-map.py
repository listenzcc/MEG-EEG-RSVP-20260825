"""
File: group-source-map.py
Author: Chuncheng Zhang
Date: 2026-09-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Group level source map over all the subjects.

    Stage 8 writes one stc per subject on the fsaverage template, so every
    subject already lives on the same set of vertices and the maps can be
    averaged directly. This script does four things with them:

    1. It normalizes every subject, by default into the z score against the
       pre stimulus baseline of the same vertex, so a subject with a strong
       inverse solution does not dominate the group map.
    2. It averages the normalized maps into the group map and tests the mean
       against zero across the subjects, which gives a t map and a p map.
    3. It reports where and when the group map peaks, as a latency, an MNI
       coordinate and, when the fsaverage annotation is readable, a list of
       the labels that carry the response.
    4. It writes the group map back as an stc, so the same map can be plotted
       in three dimensions later without keeping the single subject files.

    With --contrast_epochs a second condition of the same subject is
    subtracted before the normalization, which turns the map into the
    difference between two conditions, e.g. target minus non target. The
    statistics then test whether the difference is there at all, which is
    the claim the article makes about the 0.3 s component.

    Two corrections are reported and they answer different questions. The
    vertex wise false discovery rate keeps the spatial detail and is the
    honest description of a map whose effects are focal. The permutation
    test runs on the label by time matrix, not on the single vertices: with
    two hemispheres and a few hundred samples there are millions of vertex
    time points, and the maximum of that many statistics is a fluctuation
    rather than a response, even inside the baseline window. The labels
    reduce it to a few thousand tests, which is what a correction of the
    maximum statistic can carry.

    The anatomical part needs the fsaverage annotation and MRI. Both are
    optional, when they are missing the map and the false discovery rate are
    still written and only the labels, the MNI coordinate and the
    permutation test stay empty.

Usage:
    python python/group-source-map.py -m MEG -e epochs-1-notch-epo.fif -t ave
    python python/group-source-map.py -m EEG \
        -e epochs-1-notch-removal-artificial-epo.fif -t ave \
        -c epochs-2-notch-removal-artificial-epo.fif
    python python/group-source-map.py -m MEG -e epochs-1-epo.fif \
        -t ssvep10-power --normalize none

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-09-21 ------------------------
# Requirements and constants
from scipy import stats

from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Group level source map over all the subjects')
parser.add_argument('-m', '--mode', default='MEG', help='Mode name EEG | MEG')
parser.add_argument('-e', '--epochs_fname',
                    default='epochs-1-notch-removal-artificial-epo.fif',
                    help='The epochs the source map was estimated from')
parser.add_argument('-t', '--tag', default='ave',
                    help='ave | ssvep10-evoked | ssvep10-power')
parser.add_argument('-c', '--contrast_epochs', default=None,
                    help='Subtract this epochs of the same subject before the '
                         'normalization, e.g. the non target epochs')
parser.add_argument('-s', '--subjects', nargs='*', default=None,
                    help='Subject names like S01, every subject found on the '
                         'disk by default')

# The normalization
parser.add_argument('--normalize', default='baseline',
                    choices=['baseline', 'rms', 'none'],
                    help='baseline = z score against the baseline of the same '
                         'vertex, rms = divide by the root mean square of the '
                         'whole map, none = keep the physical unit')
parser.add_argument('--baseline', type=float, nargs=2, default=[-0.5, 0.],
                    help='The baseline window of the --normalize baseline')

# What to report
parser.add_argument('--search-window', type=float, nargs=2,
                    default=[0.05, 0.8],
                    help='Where the peak of the group map is looked for')
parser.add_argument('--label-window', type=float, nargs=2, default=None,
                    help='The window the labels are averaged in, 50 ms to '
                         'each side of the peak by default. A wide window '
                         'dilutes a transient component into its own noise, '
                         'so the default is a description of the peak rather '
                         'than a test of it')
parser.add_argument('--top-labels', type=int, default=12,
                    help='How many labels the figure and the table keep')
parser.add_argument('--annot', default='aparc',
                    help="The fsaverage annotation, 'none' to skip the anatomy")

# The statistics
parser.add_argument('--fdr', type=float, default=0.05,
                    help='The false discovery rate of the vertex wise test')
parser.add_argument('--n-perm', type=int, default=0,
                    help='The number of sign flips of the label wise maximum '
                         'statistic test, 0 skips it')
parser.add_argument('--perm-window', type=float, nargs=2, default=[0.0, 0.8],
                    help='The window the permutation test runs in')

parser.add_argument('--brain', action='store_true',
                    help='Also try to render the group map on the surface, '
                         'it needs a 3d backend to be installed')
parser.add_argument('-o', '--output-dir', default='output/group-source-map',
                    help='Where the group maps are written')

args = parser.parse_args()
MODE = args.mode
EPOCHS_FNAME = args.epochs_fname
TAG = args.tag
CONTRAST_FNAME = args.contrast_epochs

logger.info(f'Start with {args=}')

# %%
DATA_DIR = Path('output/source-estimation')
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# The name every output of this run shares
NAME = f'{MODE}-{EPOCHS_FNAME}'
if CONTRAST_FNAME is not None:
    NAME += f'-minus-{CONTRAST_FNAME}'

# %% ---- 2026-09-21 ------------------------
# Function and class


def stc_stem(mode: str, subj: str, epochs_fname: str, tag: str):
    '''
    Find the stc of one subject.

    The stem is the name without the hemisphere, mne appends -lh.stc and
    -rh.stc to it when it reads the file. Stage 8 saves the stc under
    {epochs}.{tag}.stc, so both that stem and the one without the trailing
    .stc are accepted.

    Args:
        mode: MEG or EEG
        subj: The subject name
        epochs_fname: The epochs the source map was estimated from
        tag: ave | ssvep10-evoked | ssvep10-power

    Returns:
        stem: The Path to read, None when the subject is not there
    '''
    folder = DATA_DIR / f'{mode}-{subj}'
    for stem in (folder / f'{epochs_fname}.{tag}.stc',
                 folder / f'{epochs_fname}.{tag}'):
        if Path(str(stem) + '-lh.stc').exists():
            return stem
    return None


def load_subjects():
    '''
    Read the stc of every subject, with the contrast subtracted when there
    is one.

    Returns:
        data: Array of shape (n_subjects, n_vertices, n_times)
        template: The stc of the first subject, it carries the vertices and
            the time axis
        subjects: The names of the subjects that are in the array
    '''
    if args.subjects is None:
        folders = sorted(DATA_DIR.glob(f'{MODE}-*'))
        subjects = [p.name.split('-')[-1] for p in folders]
    else:
        subjects = list(args.subjects)

    blocks, template, kept = [], None, []
    for subj in subjects:
        stem = stc_stem(MODE, subj, EPOCHS_FNAME, TAG)
        if stem is None:
            logger.warning(f'{MODE}-{subj} has no {EPOCHS_FNAME}.{TAG}')
            continue
        stc = mne.read_source_estimate(str(stem))
        data = stc.data.astype(np.float64)

        if CONTRAST_FNAME is not None:
            stem_b = stc_stem(MODE, subj, CONTRAST_FNAME, TAG)
            if stem_b is None:
                logger.warning(
                    f'{MODE}-{subj} has no {CONTRAST_FNAME}.{TAG}, the '
                    'subject is left out of the contrast')
                continue
            stc_b = mne.read_source_estimate(str(stem_b))
            if stc_b.shape != stc.shape:
                logger.warning(
                    f'{MODE}-{subj}: {stc_b.shape} does not match '
                    f'{stc.shape}, the subject is left out')
                continue
            data = data - stc_b.data.astype(np.float64)

        if template is None:
            template = stc
        elif stc.data.shape != template.data.shape:
            logger.warning(f'{MODE}-{subj} has another shape, left out')
            continue

        blocks.append(data)
        kept.append(subj)

    if not blocks:
        raise SystemExit(
            f'No stc is found, {DATA_DIR=}, {MODE=}, {EPOCHS_FNAME=}, {TAG=}')

    return np.array(blocks), template, kept


def normalize(data: np.ndarray, times: np.ndarray):
    '''
    Make the subjects comparable before they are averaged.

    The baseline version is the z score of every vertex against the same
    vertex in the baseline window, which removes both the overall scale of
    the inverse solution and the offset of the vertex. The denominator is
    floored at a tenth of the median deviation, otherwise a vertex whose
    baseline barely moves turns a tiny response into a huge z score.

    A vertex whose baseline does not move at all has no scale to be divided
    by, its z score would be a huge number produced by the floor of the
    denominator alone. Those vertices are set to zero, they carry no
    measurement.

    Args:
        data: Array of shape (n_subjects, n_vertices, n_times)
        times: The time axis of the last dimension

    Returns:
        out: The normalized array, the same shape
    '''
    if args.normalize == 'none':
        return data

    if args.normalize == 'rms':
        scale = np.sqrt((data ** 2).mean(axis=(1, 2), keepdims=True))
        return data / np.maximum(scale, 1e-30)

    mask = (times >= args.baseline[0]) & (times <= args.baseline[1])
    if not mask.any():
        raise SystemExit(f'The baseline window {args.baseline} has no sample')

    base = data[:, :, mask]
    mu = base.mean(axis=2, keepdims=True)
    sd = base.std(axis=2, keepdims=True)
    floor = 0.1 * np.median(sd)

    # The floor is a rescue for the vertices that barely move, it must not
    # turn them into the strongest vertices of the map
    flat = (sd <= floor).squeeze(axis=2)
    out = (data - mu) / np.maximum(sd, floor)
    out[flat] = 0.
    if flat.any():
        logger.warning(f'{int(flat.sum())} of {flat.size} subject vertices '
                       'do not move in the baseline, they are set to zero')
    return out


def label_indices(stc, labels):
    '''
    The vertex indices of every label inside the vertices of a source
    estimate.

    Args:
        stc: The source estimate whose vertices are used
        labels: The labels of the annotation

    Returns:
        out: A list of (name, hemisphere, indices)
    '''
    n_lh = len(stc.lh_vertno)
    out = []
    for label in labels:
        idx = []
        for h, vertno in enumerate(stc.vertices):
            hemi = 'lh' if h == 0 else 'rh'
            if label.hemi != 'both' and label.hemi != hemi:
                continue
            idx.append(np.where(np.isin(vertno, label.vertices))[0]
                       + (0 if h == 0 else n_lh))
        idx = np.concatenate(idx) if idx else np.array([], int)
        if idx.size:
            out.append((label.name, label.hemi, idx))
    return out


def bh_fdr(p: np.ndarray, alpha: float):
    '''
    The Benjamini Hochberg threshold of a set of p values.

    Args:
        p: The p values, the nan ones are ignored
        alpha: The false discovery rate

    Returns:
        p_crit: The largest p value that is still rejected, nan when none is
        n_reject: How many p values pass it
    '''
    flat = np.sort(p[np.isfinite(p)])
    n = flat.size
    if n == 0:
        return np.nan, 0
    below = flat <= alpha * (np.arange(1, n + 1) / n)
    if not below.any():
        return np.nan, 0
    k = int(np.max(np.nonzero(below)[0]))
    return float(flat[k]), int(k + 1)


def maximum_statistic(data: np.ndarray, window: np.ndarray):
    '''
    The maximum |t| of the one sample test against zero, and the p value of
    that maximum under the sign flips of the subjects.

    The null hypothesis is that the map is zero everywhere, in which case
    flipping the sign of a subject changes nothing, so the sign flips are an
    exact permutation of the test.

    The input is the label by time matrix rather than the single vertices.
    Millions of vertex time points make the maximum statistic useless, its
    null distribution is set by the number of tests and not by the response,
    a few thousand tests are what this correction can carry.

    Args:
        data: Array of shape (n_subjects, n_labels, n_times)
        window: The mask of the time samples that are tested

    Returns:
        t_max: The maximum |t| of the observed matrix
        p: The corrected p value of t_max, nan without the permutations
        null: The maximum |t| of every permutation, empty without them
        where: The (label, time) index of t_max
    '''
    block = data[:, :, window]
    n = block.shape[0]
    t = stats.ttest_1samp(block, 0., axis=0).statistic
    where = np.unravel_index(int(np.nanargmax(np.abs(t))), t.shape)
    t_max = float(np.abs(t[where]))

    if args.n_perm <= 0:
        return t_max, np.nan, np.array([]), where

    rng = np.random.default_rng(0)
    null = np.empty(args.n_perm)
    for i in range(args.n_perm):
        signs = rng.choice((-1., 1.), size=n)
        flipped = block * signs[:, None, None]
        null[i] = np.nanmax(np.abs(stats.ttest_1samp(flipped, 0., axis=0)
                                   .statistic))

    p = (1. + int((null >= t_max).sum())) / (1. + args.n_perm)
    return t_max, float(p), null, where


def upsert(row: dict, fpath: Path, keys=('mode', 'epochs_fname', 'tag',
                                         'contrast_epochs')):
    '''
    Write one row into the summary csv, replacing the older row of the same
    condition if there is one, so rerunning never duplicates rows.
    '''
    if fpath.exists():
        df = pd.read_csv(fpath)
        # A key column the old table does not have yet is filled with the
        # value that stands for the default run, otherwise rerunning the
        # default appends a second row instead of replacing it.
        for k in keys:
            if k not in df.columns and k in row:
                df[k] = '' if row[k] is None else row[k]
        same = np.ones(len(df), bool)
        for k in keys:
            if row.get(k) is None:
                same &= df.get(k, pd.Series(index=df.index)).isna().values
                continue
            if k not in df.columns:
                same &= False
                continue
            same &= (df[k].astype(str) == str(row[k])).values
        df = df[~same]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(fpath, index=False)


# %% ---- 2026-09-21 ------------------------
# Play ground
data, template, subjects = load_subjects()
times = template.times
logger.info(f'{len(subjects)} subjects: {subjects}')

data = normalize(data, times)
logger.info(f'{args.normalize=}, {args.baseline=}')

# The group map and the vertex wise test against zero
group_z = data.mean(axis=0)
ttest = stats.ttest_1samp(data, 0., axis=0)
t_map, p_map = ttest.statistic, ttest.pvalue

stc_z = template.copy()
stc_z.data = group_z
stc_z.subject = 'fsaverage'
stc_t = template.copy()
stc_t.data = t_map
stc_t.subject = 'fsaverage'

fname_z = OUTPUT_DIR / f'group-{NAME}.{TAG}-z.stc'
fname_t = OUTPUT_DIR / f'group-{NAME}.{TAG}-t.stc'
stc_z.save(fname_z, overwrite=True)
stc_t.save(fname_t, overwrite=True)
logger.info(f'Saved into {fname_z} and {fname_t}')

# %% ---- 2026-09-21 ------------------------
# The anatomy, it is optional
SUBJECTS_DIR, labels = None, []
if args.annot.lower() != 'none':
    try:
        fs_dir = mne.datasets.fetch_fsaverage(verbose='ERROR')
        SUBJECTS_DIR = os.path.dirname(fs_dir)
        labels = mne.read_labels_from_annot(
            'fsaverage', parc=args.annot, subjects_dir=SUBJECTS_DIR,
            verbose='ERROR')
        logger.info(f'{len(labels)} labels from {args.annot}')
    except Exception as e:
        logger.warning(f'The annotation is not available, {e}')

parcels = label_indices(template, labels) if len(labels) else []
logger.info(f'{len(parcels)} labels have vertices in the source space')

# %% ---- 2026-09-21 ------------------------
# Where and when the group map peaks
search = (times >= args.search_window[0]) & (times <= args.search_window[1])

# The latency comes from the vertex averaged curve, the strongest single
# vertex of the whole map is dominated by noise: with two hemispheres and a
# few hundred samples there are millions of vertex time points in the search
# window and the largest of them is a fluctuation rather than the response.
# The place is then read at that latency.
curve = np.sqrt((group_z ** 2).mean(axis=0))
inside = np.where(search)[0]
i_time = inside[int(np.argmax(curve[inside]))]
i_vertex = int(np.argmax(np.abs(group_z[:, i_time])))

n_lh = len(template.lh_vertno)
hemi = 'lh' if i_vertex < n_lh else 'rh'
vertex = int((template.lh_vertno if hemi == 'lh' else template.rh_vertno)
             [i_vertex if hemi == 'lh' else i_vertex - n_lh])

peak = dict(peak_time=round(float(times[i_time]), 4),
            peak_rms_z=float(f'{curve[i_time]:.4g}'),
            peak_z=float(f'{group_z[i_vertex, i_time]:.4g}'),
            peak_t=float(f'{t_map[i_vertex, i_time]:.4g}'),
            peak_p=float(f'{p_map[i_vertex, i_time]:.4g}'),
            peak_hemi=hemi, peak_vertex=vertex)

logger.info(f'The group map peaks at {times[i_time]:.3f} s, '
            f'rms z {curve[i_time]:.3g}, the strongest vertex is {hemi} '
            f'{vertex} at z {group_z[i_vertex, i_time]:.3g}')

# The labels are described in a narrow window around the peak, a transient
# component averaged over the whole search window is diluted by the samples
# where it is not there
if args.label_window is None:
    label_window = [max(times[0], times[i_time] - .05),
                    min(times[-1], times[i_time] + .05)]
else:
    label_window = list(args.label_window)
labels_mask = ((times >= label_window[0]) & (times <= label_window[1]))
logger.info(f'The labels are averaged in '
            f'[{label_window[0]:.3f}, {label_window[1]:.3f}] s')

# The MNI coordinate of the peak, the fsaverage MRI is needed for it
for key in ('peak_mni_x', 'peak_mni_y', 'peak_mni_z'):
    peak[key] = np.nan
if SUBJECTS_DIR is not None:
    try:
        mni = mne.vertex_to_mni([vertex], [0 if hemi == 'lh' else 1],
                                'fsaverage', subjects_dir=SUBJECTS_DIR)
        peak['peak_mni_x'] = round(float(mni[0][0]), 1)
        peak['peak_mni_y'] = round(float(mni[0][1]), 1)
        peak['peak_mni_z'] = round(float(mni[0][2]), 1)
        logger.info(f'The peak is at MNI {mni[0].round(1)}')
    except Exception as e:
        logger.warning(f'The MNI coordinate is not available, {e}')

# %% ---- 2026-09-21 ------------------------
# The vertex wise false discovery rate inside the search window
p_crit, n_reject = bh_fdr(p_map[:, inside], args.fdr)
peak['fdr_alpha'] = args.fdr
peak['fdr_p_crit'] = p_crit
peak['fdr_n_reject'] = n_reject
peak['fdr_n_tests'] = int(p_map[:, inside].size)
if np.isfinite(p_crit):
    logger.info(f'FDR {args.fdr}: {n_reject} of {p_map[:, inside].size} '
                f'vertex time points pass p <= {p_crit:.3g}')
else:
    logger.info(f'FDR {args.fdr}: nothing passes, the smallest p is '
                f'{np.nanmin(p_map[:, inside]):.3g}')

# %% ---- 2026-09-21 ------------------------
# The labels
rows = []
for name, parcel_hemi, idx in parcels:
    rows.append(dict(
        label=name, hemi=parcel_hemi, n_vertices=idx.size,
        mean_z=float(f'{data[:, idx][:, :, labels_mask].mean():.4g}'),
        peak_z=float(f'{group_z[idx, i_time].mean():.4g}')))

df_labels = pd.DataFrame(rows)
fname_labels = OUTPUT_DIR / f'group-{NAME}.{TAG}-labels.csv'
df_labels.to_csv(fname_labels, index=False)
logger.info(f'Saved {len(df_labels)} labels into {fname_labels}')

top = pd.DataFrame()
if len(df_labels):
    top = df_labels.reindex(
        df_labels['mean_z'].abs().sort_values(ascending=False).index)
    best = top.iloc[0]
    peak['top_label'] = f"{best['hemi']}-{best['label']}"
    peak['top_label_mean_z'] = float(best['mean_z'])
    logger.info('The strongest labels are ' + ', '.join(
        f"{r.hemi}-{r.label} {r.mean_z:+.2f}"
        for r in top.head(5).itertuples()))
else:
    peak['top_label'] = ''
    peak['top_label_mean_z'] = np.nan

# The label by time matrix of every subject, the permutation test runs on it
roi, roi_names = None, []
if len(parcels):
    roi = np.array([data[:, idx, :].mean(axis=1)
                    for _, _, idx in parcels])
    roi = np.swapaxes(roi, 0, 1)          # (n_subjects, n_labels, n_times)
    roi_names = [f'{h}-{n}' for n, h, _ in parcels]
    logger.info(f'The label matrix is {roi.shape}')

# %% ---- 2026-09-21 ------------------------
# The label wise maximum statistic and its sign flip permutation
for key in ('roi_t_max', 'roi_perm_p', 'roi_peak_label', 'roi_peak_time',
            'roi_fdr_p_crit', 'roi_fdr_n_reject', 'roi_fdr_n_tests',
            'roi_fdr_label', 'roi_fdr_time'):
    peak[key] = np.nan
null = np.array([])
perm_window = ((times >= args.perm_window[0])
               & (times <= args.perm_window[1]))

if roi is not None:
    # The same correction on the label matrix, it is the sensitive one, the
    # maximum statistic above is the conservative one
    roi_ttest = stats.ttest_1samp(roi[:, :, perm_window], 0., axis=0)
    roi_p_crit, roi_n = bh_fdr(roi_ttest.pvalue, args.fdr)
    peak['roi_fdr_p_crit'] = roi_p_crit
    peak['roi_fdr_n_reject'] = roi_n
    peak['roi_fdr_n_tests'] = int(roi_ttest.pvalue.size)
    if np.isfinite(roi_p_crit):
        where = np.unravel_index(
            int(np.nanargmax(np.abs(roi_ttest.statistic)
                             * (roi_ttest.pvalue <= roi_p_crit))),
            roi_ttest.statistic.shape)
        peak['roi_fdr_label'] = roi_names[where[0]]
        peak['roi_fdr_time'] = round(float(times[perm_window][where[1]]), 4)
        logger.info(f'Label FDR {args.fdr}: {roi_n} of '
                    f'{roi_ttest.pvalue.size} pass p <= {roi_p_crit:.3g}, '
                    f'the strongest is {roi_names[where[0]]} at '
                    f'{times[perm_window][where[1]]:.3f} s')
    else:
        peak['roi_fdr_label'] = ''
        peak['roi_fdr_time'] = np.nan
        logger.info(f'Label FDR {args.fdr}: nothing passes')

if roi is not None:
    t_max, p_max, null, where = maximum_statistic(roi, perm_window)
    peak['roi_t_max'] = float(f'{t_max:.4g}')
    peak['roi_perm_p'] = p_max
    peak['roi_peak_label'] = roi_names[where[0]]
    peak['roi_peak_time'] = round(float(times[perm_window][where[1]]), 4)
    peak['n_perm'] = args.n_perm
    logger.info(f'The label matrix peaks at {roi_names[where[0]]}, '
                f'{times[perm_window][where[1]]:.3f} s, |t| {t_max:.2f}'
                + (f', sign flip p {p_max:.4f} of {args.n_perm}'
                   if args.n_perm > 0 else ', no permutation was asked'))
else:
    peak['n_perm'] = 0
    if args.n_perm > 0:
        logger.warning('The permutation test needs the annotation, it is '
                       'skipped')

# %% ---- 2026-09-21 ------------------------
# The summary table
row = dict(mode=MODE, epochs_fname=EPOCHS_FNAME, tag=TAG,
           contrast_epochs=CONTRAST_FNAME, n_subjects=len(subjects),
           subjects=' '.join(subjects), normalize=args.normalize,
           baseline=f'{args.baseline[0]:g} {args.baseline[1]:g}',
           **peak)
df_summary = pd.DataFrame([row])
display(df_summary)

fname_summary = OUTPUT_DIR / f'group-{NAME}.{TAG}-summary.csv'
df_summary.to_csv(fname_summary, index=False)
logger.info(f'Saved into {fname_summary}')
upsert(row, OUTPUT_DIR / 'group-source-summary.csv')

# %% ---- 2026-09-21 ------------------------
# The figure
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
title = f'{MODE} {EPOCHS_FNAME}.{TAG}'
if CONTRAST_FNAME is not None:
    title += f' minus {CONTRAST_FNAME}'
fig.suptitle(f'{title}, {len(subjects)} subjects, '
             f'{args.normalize} normalized')

# The vertex averaged amplitude of the group map
ax = axes[0, 0]
ax.plot(times, curve, lw=1.8, color='tab:blue', label='rms of the group z')
ax.plot(times, group_z.mean(axis=0), lw=1., color='tab:orange', alpha=.8,
        label='mean of the group z')
ax.axvspan(args.search_window[0], args.search_window[1], color='tab:blue',
           alpha=.06)
if args.normalize == 'baseline':
    ax.axvspan(args.baseline[0], args.baseline[1], color='gray', alpha=.08,
               label='baseline')
ax.axvline(0, color='k', ls=':', lw=.8)
ax.axvline(peak['peak_time'], color='tab:red', ls='--', lw=1.,
           label=f"peak {peak['peak_time']:.3f} s")
ax.set_xlabel('Time (s)')
ax.set_ylabel('Group z')
ax.set_title('The group map over time')
ax.legend(fontsize=8, loc='upper right')

# The labels that carry the response
ax = axes[0, 1]
if len(top):
    show = top.head(args.top_labels).iloc[::-1]
    ax.barh(np.arange(len(show)), show['mean_z'],
            color=['tab:red' if v < 0 else 'tab:blue'
                   for v in show['mean_z']])
    ax.set_yticks(np.arange(len(show)))
    ax.set_yticklabels([f"{r.hemi}-{r.label}" for r in show.itertuples()],
                       fontsize=8)
    ax.axvline(0, color='k', lw=.8)
    ax.set_xlabel(f'Mean z in [{label_window[0]:.3f}, '
                  f'{label_window[1]:.3f}] s')
    ax.set_title(f'The {len(show)} strongest labels of {args.annot}')
else:
    ax.text(.5, .5, 'no annotation', ha='center', va='center')
    ax.set_title('The labels')

# The time course of the strongest labels
ax = axes[1, 0]
if roi is not None and len(top):
    order = [roi_names.index(f"{r.hemi}-{r.label}")
             for r in top.head(args.top_labels).itertuples()]
    image = roi[:, order, :].mean(axis=0)
    vmax = float(np.nanmax(np.abs(image)))
    mesh = ax.imshow(image, aspect='auto', cmap='RdBu_r', origin='lower',
                     extent=[times[0], times[-1], -.5, len(order) - .5],
                     vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([roi_names[i] for i in order], fontsize=8)
    ax.axvline(0, color='k', ls=':', lw=.8)
    ax.axvline(peak['peak_time'], color='k', ls='--', lw=.8)
    ax.set_xlim(-.2, .8)
    ax.set_xlabel('Time (s)')
    ax.set_title('The group z of the strongest labels')
    fig.colorbar(mesh, ax=ax, shrink=.8)
else:
    ax.text(.5, .5, 'no annotation', ha='center', va='center')
    ax.set_title('The time course of the labels')

# How much of the map survives the correction
ax = axes[1, 1]
# threshold = p_crit if np.isfinite(p_crit) else 0.
# n_sig = np.array([int((p_map[:, i] <= threshold).sum())
#                   for i in range(len(times))])
# ax.plot(times, n_sig, lw=1.4, color='tab:green',
#         label=f'vertices passing FDR {args.fdr}')
# ax.axvspan(args.search_window[0], args.search_window[1], color='tab:blue',
#            alpha=.06)
# ax.axvline(peak['peak_time'], color='tab:red', ls='--', lw=1.)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Vertices')
if np.isfinite(p_crit):
    ax.set_title(f'{n_reject} vertex time points pass p <= {p_crit:.3g}')
else:
    ax.set_title(f'nothing passes the FDR {args.fdr}')
if null.size:
    ax2 = ax.twinx()
    ax2.hist(null, bins=40, color='gray', alpha=.4)
    ax2.axvline(peak['roi_t_max'], color='tab:red', lw=1.2,
                label=f"label |t| {peak['roi_t_max']:.2f}, "
                      f"p {peak['roi_perm_p']:.3f}")
    ax2.set_ylabel('Sign flips')
    ax2.legend(fontsize=8, loc='upper left')
ax.legend(fontsize=8, loc='upper right')

fig.tight_layout()
fname = OUTPUT_DIR / f'group-{NAME}.{TAG}.png'
fig.savefig(fname, dpi=120)
plt.close(fig)
logger.info(f'Saved into {fname}')

# %% ---- 2026-09-21 ------------------------
# The surface plot, it needs a 3d backend
if args.brain:
    try:
        import matplotlib
        matplotlib.use('Agg')
        brain = stc_z.plot(hemi='both', initial_time=peak['peak_time'],
                           subjects_dir=SUBJECTS_DIR, time_viewer=False,
                           show=False)
        fname = (OUTPUT_DIR /
                 f'group-{NAME}.{TAG}-peak{peak["peak_time"]:.3f}s.png')
        brain.save_image(fname)
        logger.info(f'Saved the surface into {fname}')
    except Exception as e:
        logger.warning(f'The surface plot failed, {e}')


# %% ---- 2026-09-21 ------------------------
# Pending
# 1. The spatio temporal cluster permutation of mne.stats would be more
#    sensitive than the label wise maximum statistic, it needs the surface
#    adjacency of the source space and a lot of memory.
# 2. The contrast is a subtraction followed by a one sample test, a paired
#    test of the two conditions would keep the variance of both.


# %% ---- 2026-09-21 ------------------------
# Pending
