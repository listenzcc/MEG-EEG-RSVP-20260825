"""
File: group-source-plot.py
Author: Chuncheng Zhang
Date: 2026-09-23
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Screenshot the group mean source map of every modality with stc.plot.

    Stage 8 writes one stc per subject on the fsaverage template, so all the
    subjects share the same vertices and the maps can be averaged directly.
    This script averages them into one map per modality and renders that map on
    the fsaverage surface, which is the figure the article needs and the part
    group-source-map.py only describes in numbers.

    Two things are decided here and they are the ones that used to be done by
    hand:

    1. Which time point is shown. The default is the latency where the root mean
       square of the group map peaks inside --search-window, the same readout
       group-source-map.py reports, so the figure and the table always agree.
       --times overrides it with an explicit list.
    2. The colour scale. It is taken from a percentile of the map itself, so MEG
       and EEG are not forced onto the same numbers but both show their own
       whole dynamic range. MEG is in fT and EEG in arbitrary units of the
       inverse solution, a shared colour bar would make one of the two flat.

    The rendering is stc.plot, so it needs a 3d backend (pyvista or mayavi) on
    the machine that runs it. On a machine without one the script stops with
    the installation hint instead of failing halfway through, and --dry-run
    walks the whole data path without touching the renderer, which is how the
    averaging can be checked on a machine that only holds the stc files.

Usage:
    python python/group-source-plot.py
    python python/group-source-plot.py -e epochs-1-notch-epo.fif
    python python/group-source-plot.py --times 0.29 0.41 --views lat med dor
    python python/group-source-plot.py -t ssvep10-power --hemi both
    python python/group-source-plot.py --dry-run

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-09-23 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Screenshot the group mean source map with stc.plot')
parser.add_argument('-m', '--modes', nargs='+', default=['MEG', 'EEG'],
                    help='The modalities to render, MEG and EEG by default')
parser.add_argument('-e', '--epochs_fname',
                    default='epochs-1-notch-removal-artificial-epo.fif',
                    help='The epochs the source map was estimated from')
parser.add_argument('-t', '--tag', default='ave',
                    help='ave | ssvep10-evoked | ssvep10-power')
parser.add_argument('-s', '--subjects', nargs='*', default=None,
                    help='Subject names like S01, every subject found on the '
                         'disk by default')

# What is shown
parser.add_argument('--times', type=float, nargs='*', default=None,
                    help='The latencies to render, the peak of the group map '
                         'by default')
parser.add_argument('--search-window', type=float, nargs=2,
                    default=[0.05, 0.8],
                    help='Where the peak of the group map is looked for')
parser.add_argument('--hemi', default='split',
                    choices=['split', 'lh', 'rh', 'both'],
                    help="split renders one figure per hemisphere, both puts "
                         'the two hemispheres into one figure')
parser.add_argument('--views', nargs='+', default=['lat', 'med'],
                    help='The views of every figure, e.g. lat med dor ven')
parser.add_argument('--clim-percent', type=float, default=98.,
                    help='The colour scale is this percentile of |map|, 0 '
                         'lets stc.plot decide')
parser.add_argument('--smoothing-steps', type=int, default=7,
                    help='The smoothing of the map on the surface')
parser.add_argument('--panel-size', type=int, nargs=2, default=[380, 320],
                    help='The size of one view, the figure is the width times '
                         'the number of views')

# The renderer
parser.add_argument('--backend', default=None,
                    help="The 3d backend, e.g. pyvista or mayavi, the one mne "
                         'already uses by default')
parser.add_argument('--dry-run', action='store_true',
                    help='Stop before the renderer, it checks the averaging '
                         'and the peak without a 3d backend')
parser.add_argument('--no-contact-sheet', action='store_true',
                    help='Do not assemble the rendered images into one figure')
parser.add_argument('-o', '--output-dir', default='output/group-source-map',
                    help='Where the maps and the images are written')

args = parser.parse_args()
MODES = list(args.modes)
EPOCHS_FNAME = args.epochs_fname
TAG = args.tag

logger.info(f'Start with {args=}')

# %%
DATA_DIR = Path('output/source-estimation')
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# The name every output of this run shares
NAME = f'{EPOCHS_FNAME}.{TAG}'


# %% ---- 2026-09-23 ------------------------
# Function and class
def stc_stem(mode: str, subj: str):
    '''
    Find the stc of one subject.

    The stem is the name without the hemisphere, mne appends -lh.stc and
    -rh.stc to it when it reads the file. Stage 8 saves the stc under
    {epochs}.{tag}.stc, so both that stem and the one without the trailing
    .stc are accepted.

    Args:
        mode: MEG or EEG
        subj: The subject name

    Returns:
        stem: The Path to read, None when the subject is not there
    '''
    folder = DATA_DIR / f'{mode}-{subj}'
    for stem in (folder / f'{EPOCHS_FNAME}.{TAG}.stc',
                 folder / f'{EPOCHS_FNAME}.{TAG}'):
        if Path(str(stem) + '-lh.stc').exists():
            return stem
    return None


def read_group(mode: str):
    '''
    Read the stc of every subject of one modality and average it.

    The average is taken on the raw values, the subjects are already on the
    same template so there is nothing to morph and nothing to normalize. A
    subject whose stc has another shape is left out, it happens when the
    inverse solution of that subject was computed on another source space.

    Args:
        mode: MEG or EEG

    Returns:
        stc: The group mean as a source estimate
        subjects: The names of the subjects that went into it
        missing: The names of the subjects whose stc is not on the disk
    '''
    folders = sorted(DATA_DIR.glob(f'{mode}-*'))
    if args.subjects is None:
        wanted = [p.name.split('-')[-1] for p in folders]
    else:
        wanted = list(args.subjects)

    blocks, template, kept, missing = [], None, [], []
    for subj in wanted:
        stem = stc_stem(mode, subj)
        if stem is None:
            missing.append(subj)
            continue
        stc = mne.read_source_estimate(str(stem))
        if template is None:
            template = stc
        elif stc.data.shape != template.data.shape:
            logger.warning(f'{mode}-{subj} has another shape, left out')
            continue
        blocks.append(stc.data.astype(np.float64))
        kept.append(subj)

    if not blocks:
        raise SystemExit(
            f'No stc is found for {mode}, {DATA_DIR=}, {EPOCHS_FNAME=}, '
            f'{TAG=}. Run stage 8 first, it writes the stc of every subject.')

    stc = template.copy()
    stc.data = np.mean(blocks, axis=0)
    stc.subject = 'fsaverage'
    return stc, kept, missing


def peak_time(stc):
    '''
    The latency where the root mean square of the map peaks.

    The strongest single vertex is not used: with two hemispheres and a few
    hundred samples there are millions of vertex time points and the largest
    of them is a fluctuation rather than the response. The vertex averaged
    curve is the same readout group-source-map.py writes into its summary, so
    the figure and that table show the same thing.

    Args:
        stc: The group map

    Returns:
        t: The latency in seconds
    '''
    times = stc.times
    inside = np.where((times >= args.search_window[0])
                      & (times <= args.search_window[1]))[0]
    if not inside.any():
        raise SystemExit(f'{args.search_window} has no sample')
    curve = np.sqrt((stc.data ** 2).mean(axis=0))
    return float(times[inside[int(np.argmax(curve[inside]))]])


def color_scale(stc, times):
    '''
    The colour scale of the figure.

    It is a percentile of the map itself instead of a shared number, MEG is in
    fT and EEG in the units of the inverse solution and one scale for both
    would flatten one of them. A map that is positive everywhere, the SSVEP
    power is one, gets a positive only scale.

    Args:
        stc: The group map
        times: The latencies that are rendered

    Returns:
        clim: The dict stc.plot understands, 'auto' without --clim-percent
    '''
    if args.clim_percent <= 0:
        return 'auto'

    window = np.zeros(len(stc.times), bool)
    for t in times:
        window |= np.isclose(stc.times, t, atol=1. / stc.sfreq)
    if not window.any():
        window[:] = True

    block = stc.data[:, window]
    vmax = float(np.nanpercentile(np.abs(block), args.clim_percent))
    if vmax <= 0:
        logger.warning('The map is flat, the colour scale is left to mne')
        return 'auto'

    if float(np.nanmin(block)) >= 0.:
        return dict(kind='value', pos_lims=[0., vmax / 2., vmax])
    return dict(kind='value', lims=[-vmax, 0., vmax])


def nearest_index(stc, t: float):
    '''
    The sample of the latency that is asked for.

    stc.plot takes an initial_time and snaps it to the closest sample, the
    figure title would otherwise show a latency that is not in the data.

    Args:
        stc: The group map, it carries the time axis
        t: The latency in seconds

    Returns:
        i: The index of the sample
        snapped: The latency of that sample
    '''
    i = int(np.argmin(np.abs(stc.times - t)))
    return i, float(stc.times[i])


def check_backend():
    '''
    Make sure a 3d backend is there before anything is rendered.

    Returns:
        backend: The name of the backend
    '''
    if args.backend is not None:
        mne.viz.set_3d_backend(args.backend, verbose='ERROR')
    try:
        backend = mne.viz.get_3d_backend()
    except Exception as e:
        raise SystemExit(
            f'No 3d backend is available, {e}. Install one of them:\n'
            "  pip install pyvista pyvistaqt   (the mne default)\n"
            "  conda install -c conda-forge mayavi   (the old PySurfer one)\n"
            'On a headless machine run it under xvfb-run -a.')
    logger.info(f'The 3d backend is {backend}')
    return backend


def render(stc, mode: str, times, subjects_dir):
    '''
    Render the map at every latency and save the figures.

    Every figure is one hemisphere seen from --views, which is the layout an
    article panel uses. With --hemi both the two hemispheres share one figure
    instead.

    Args:
        stc: The group map
        mode: MEG or EEG, it goes into the file name
        times: The latencies to render
        subjects_dir: The folder that holds fsaverage

    Returns:
        images: A list of (mode, hemisphere, latency, path)
    '''
    hemis = {'split': ['lh', 'rh'], 'lh': ['lh'], 'rh': ['rh'],
             'both': ['both']}[args.hemi]
    clim = color_scale(stc, times)
    logger.info(f'The colour scale of {mode} is {clim}')

    images = []
    for t in times:
        i, snapped = nearest_index(stc, t)
        for hemi in hemis:
            brain = stc.plot(
                subject='fsaverage', subjects_dir=subjects_dir, hemi=hemi,
                views=list(args.views), initial_time=snapped,
                time_viewer=False, show=False, clim=clim, colorbar=True,
                size=(args.panel_size[0] * len(args.views),
                      args.panel_size[1]),
                background='white', foreground='black',
                smoothing_steps=args.smoothing_steps, cortex='classic',
                title=f'{mode} {NAME} {snapped * 1000:.0f} ms '
                      f'({len(subjects)} subjects)',
                verbose='ERROR')
            fname = (OUTPUT_DIR /
                     f'group-{mode}-{NAME}-mean-{hemi}-'
                     f'{snapped:.3f}s-{"".join(args.views)}.png')
            brain.save_image(str(fname))
            images.append((mode, hemi, snapped, fname))
            logger.info(f'Saved into {fname}')
            if hasattr(brain, 'close'):
                brain.close()
    return images


def contact_sheet(images):
    '''
    Put the rendered images into one figure.

    The rows are the latencies and the columns are the modality and the
    hemisphere, so MEG and EEG can be compared at a glance. The images are
    read back from the disk, the renderer is not involved.

    Args:
        images: A list of (mode, hemisphere, latency, path)
    '''
    if args.no_contact_sheet or not images:
        return

    lats = sorted({round(t, 4) for _, _, t, _ in images})
    # The columns follow --modes rather than the alphabet, so MEG stays left of
    # EEG and the panel order of the figure matches the order of the report
    order = {m: i for i, m in enumerate(MODES)}
    cols = sorted({(m, h) for m, h, _, _ in images},
                  key=lambda mh: (order.get(mh[0], 99), mh[1]))
    fig, axes = plt.subplots(len(lats), len(cols),
                             figsize=(3.2 * len(cols), 3.4 * len(lats)),
                             squeeze=False)
    for r, t in enumerate(lats):
        for c, (mode, hemi) in enumerate(cols):
            ax = axes[r][c]
            hit = [p for m, h, lt, p in images
                   if m == mode and h == hemi and round(lt, 4) == t]
            if not hit:
                ax.axis('off')
                continue
            ax.imshow(plt.imread(str(hit[0])))
            ax.axis('off')
            ax.set_title(f'{mode} {hemi} {t * 1000:.0f} ms', fontsize=10)

    fig.suptitle(f'Group mean {NAME}, {len(images)} panels', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = OUTPUT_DIR / f'group-{NAME}-mean-contact-sheet.png'
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    logger.info(f'Saved into {fname}')


# %% ---- 2026-09-23 ------------------------
# Play ground
SUBJECTS_DIR = None
try:
    fs_dir = mne.datasets.fetch_fsaverage(verbose='ERROR')
    SUBJECTS_DIR = os.path.dirname(fs_dir)
    logger.info(f'The template is in {SUBJECTS_DIR}')
except Exception as e:
    logger.warning(f'The fsaverage template is not available, {e}')

if not args.dry_run:
    check_backend()

rows, all_images = [], []
for mode in MODES:
    stc, subjects, missing = read_group(mode)
    logger.info(f'{mode}: {len(subjects)} subjects, {stc.data.shape} vertices '
                f'and {stc.data.shape[1]} samples')
    if missing:
        logger.warning(f'{mode}: no stc for {missing}')

    fname = OUTPUT_DIR / f'group-{mode}-{NAME}-mean.stc'
    stc.save(fname, overwrite=True)
    logger.info(f'Saved into {fname}')

    if args.times is None:
        times = [peak_time(stc)]
    else:
        times = [nearest_index(stc, t)[1] for t in args.times]
    logger.info(f'{mode}: the peak is at {times[0]:.3f} s, rendering {times}')

    rows.append(dict(mode=mode, epochs_fname=EPOCHS_FNAME, tag=TAG,
                     n_subjects=len(subjects), subjects=' '.join(subjects),
                     missing=' '.join(missing),
                     peak_time=round(times[0], 4),
                     unit='fT' if mode == 'MEG' else 'inverse solution',
                     stc=str(fname)))

    if not args.dry_run:
        all_images += render(stc, mode, times, SUBJECTS_DIR)

df = pd.DataFrame(rows)
display(df)
fname = OUTPUT_DIR / f'group-{NAME}-mean-summary.csv'
df.to_csv(fname, index=False)
logger.info(f'Saved into {fname}')

contact_sheet(all_images)

if args.dry_run:
    logger.info('The dry run stops here, nothing was rendered')


# %% ---- 2026-09-23 ------------------------
# Pending
# 1. The peak is the vertex averaged one, a cluster of the strongest vertices
#    would describe a focal component better and would give a window rather
#    than one latency.
# 2. The colour scale is per modality, a common z scale would make MEG and EEG
#    comparable at the cost of showing the weaker one as noise.


# %% ---- 2026-09-23 ------------------------
# Pending
