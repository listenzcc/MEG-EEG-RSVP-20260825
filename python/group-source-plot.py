"""
File: group-source-plot.py
Author: Chuncheng Zhang
Date: 2026-09-23
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Screenshot the group mean source map of every modality with stc.plot.

    Stage 8 writes one stc per subject on the fsaverage template, so all the
    subjects share the same vertices and the maps can be averaged directly.
    This script averages them into one map per condition and modality and
    renders that map on the fsaverage surface, which is the figure the article
    needs and the part group-source-map.py only describes in numbers.

    Three things are decided here and they are the ones that used to be done by
    hand:

    1. Which time point is shown. The default is the latency where the root mean
       square of the group map peaks inside --search-window, the same readout
       group-source-map.py reports, so the figure and the table always agree.
       --times overrides it with an explicit list, which is what a comparison
       of two conditions wants: both have to be rendered at the same latency,
       otherwise the difference between the two panels is timing and not
       amplitude.
    2. The colour scale. It is taken from a percentile of the map itself, so MEG
       and EEG are not forced onto the same numbers but both show their own
       whole dynamic range. MEG is in fT and EEG in arbitrary units of the
       inverse solution, a shared colour bar would make one of the two flat.
       A difference map is always scaled around zero, a one sided scale would
       hide the sign of the contrast.
    3. How many conditions go into one contact sheet. -t takes a list, so the
       quick and the slow reaction time group can be rendered in one run and
       land in one figure.

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
    python python/group-source-plot.py -t ave-quick --contrast-tag ave-slow
    python python/group-source-plot.py -t ave-quick ave-slow --times 0.30
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
parser.add_argument('-t', '--tags', nargs='+', default=['ave'],
                    help='The tags to render, ave | ave-quick ave-slow | '
                         'ssvep10-power')
parser.add_argument('--contrast-tag', default=None,
                    help='Render the difference of two tags of the same epochs '
                         'instead of the tags themselves, e.g. -t ave-quick '
                         '--contrast-tag ave-slow draws quick minus slow')
parser.add_argument('-s', '--subjects', nargs='*', default=None,
                    help='Subject names like S01, every subject found on the '
                         'disk by default')

# What is shown
parser.add_argument('--times', type=float, nargs='*', default=None,
                    help='The latencies to render, the peak of the group map '
                         'by default. Give the list explicitly when two '
                         'conditions are compared, the same latency for both '
                         'is what makes the two panels comparable')
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
TAGS = list(args.tags)
CONTRAST_TAG = args.contrast_tag

if CONTRAST_TAG is not None and CONTRAST_TAG in TAGS:
    raise SystemExit(f'--contrast_tag {CONTRAST_TAG} is one of the tags to '
                     'render, the difference against itself would be empty')

logger.info(f'Start with {args=}')

# %%
DATA_DIR = Path('output/source-estimation')
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# The conditions to render, one entry per tag. Without --contrast_tag the
# condition is the tag itself, with it the condition is the difference of that
# tag and the contrast tag, both of which live in the same epochs file of the
# subject and are therefore a within subject contrast.
CONDITIONS = []
for tag in TAGS:
    if CONTRAST_TAG is None:
        CONDITIONS.append(dict(tag=tag, contrast=None, label=tag, name=tag))
    else:
        CONDITIONS.append(dict(tag=tag, contrast=CONTRAST_TAG,
                               label=f'{tag}-minus-{CONTRAST_TAG}',
                               name=f'{tag} - {CONTRAST_TAG}'))

# The name every output of this run shares. The contrast enters the name so a
# run against another condition never overwrites the plain one, and the name of
# a single tag without a contrast stays exactly what it was before this script
# learned about more than one condition.
SHEET_NAME = f'{EPOCHS_FNAME}.{"-and-".join(c["label"] for c in CONDITIONS)}'

# 'mean' is the average of the subjects, 'diff' is that average of a
# subtraction. The word goes into the file name because a difference map read as
# an average map would be a mistake.
QUANTITY = 'mean' if CONTRAST_TAG is None else 'diff'


# %% ---- 2026-09-23 ------------------------
# Function and class
def stc_stem(mode: str, subj: str, tag: str):
    '''
    Find the stc of one subject.

    The stem is the name without the hemisphere, mne appends -lh.stc and
    -rh.stc to it when it reads the file. Stage 8 saves the stc under
    {epochs}.{tag}.stc, so both that stem and the one without the trailing
    .stc are accepted.

    Args:
        mode: MEG or EEG
        subj: The subject name
        tag: ave | ave-quick | ave-slow | ssvep10-evoked | ssvep10-power

    Returns:
        stem: The Path to read, None when the subject is not there
    '''
    folder = DATA_DIR / f'{mode}-{subj}'
    for stem in (folder / f'{EPOCHS_FNAME}.{tag}.stc',
                 folder / f'{EPOCHS_FNAME}.{tag}'):
        if Path(str(stem) + '-lh.stc').exists():
            return stem
    return None


def read_group(mode: str, cond: dict):
    '''
    Read the stc of every subject of one condition and average it.

    The average is taken on the raw values, the subjects are already on the
    same template so there is nothing to morph and nothing to normalize. With a
    contrast the subtraction happens per subject before the average, so the
    map is the group average of the within subject difference and not the
    difference of two group averages. The two are the same number, but only the
    first one is the quantity the statistics of a paired design are computed
    on, and keeping the two routes identical avoids the question.

    A subject whose stc is missing for either of the two conditions is left out
    of both, otherwise the average of the difference would be taken over
    another set of subjects than the average of the tag itself.

    Args:
        mode: MEG or EEG
        cond: The condition, it carries the tag and the contrast tag

    Returns:
        stc: The group map as a source estimate
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
        stem = stc_stem(mode, subj, cond['tag'])
        if stem is None:
            missing.append(subj)
            continue
        stc = mne.read_source_estimate(str(stem))
        data = stc.data.astype(np.float64)

        if cond['contrast'] is not None:
            stem_b = stc_stem(mode, subj, cond['contrast'])
            if stem_b is None:
                logger.warning(f'{mode}-{subj} has no {EPOCHS_FNAME}.'
                               f'{cond["contrast"]}, the subject is left out')
                missing.append(subj)
                continue
            stc_b = mne.read_source_estimate(str(stem_b))
            if stc_b.shape != stc.shape:
                logger.warning(f'{mode}-{subj}: {stc_b.shape} does not match '
                               f'{stc.shape}, the subject is left out')
                missing.append(subj)
                continue
            data = data - stc_b.data.astype(np.float64)

        if template is None:
            template = stc
        elif stc.data.shape != template.data.shape:
            logger.warning(f'{mode}-{subj} has another shape, left out')
            continue

        blocks.append(data)
        kept.append(subj)

    if not blocks:
        raise SystemExit(
            f'No stc is found for {mode} {cond["label"]}, {DATA_DIR=}, '
            f'{EPOCHS_FNAME=}, tag={cond["tag"]}, '
            f'contrast={cond["contrast"]}. Run stage 8 first, it writes the '
            'stc of every subject.')

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
    power is one, gets a positive only scale. A difference keeps the two sided
    scale whatever its own numbers look like, the sign is the message there.

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

    if QUANTITY == 'mean' and float(np.nanmin(block)) >= 0.:
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


def render(stc, cond: dict, mode: str, times, subjects_dir, n_subjects: int):
    '''
    Render the map at every latency and save the figures.

    Every figure is one hemisphere seen from --views, which is the layout an
    article panel uses. With --hemi both the two hemispheres share one figure
    instead.

    Args:
        stc: The group map
        cond: The condition, it names the figure
        mode: MEG or EEG, it goes into the file name
        times: The latencies to render
        subjects_dir: The folder that holds fsaverage
        n_subjects: How many subjects the map averages, it goes into the title

    Returns:
        images: A list of (condition, mode, hemisphere, latency, path)
    '''
    hemis = {'split': ['lh', 'rh'], 'lh': ['lh'], 'rh': ['rh'],
             'both': ['both']}[args.hemi]
    clim = color_scale(stc, times)
    logger.info(f'The colour scale of {mode} {cond["name"]} is {clim}')

    images = []
    for t in times:
        i, snapped = nearest_index(stc, t)
        for hemi in hemis:
            brain = stc.plot(
                subject='fsaverage', subjects_dir=subjects_dir, hemi=hemi,
                views=list(args.views), initial_time=snapped,
                time_viewer=False,
                # show=False,
                clim=clim, colorbar=True,
                size=(args.panel_size[0] * len(args.views),
                      args.panel_size[1]),
                background='white', foreground='black',
                smoothing_steps=args.smoothing_steps, cortex='classic',
                title=f'{mode} {cond["name"]} {snapped * 1000:.0f} ms '
                      f'({n_subjects} subjects)',
                verbose='ERROR')
            fname = (OUTPUT_DIR /
                     f'group-{mode}-{SHEET_NAME}-{QUANTITY}-{hemi}-'
                     f'{snapped:.3f}s-{"".join(args.views)}.png')
            brain.save_image(str(fname))
            images.append((cond['name'], mode, hemi, snapped, fname))
            logger.info(f'Saved into {fname}')
            if hasattr(brain, 'close'):
                brain.close()
    return images


def contact_sheet(images):
    '''
    Put the rendered images into one figure.

    The rows are the condition and the latency and the columns are the modality
    and the hemisphere, so MEG and EEG can be compared at a glance and the two
    conditions of one contrast sit above each other. The images are read back
    from the disk, the renderer is not involved.

    Args:
        images: A list of (condition, mode, hemisphere, latency, path)
    '''
    if args.no_contact_sheet or not images:
        return

    # The rows keep the order they were rendered in, which is the order of -t
    order = {m: i for i, m in enumerate(MODES)}
    cols = sorted({(m, h) for _, m, h, _, _ in images},
                  key=lambda mh: (order.get(mh[0], 99), mh[1]))
    rows = []
    for name, mode, hemi, t, _ in images:
        key = (name, round(t, 4))
        if key not in rows:
            rows.append(key)

    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(3.2 * len(cols), 3.4 * len(rows)),
                             squeeze=False)
    for r, (name, t) in enumerate(rows):
        for c, (mode, hemi) in enumerate(cols):
            ax = axes[r][c]
            hit = [p for n, m, h, lt, p in images
                   if n == name and m == mode and h == hemi
                   and round(lt, 4) == t]
            if not hit:
                ax.axis('off')
                continue
            ax.imshow(plt.imread(str(hit[0])))
            ax.axis('off')
            ax.set_title(f'{mode} {hemi} {t * 1000:.0f} ms', fontsize=10)

    fig.suptitle(f'Group {QUANTITY} of {SHEET_NAME}, {len(images)} panels',
                 fontsize=12)
    # The condition of every row goes into the left margin. It cannot be the
    # ylabel of the first column: axis('off') turns the axis label off with
    # everything else, and a plain ax.text survives it.
    fig.tight_layout(rect=[0.05, 0, 1, 0.95])
    for r, (name, t) in enumerate(rows):
        box = axes[r][0].get_position()
        fig.text(0.015, box.y0 + box.height / 2., name, rotation=90,
                 ha='center', va='center', fontsize=9)
    fname = OUTPUT_DIR / f'group-{SHEET_NAME}-{QUANTITY}-contact-sheet.png'
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

# The maps are read once and rendered per condition, the averaging does not
# depend on anything the renderer does
rows, all_images = [], []
for cond in CONDITIONS:
    for mode in MODES:
        stc, subjects, missing = read_group(mode, cond)
        logger.info(f'{mode} {cond["name"]}: {len(subjects)} subjects, '
                    f'{stc.data.shape[0]} vertices and '
                    f'{stc.data.shape[1]} samples')
        if missing:
            logger.warning(f'{mode} {cond["name"]}: no stc for {missing}')

        fname = OUTPUT_DIR / f'group-{mode}-{SHEET_NAME}-{QUANTITY}.stc'
        stc.save(fname, overwrite=True)
        logger.info(f'Saved into {fname}')

        # The explicit --times is used as it is, so that two conditions of one
        # run land on the same latency and the panels can be read against each
        # other. Without it every condition shows its own peak.
        if args.times is None:
            times = [peak_time(stc)]
        else:
            times = [nearest_index(stc, t)[1] for t in args.times]
        logger.info(f'{mode} {cond["name"]}: the peak is at '
                    f'{times[0]:.3f} s, rendering {times}')

        rows.append(dict(mode=mode, epochs_fname=EPOCHS_FNAME,
                         tag=cond['tag'], contrast_tag=cond['contrast'],
                         quantity=QUANTITY, n_subjects=len(subjects),
                         subjects=' '.join(subjects),
                         missing=' '.join(missing),
                         peak_time=round(times[0], 4),
                         rendered=' '.join(f'{t:.3f}' for t in times),
                         unit='fT' if mode == 'MEG' else 'inverse solution',
                         stc=str(fname)))

        if not args.dry_run:
            all_images += render(stc, cond, mode, times, SUBJECTS_DIR,
                                 len(subjects))

df = pd.DataFrame(rows)
display(df)
fname = OUTPUT_DIR / f'group-{SHEET_NAME}-{QUANTITY}-summary.csv'
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
