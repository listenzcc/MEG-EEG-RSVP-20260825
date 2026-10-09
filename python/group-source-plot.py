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
       land in one figure. -e takes a list as well and the conditions are the
       cross product of the two, which is how the same condition estimated from
       two epochs files, e.g. before and after the keypress projection, gets
       into one figure where the panels can be read against each other.
    4. Whether the colour scale is per map or shared. Every panel of a
       comparison has to use the same colour bar, otherwise the difference
       between two panels can be the scale and not the map. --clim-shared
       takes the percentile over all the conditions of one modality together,
       without it every map gets its own, which is what a single map wants.

    Every panel is written to a file whose name carries the condition it
    belongs to. Without that two conditions rendered at the same latency, a
    comparison at one fixed --times for example, would write the same file and
    only the last one would survive, and the contact sheet would then show the
    same image in every row.

    The two products of a whole run, the contact sheet and the summary csv,
    carry the modalities in their name as well. MEG and EEG are usually two
    commands with the same -e and -t, and a name without the modality makes the
    second one overwrite the first: the summary is written with to_csv and not
    merged, so the first modality would be gone rather than kept.

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
    python python/group-source-plot.py -t ave-quick ave-slow \
        -e epochs-1-notch-epo.fif epochs-1-notch-removal-artificial-epo.fif \
        --times 0.30 --clim-shared
    python python/group-source-plot.py -t ave-quick \
        -e epochs-1-notch-epo.fif -c epochs-1-notch-removal-artificial-epo.fif
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
parser.add_argument('-e', '--epochs_fname', nargs='+',
                    default=['epochs-1-notch-removal-artificial-epo.fif'],
                    help='The epochs the source map was estimated from. Give '
                         'several to render the same tags of several epochs '
                         'files, e.g. the epochs before and after the keypress '
                         'projection')
parser.add_argument('-t', '--tags', nargs='+', default=['ave'],
                    help='The tags to render, ave | ave-quick ave-slow | '
                         'ssvep10-power')
parser.add_argument('--contrast-tag', default=None,
                    help='Render the difference of two tags of the same epochs '
                         'instead of the tags themselves, e.g. -t ave-quick '
                         '--contrast-tag ave-slow draws quick minus slow')
parser.add_argument('-c', '--contrast_epochs', default=None,
                    help='Subtract this tag of another epochs file of the same '
                         'subject instead of another tag of the same file, e.g. '
                         'the epochs before the keypress projection. Only one '
                         'of the two describes the contrast')
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
parser.add_argument('--clim-shared', action='store_true',
                    help='Take that percentile over every condition of one '
                         'modality together instead of per map, so the panels '
                         'of a comparison share one colour bar. Without it a '
                         'map that is weaker than its neighbour looks just as '
                         'strong as long as it is read on its own scale')
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
# The run wide products, the contact sheet and the summary, are read as one
# file by whoever opens them, and MEG and EEG are usually two commands with the
# same -e and -t. Without the modality in the name the second command overwrites
# the first, and the summary of the first modality is gone rather than merged.
MODES_TAG = '-'.join(MODES)
EPOCHS_FNAMES = list(args.epochs_fname)
TAGS = list(args.tags)
CONTRAST_TAG = args.contrast_tag
CONTRAST_EPOCHS = args.contrast_epochs

if CONTRAST_TAG is not None and CONTRAST_TAG in TAGS:
    raise SystemExit(f'--contrast_tag {CONTRAST_TAG} is one of the tags to '
                     'render, the difference against itself would be empty')
if CONTRAST_TAG is not None and CONTRAST_EPOCHS is not None:
    raise SystemExit('--contrast-tag takes another tag of the same epochs and '
                     '-c/--contrast_epochs takes another epochs file of the same '
                     'tag, only one of the two describes the contrast')
if CONTRAST_EPOCHS is not None and CONTRAST_EPOCHS in EPOCHS_FNAMES:
    raise SystemExit(f'-c/--contrast_epochs {CONTRAST_EPOCHS} is one of the '
                     'epochs being rendered, the difference would be empty')

logger.info(f'Start with {args=}')

# %%
DATA_DIR = Path('output/source-estimation')
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def short_epochs(fname: str) -> str:
    '''
    A readable short form of an epochs file name.

    It is only used in the titles and in the name of a run that mixes several
    epochs files, never in the name of a file that is written per condition, so
    the long name stays the unambiguous one.

    Args:
        fname: The epochs file name

    Returns:
        The short form
    '''
    s = str(fname)
    for prefix in ('epochs-1-', 'epochs-2-', 'epochs-3-'):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    for suffix in ('-epo.fif', '-ave.fif', '.fif'):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            break
    return s or str(fname)


# The conditions to render, the cross product of the epochs files and the tags.
# One epochs file and one tag, the default, gives one condition and every output
# keeps the name it had before this script learned about more than one condition.
CONDITIONS = []
for epochs_fname in EPOCHS_FNAMES:
    for tag in TAGS:
        if CONTRAST_EPOCHS is not None:
            # The contrast is the same tag of another epochs file
            other = short_epochs(CONTRAST_EPOCHS)
            label, name = f'{tag}-minus-{other}', f'{tag} - {other}'
        elif CONTRAST_TAG is not None:
            label = f'{tag}-minus-{CONTRAST_TAG}'
            name = f'{tag} - {CONTRAST_TAG}'
        else:
            label, name = tag, tag
        CONDITIONS.append(dict(
            epochs=epochs_fname, tag=tag, contrast_tag=CONTRAST_TAG,
            contrast_epochs=CONTRAST_EPOCHS,
            label=label, name=name,
            # A condition of a run that mixes epochs files is not named by its
            # tag alone, the two files carry the same tag by design
            display=(name if len(EPOCHS_FNAMES) == 1
                     else f'{short_epochs(epochs_fname)} {name}'),
            # The name every panel file of this condition shares. It carries the
            # epochs fname, the two projection states differ in nothing else.
            stem=f'{epochs_fname}.{label}'))

# The name every output of this run shares. The contrast enters the name so a
# run against another condition never overwrites the plain one, and the name of
# a single tag without a contrast stays exactly what it was before this script
# learned about more than one condition.
if len(EPOCHS_FNAMES) == 1:
    SHEET_NAME = (f'{EPOCHS_FNAMES[0]}.'
                  f'{"-and-".join(c["label"] for c in CONDITIONS)}')
else:
    # The epochs file names are long enough that joining all of them builds a
    # path Windows refuses, and the panel file name and the row label carry the
    # detail anyway
    SHEET_NAME = (f'{len(EPOCHS_FNAMES)}epochs.'
                  f'{"-and-".join(sorted({c["label"] for c in CONDITIONS}))}')

# 'mean' is the average of the subjects, 'diff' is that average of a
# subtraction. The word goes into the file name because a difference map read as
# an average map would be a mistake.
QUANTITY = 'mean' if CONTRAST_TAG is None and CONTRAST_EPOCHS is None else 'diff'


# %% ---- 2026-09-23 ------------------------
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
        tag: ave | ave-quick | ave-slow | ssvep10-evoked | ssvep10-power

    Returns:
        stem: The Path to read, None when the subject is not there
    '''
    folder = DATA_DIR / f'{mode}-{subj}'
    for stem in (folder / f'{epochs_fname}.{tag}.stc',
                 folder / f'{epochs_fname}.{tag}'):
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
        stem = stc_stem(mode, subj, cond['epochs'], cond['tag'])
        if stem is None:
            missing.append(subj)
            continue
        stc = mne.read_source_estimate(str(stem))
        data = stc.data.astype(np.float64)

        if cond['contrast_tag'] is not None or cond['contrast_epochs'] is not None:
            # The contrast is either another tag of the same epochs file, the
            # quick and the slow reaction time group for example, or the same
            # tag of another epochs file, the epochs with and without the
            # keypress projection
            other = cond['contrast_epochs'] or cond['epochs']
            stem_b = stc_stem(mode, subj, other, cond['contrast_tag'] or cond['tag'])
            if stem_b is None:
                logger.warning(f'{mode}-{subj} has no {other}.'
                               f'{cond["contrast_tag"] or cond["tag"]}, the '
                               'subject is left out')
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
            f'No stc is found for {mode} {cond["display"]}, {DATA_DIR=}, '
            f'epochs={cond["epochs"]}, tag={cond["tag"]}, '
            f'contrast={cond["contrast_tag"] or cond["contrast_epochs"]}. '
            'Run stage 8 first, it writes the stc of every subject.')

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


def sample_mask(stc, times):
    '''
    The samples of the latencies that are rendered.

    Args:
        stc: The group map, it carries the time axis
        times: The latencies in seconds

    Returns:
        mask: The boolean mask over stc.times
    '''
    mask = np.zeros(len(stc.times), bool)
    for t in times:
        mask |= np.isclose(stc.times, t, atol=1. / stc.sfreq)
    if not mask.any():
        mask[:] = True
    return mask


def map_stats(stcs, times_list):
    '''
    The percentile of |map| and the sign of the maps, taken over several
    conditions at once.

    This is what --clim-shared needs: one number for every panel of a
    comparison, computed from the samples the panels actually display. The sign
    goes with it, a scale that is positive only for one panel and two sided for
    its neighbour would draw the same value in two colours.

    Args:
        stcs: The group maps of one modality
        times_list: The latencies rendered of every one of them

    Returns:
        vmax: The percentile of |map|, 0 when the maps are flat
        positive_only: True when every map stays positive in the window
    '''
    vals, mins = [], []
    for stc, times in zip(stcs, times_list):
        block = stc.data[:, sample_mask(stc, times)]
        vals.append(np.abs(block).ravel())
        mins.append(float(np.nanmin(block)))
    if not vals:
        return 0., True
    vmax = float(np.nanpercentile(np.concatenate(vals), args.clim_percent))
    return vmax, all(v >= 0. for v in mins)


def color_scale(stc, times, vmax=None, positive_only=None):
    '''
    The colour scale of the figure.

    It is a percentile of the map instead of a shared number, MEG is in fT and
    EEG in the units of the inverse solution and one scale for both would
    flatten one of them. A map that is positive everywhere, the SSVEP power is
    one, gets a positive only scale. A difference keeps the two sided scale
    whatever its own numbers look like, the sign is the message there.

    Args:
        stc: The group map
        times: The latencies that are rendered
        vmax: The upper limit, computed from this map alone when None. Giving
            it is what --clim-shared does, and it is then shared by every
            condition of the same modality
        positive_only: Whether the scale is one sided, decided the same way as
            vmax when None

    Returns:
        clim: The dict stc.plot understands, 'auto' without --clim-percent
    '''
    if args.clim_percent <= 0:
        return 'auto'

    block = stc.data[:, sample_mask(stc, times)]
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(block), args.clim_percent))
    if positive_only is None:
        positive_only = (QUANTITY == 'mean'
                         and float(np.nanmin(block)) >= 0.)
    if vmax <= 0:
        logger.warning('The map is flat, the colour scale is left to mne')
        return 'auto'

    if positive_only:
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


def render(stc, cond: dict, mode: str, times, subjects_dir, n_subjects: int,
           clim):
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
        clim: The colour scale of this panel

    Returns:
        images: A list of dicts describing the panels that were written
    '''
    hemis = {'split': ['lh', 'rh'], 'lh': ['lh'], 'rh': ['rh'],
             'both': ['both']}[args.hemi]
    logger.info(f'The colour scale of {mode} {cond["display"]} is {clim}')

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
                title=f'{mode} {cond["display"]} {snapped * 1000:.0f} ms '
                      f'({n_subjects} subjects)',
                verbose='ERROR')
            # The condition is in the name. Without it two conditions rendered
            # at the same latency would write the same file and only the last
            # one would survive, and the contact sheet would show that one
            # image in every row.
            fname = (OUTPUT_DIR /
                     f'group-{mode}-{cond["stem"]}-{QUANTITY}-{hemi}-'
                     f'{snapped:.3f}s-{"".join(args.views)}.png')
            brain.save_image(str(fname))
            images.append(dict(stem=cond['stem'], display=cond['display'],
                               mode=mode, hemi=hemi, time=snapped, path=fname))
            logger.info(f'Saved into {fname}')
            if hasattr(brain, 'close'):
                brain.close()
    return images


def contact_sheet(images):
    '''
    Put the rendered images into one figure.

    The rows are the condition and the latency and the columns are the modality
    and the hemisphere, so MEG and EEG can be compared at a glance and the
    conditions of one comparison sit above each other. The images are read back
    from the disk, the renderer is not involved.

    Args:
        images: The panels render returned
    '''
    if args.no_contact_sheet or not images:
        return

    # The rows keep the order they were rendered in, which is the order of -e
    # and -t
    order = {m: i for i, m in enumerate(MODES)}
    cols = sorted({(im['mode'], im['hemi']) for im in images},
                  key=lambda mh: (order.get(mh[0], 99), mh[1]))
    rows = []
    for im in images:
        key = (im['stem'], round(im['time'], 4))
        if key not in rows:
            rows.append(key)

    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(3.2 * len(cols), 3.4 * len(rows)),
                             squeeze=False)
    for r, (stem, t) in enumerate(rows):
        for c, (mode, hemi) in enumerate(cols):
            ax = axes[r][c]
            hit = [im for im in images
                   if im['stem'] == stem and im['mode'] == mode
                   and im['hemi'] == hemi and round(im['time'], 4) == t]
            if not hit:
                ax.axis('off')
                continue
            ax.imshow(plt.imread(str(hit[0]['path'])))
            ax.axis('off')
            ax.set_title(f'{mode} {hemi} {t * 1000:.0f} ms', fontsize=10)

    fig.suptitle(f'Group {QUANTITY} of {SHEET_NAME}, {len(images)} panels',
                 fontsize=12)
    # The condition of every row goes into the left margin. It cannot be the
    # ylabel of the first column: axis('off') turns the axis label off with
    # everything else, and a plain ax.text survives it.
    fig.tight_layout(rect=[0.05, 0, 1, 0.95])
    for r, (stem, t) in enumerate(rows):
        box = axes[r][0].get_position()
        label = next(im['display'] for im in images if im['stem'] == stem)
        fig.text(0.015, box.y0 + box.height / 2., label, rotation=90,
                 ha='center', va='center', fontsize=9)
    fname = OUTPUT_DIR / \
        f'group-{MODES_TAG}-{SHEET_NAME}-{QUANTITY}-contact-sheet.png'
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
# depend on anything the renderer does. Every condition is read before anything
# is written, because a shared colour scale has to know every sample the panels
# will display.
groups = []
for cond in CONDITIONS:
    for mode in MODES:
        stc, subjects, missing = read_group(mode, cond)
        logger.info(f'{mode} {cond["display"]}: {len(subjects)} subjects, '
                    f'{stc.data.shape[0]} vertices and '
                    f'{stc.data.shape[1]} samples')
        if missing:
            logger.warning(f'{mode} {cond["display"]}: no stc for {missing}')

        # The explicit --times is used as it is, so that several conditions of
        # one run land on the same latency and the panels can be read against
        # each other. Without it every condition shows its own peak.
        own_peak = peak_time(stc)
        times = ([own_peak] if args.times is None
                 else [nearest_index(stc, t)[1] for t in args.times])
        logger.info(f'{mode} {cond["display"]}: the map peaks at '
                    f'{own_peak:.3f} s, rendering {times}')

        groups.append(dict(cond=cond, mode=mode, stc=stc, subjects=subjects,
                           missing=missing, times=times, own_peak=own_peak))

# The colour scale. It is taken per modality, MEG is in fT and EEG in the units
# of the inverse solution and one scale for both would flatten one of them.
# --clim-shared takes the percentile over every condition of one modality
# together, so the panels of a comparison can be read against each other, and
# that is the mode a comparison has to be rendered in.
clim_of = {}
for mode in MODES:
    members = [g for g in groups if g['mode'] == mode]
    if not members:
        continue
    shared = (map_stats([g['stc'] for g in members],
                        [g['times'] for g in members])
              if args.clim_shared else (None, None))
    if args.clim_shared:
        logger.info(f'{mode} shares one colour scale, '
                    f'the percentile of |map| is {shared[0]:.4g}')
    for g in members:
        clim_of[(mode, g['cond']['stem'])] = color_scale(
            g['stc'], g['times'], *shared)

rows, all_images = [], []
for g in groups:
    cond, mode, stc = g['cond'], g['mode'], g['stc']

    # The condition is in the name of the map as well, otherwise two conditions
    # of one run would write the same stc and the second would silently replace
    # the first
    fname = OUTPUT_DIR / f'group-{mode}-{cond["stem"]}-{QUANTITY}.stc'
    stc.save(fname, overwrite=True)
    logger.info(f'Saved into {fname}')

    rows.append(dict(mode=mode, epochs_fname=cond['epochs'],
                     tag=cond['tag'], contrast_tag=cond['contrast_tag'],
                     contrast_epochs=cond['contrast_epochs'],
                     quantity=QUANTITY, n_subjects=len(g['subjects']),
                     subjects=' '.join(g['subjects']),
                     missing=' '.join(g['missing']),
                     peak_time=round(g['times'][0], 4),
                     own_peak_time=round(g['own_peak'], 4),
                     rendered=' '.join(f'{t:.3f}' for t in g['times']),
                     unit='fT' if mode == 'MEG' else 'inverse solution',
                     clim=str(clim_of[(mode, cond['stem'])]),
                     stc=str(fname)))

    if not args.dry_run:
        all_images += render(stc, cond, mode, g['times'], SUBJECTS_DIR,
                             len(g['subjects']),
                             clim_of[(mode, cond['stem'])])

df = pd.DataFrame(rows)
display(df)
fname = OUTPUT_DIR / f'group-{MODES_TAG}-{SHEET_NAME}-{QUANTITY}-summary.csv'
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
