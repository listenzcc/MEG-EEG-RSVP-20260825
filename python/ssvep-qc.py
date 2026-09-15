"""
File: ssvep-qc.py
Author: Chuncheng Zhang
Date: 2026-09-15
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Sensor level quality check of the SSVEP component.

    It is the cheap first step of the component pipeline, it does not need
    the fsaverage template nor the forward solution. For every requested
    epochs file it reports the band to neighbour power ratio, the topography
    of the component band power, and the band passed evoked with its Hilbert
    envelope of the strongest channel.

    It accepts both the target (epochs-1) and the non-target (epochs-2)
    epochs: the component is driven by the stimulus sequence, so it should
    be present in both, and the figure documents exactly that.

Usage:
    python python/ssvep-qc.py --subject S01 --mode MEG --epochs_fname epochs-1-epo.fif
    python python/ssvep-qc.py --subject S01 --mode EEG --all_events

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-09-15 ------------------------
# Requirements and constants
from util.easy_imports import *
from util.ssvep_qc import component_band_ratio, extract_component, \
    save_component_qc_figure

# %%
parser = argparse.ArgumentParser(
    description='Require SUBJ and MODE parameters')
parser.add_argument('-s', '--subject', default='S01',
                    help='Subject name like S01')
parser.add_argument('-m', '--mode', default='MEG', help='Mode name EEG | MEG')
parser.add_argument('-e', '--epochs_fname', default='epochs-1-epo.fif',
                    help='Epochs fname, it should be inside the $DATA_DIR')
parser.add_argument('--freq', type=float, default=10.,
                    help='Central frequency of the component')
parser.add_argument('--bw', type=float, default=1.,
                    help='Half bandwidth (Hz) of the component band')
parser.add_argument('--min_ratio', type=float, default=1.5,
                    help='Minimal band to neighbour power ratio to pass the check')
parser.add_argument('--all_events', action='store_true',
                    help='Check epochs-1 and epochs-2 in addition to --epochs_fname')

args = parser.parse_args()
SUBJ = args.subject
MODE = args.mode

logger.info(f'Start with {args=}')

# %%
DATA_DIR = Path(f'output/epochs/{MODE}-{SUBJ}')
OUTPUT_DIR = Path(f'output/ssvep-qc/{MODE}-{SUBJ}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TAG = f'ssvep{args.freq:g}'


# %% ---- 2026-09-15 ------------------------
# Function and class
def check_one(fname: Path):
    '''Run the sensor level QC on one epochs file.

    Args:
        fname: The epochs file name inside $DATA_DIR

    Returns:
        passed, ratio_max: Whether the component passes the --min_ratio check
    '''
    epochs = mne.read_epochs(DATA_DIR / fname, preload=True, verbose='ERROR')

    psd, ratio = component_band_ratio(epochs, args.freq, args.bw)
    logger.info(f'{fname}: {args.freq:g} Hz to neighbour power ratio: '
                f'median {np.median(ratio):.2f}, max {ratio.max():.2f} '
                f'({epochs.ch_names[int(np.argmax(ratio))]})')

    epochs_band = extract_component(epochs, args.freq, args.bw)
    evoked = epochs_band.average()

    fname_img = OUTPUT_DIR / f'{fname.name}.{TAG}-qc.png'
    save_component_qc_figure(
        psd, evoked, ratio, args.freq, args.bw, fname_img, mode=MODE)
    logger.debug(f'Saved into {fname_img}')

    passed = ratio.max() >= args.min_ratio
    if passed:
        logger.info(f'{fname.name}: PASSED (ratio {ratio.max():.2f})')
    else:
        logger.warning(
            f'{fname.name}: FAILED (ratio {ratio.max():.2f} < '
            f'{args.min_ratio}), the component is absent, do not feed this '
            'file to source-estimation.py --ssvep_freq')
    return passed, ratio.max()


# %% ---- 2026-09-15 ------------------------
# Play ground
fnames = [Path(args.epochs_fname)]
if args.all_events:
    for evt in ['1', '2']:
        fname = Path(f'epochs-{evt}-epo.fif')
        if fname != fnames[0]:
            fnames.append(fname)

results = {}
for fname in fnames:
    if not (DATA_DIR / fname).exists():
        logger.warning(f'{fname} is not in {DATA_DIR}, skipped')
        continue
    passed, ratio_max = check_one(fname)
    results[str(fname)] = (passed, ratio_max)

# %%
if not results:
    raise FileNotFoundError(f'No epochs found in {DATA_DIR}')

df = pd.DataFrame(
    [(MODE, SUBJ, k, v[1], v[0]) for k, v in results.items()],
    columns=['mode', 'subject', 'epochs_fname', 'ratio_max', 'passed'])
display(df)

fname_csv = OUTPUT_DIR / f'{TAG}-qc-summary.csv'
if fname_csv.exists():
    df.to_csv(fname_csv, mode='a', header=False, index=False)
else:
    df.to_csv(fname_csv, index=False)
logger.info(f'Appended into {fname_csv}')


# %% ---- 2026-09-15 ------------------------
# Pending
