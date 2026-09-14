"""
File: source-estimation.py
Author: Chuncheng Zhang
Date: 2026-09-11
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Source estimation for multiple type MEG/EEG evoked.

    With --ssvep_freq the requested frequency component, which is the 10 Hz
    SSVEP here, is extracted from the epochs before the inverse solution, so the
    source map describes the generator of the component instead of the whole
    broadband ERP.

    Note that notch-epochs.py removes the 10 and 20 Hz from the epochs, so the
    notched epochs can not be used for the 10 Hz SSVEP and this script refuses
    them.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-09-11 ------------------------
# Requirements and constants
from scipy.signal import hilbert
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              apply_inverse_epochs, write_inverse_operator)

from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Require SUBJ and MODE parameters')
parser.add_argument('-s', '--subject', default='S02',
                    help='Subject name like S02')
parser.add_argument('-m', '--mode', default='EEG', help='Mode name EEG | MEG')
parser.add_argument('-e', '--epochs_fname', default='epochs-1-notch-epo.fif',
                    help='Epochs fname, it should be inside the $DATA_DIR')

# The SSVEP component extraction
parser.add_argument('--ssvep_freq', type=float, default=None,
                    help='Extract this frequency component before the source estimation, e.g. 10 for the 10 Hz SSVEP')
parser.add_argument('--ssvep_bw', type=float, default=1.,
                    help='Half bandwidth (Hz) of the component band, the band is [freq - bw, freq + bw]')
parser.add_argument('--ssvep_min_ratio', type=float, default=1.5,
                    help='Minimal band to neighbour power ratio to accept the epochs as containing the component')
parser.add_argument('--no_power', action='store_true',
                    help='Skip the trial-wise power estimate of the component')

# The inverse solution
parser.add_argument('--snr', type=float, default=3.,
                    help='SNR used by lambda2 = 1 / snr ** 2')
parser.add_argument('--method', default='MNE',
                    choices=['MNE', 'dSPM', 'sLORETA', 'eLORETA'],
                    help='Inverse method')
parser.add_argument('--cov_method', default='shrunk',
                    choices=['shrunk', 'empirical', 'auto'],
                    help='Noise covariance estimator')
parser.add_argument('--cov_band', default='same', choices=['same', 'control'],
                    help='Noise covariance band, same = the component band, control = the neighbour band')
parser.add_argument('--trans', default='fsaverage',
                    help="Head to MRI transformation, 'fsaverage' for the template brain")
parser.add_argument('--force', action='store_true',
                    help='Continue even if the component is not found in the epochs')

args = parser.parse_args()
SUBJ = args.subject
MODE = args.mode
EPOCHS_FNAME = args.epochs_fname

SSVEP_FREQ = args.ssvep_freq
SSVEP_BW = args.ssvep_bw
SNR = args.snr
METHOD = args.method
COV_METHOD = args.cov_method
COV_BAND = args.cov_band
PICK_ORI = 'normal'

logger.info(f'Start with {args=}')

# %%
DATA_DIR = Path(f'output/epochs/{MODE}-{SUBJ}')
OUTPUT_DIR = Path(f'output/source-estimation/{MODE}-{SUBJ}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# The tag of the outputs, the tag of the plain ERP is 'ave' to keep it unchanged
TAG = 'ave' if SSVEP_FREQ is None else f'ssvep{SSVEP_FREQ:g}'

# The 10 Hz SSVEP lives in the epochs without the 10 Hz notch
if SSVEP_FREQ is not None and 'notch' in EPOCHS_FNAME:
    raise SystemExit(
        f'{EPOCHS_FNAME} is the notched epochs, the {SSVEP_FREQ:g} Hz component '
        'has been removed by notch-epochs.py. Use the epochs before the notch, '
        'e.g. epochs-1-epo.fif.')

# %% ---- 2026-09-11 ------------------------
# Function and class
def component_band_ratio(epochs, freq, bw):
    '''
    Measure whether the requested component is present in the epochs.

    The spectrum is computed on the single trials, the power of the component
    band [freq - bw, freq + bw] is compared with the neighbour bands
    [freq - 4bw, freq - 2bw] and [freq + 2bw, freq + 4bw].
    A ratio around 1 means the epochs do not contain the component, which is
    the case when notch-epochs.py has already removed it, or when the epochs
    are not aligned to the stimulus of the component.

    Args:
        epochs: MNE Epochs
        freq: The central frequency of the component
        bw: The half bandwidth of the component band

    Returns:
        psd: The spectrum of the epochs
        ratio: The band to neighbour power ratio of each channel, (n_channels,)
    '''
    psd = epochs.compute_psd(
        fmin=2., fmax=min(40., epochs.info['sfreq'] / 2. - 1.), verbose='ERROR')

    freqs = psd.freqs
    data = psd.get_data()
    # The spectrum of the epochs is (n_epochs, n_channels, n_freqs)
    if data.ndim == 3:
        data = data.mean(axis=0)

    band = (freqs >= freq - bw) & (freqs <= freq + bw)
    neighbour = ((freqs >= freq - 4 * bw) & (freqs <= freq - 2 * bw)) | \
        ((freqs >= freq + 2 * bw) & (freqs <= freq + 4 * bw))

    ratio = np.median(data[:, band], axis=1) / \
        np.median(data[:, neighbour], axis=1)
    return psd, ratio


def save_component_qc_figure(psd, evoked, ratio, freq, bw, fpath):
    '''
    Save the figure to check the component extraction.

    The figure shows the spectrum of the epochs with the component band marked,
    the topography of the component band power, and the band passed evoked of
    the channel with the strongest component.

    Args:
        psd: The spectrum of the epochs, measured before the extraction
        evoked: The band passed evoked
        ratio: The band to neighbour power ratio of each channel
        freq: The central frequency of the component
        bw: The half bandwidth of the component band
        fpath: The fname of the figure
    '''
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    psd.plot(axes=axes[0], show=False, spatial_colors=False)
    axes[0].axvspan(freq - bw, freq + bw, color='tab:red', alpha=0.15)
    axes[0].axvline(freq, color='tab:red', ls='--')
    axes[0].set_title(f'Spectrum, {freq:g} Hz to neighbour power ratio: '
                      f'median {np.median(ratio):.2f}, max {ratio.max():.2f}')

    data = psd.get_data()
    if data.ndim == 3:
        data = data.mean(axis=0)
    band = (psd.freqs >= freq - bw) & (psd.freqs <= freq + bw)
    band_power = data[:, band].mean(axis=1)

    mne.viz.plot_topomap(band_power, evoked.info, axes=axes[1],
                         show=False, cmap='Reds', contours=3)
    axes[1].set_title(f'{freq:g} Hz power')

    # The channel with the strongest component
    pick = int(np.argmax(band_power))
    scale, unit = (1e15, 'fT') if MODE == 'MEG' else (1e6, 'uV')
    axes[2].plot(evoked.times, evoked.data[pick] * scale, label='band passed')
    axes[2].plot(evoked.times, np.abs(hilbert(evoked.data[pick])) * scale,
                 label='Hilbert envelope')
    axes[2].axvline(0, color='k', ls=':')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel(f'Amplitude ({unit})')
    axes[2].set_title(evoked.ch_names[pick])
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(fpath)
    plt.close(fig)
    logger.debug(f'Saved into {fpath}')


def extract_component(epochs, freq, bw):
    '''
    Extract the requested component from the epochs with a zero phase band pass
    filter.

    The filter is applied to the single trials, so both the phase locked and the
    trial wise part of the component are kept. It is zero phase, otherwise the
    phase of the component is shifted and the average is not aligned to the
    stimulus onset.

    Args:
        epochs: MNE Epochs
        freq: The central frequency of the component
        bw: The half bandwidth of the component band

    Returns:
        epochs_band: MNE Epochs in the [freq - bw, freq + bw] band
    '''
    data = mne.filter.filter_data(
        epochs.get_data(copy=True),
        sfreq=epochs.info['sfreq'],
        l_freq=freq - bw,
        h_freq=freq + bw,
        method='iir',
        iir_params=dict(order=4, ftype='butter', output='sos'),
        n_jobs=n_jobs,
        copy=False,
        verbose='ERROR',
    )

    epochs_band = mne.EpochsArray(
        data, epochs.info, epochs.events, epochs.times[0], epochs.event_id,
        verbose='ERROR')
    logger.debug(f'Extracted the {freq:g} Hz component, {epochs_band=}')
    return epochs_band


def component_power_stc(epochs, inverse_operator, lambda2, fname_mean, fname_std):
    '''
    Estimate the trial wise power of the component in the source space.

    Each single trial, which is already band passed at the component frequency,
    is projected into the source space, the Hilbert envelope of the source time
    course is computed, and the envelopes are averaged across the trials.
    The result is the power of the component on each source, it does not require
    the component to be phase locked across the trials.

    Args:
        epochs: MNE Epochs in the component band
        inverse_operator: The inverse operator
        lambda2: The regularization of the inverse solution
        fname_mean: The fname of the averaged envelope
        fname_std: The fname of the deviation of the envelope

    Returns:
        mean, std: The averaged envelope and its deviation across the trials
    '''
    total, square, template, n_trials = None, None, None, 0
    for stc in apply_inverse_epochs(
            epochs, inverse_operator, lambda2,
            method=METHOD, pick_ori=PICK_ORI,
            return_generator=True, verbose='ERROR'):
        envelope = stc.apply_hilbert(envelope=True).data.astype(np.float64)
        if total is None:
            total, square, template = np.zeros_like(
                envelope), np.zeros_like(envelope), stc
        total += envelope
        square += envelope ** 2
        n_trials += 1

    if n_trials == 0:
        logger.warning('There is no trial, the power estimate is skipped.')
        return None, None

    mean = total / n_trials
    std = np.sqrt(np.maximum(square / n_trials - mean ** 2, 0.))

    stc_mean = template.copy()
    stc_mean.data = mean
    stc_mean.save(fname_mean, overwrite=True)
    logger.debug(f'Saved into {fname_mean}, {n_trials=}')

    stc_std = template.copy()
    stc_std.data = std
    stc_std.save(fname_std, overwrite=True)
    logger.debug(f'Saved into {fname_std}')

    return mean, std


# %% ---- 2026-09-11 ------------------------
# Play ground
epochs = mne.read_epochs(DATA_DIR / EPOCHS_FNAME,
                         preload=True, verbose='ERROR')
logger.info(f'Loaded {EPOCHS_FNAME=}, {epochs=}')

# EEG的溯源要求平均参考，这里作为投影加进info，溯源和噪声协方差都用这个参考
if MODE == 'EEG':
    epochs = epochs.set_eeg_reference('average', projection=True)
    logger.debug(f'Added the average reference, {epochs.info["projs"]=}')

# The noise covariance is measured on the epochs before the extraction when the
# control band is used
epochs_cov = None

if SSVEP_FREQ is not None:
    psd, ratio = component_band_ratio(epochs, SSVEP_FREQ, SSVEP_BW)
    logger.info(
        f'{SSVEP_FREQ:g} Hz to neighbour power ratio: '
        f'median {np.median(ratio):.2f}, max {ratio.max():.2f} '
        f'({epochs.ch_names[int(np.argmax(ratio))]})')

    # The pre-stimulus window keeps the ongoing stimulus train, so the noise of
    # the component band itself is not a noise floor. The neighbour band is
    # measured instead, it keeps the component from being whitened away.
    if COV_BAND == 'control':
        l_cov, h_cov = SSVEP_FREQ + 2 * SSVEP_BW, SSVEP_FREQ + 4 * SSVEP_BW
        logger.info(f'The noise is measured in [{l_cov:g}, {h_cov:g}] Hz')
        epochs_cov = extract_component(
            epochs, SSVEP_FREQ + 3 * SSVEP_BW, SSVEP_BW)

    epochs = extract_component(epochs, SSVEP_FREQ, SSVEP_BW)

evoked = epochs.average()

if SSVEP_FREQ is not None:
    fname = OUTPUT_DIR / f'{EPOCHS_FNAME}.{TAG}-ave.fif'
    evoked.save(fname, overwrite=True, verbose='ERROR')
    logger.debug(f'Saved into {fname}')

    fname = OUTPUT_DIR / f'{EPOCHS_FNAME}.{TAG}-qc.png'
    try:
        save_component_qc_figure(
            psd, evoked, ratio, SSVEP_FREQ, SSVEP_BW, fname)
    except Exception as e:
        logger.warning(f'Failed to save the qc figure: {e}')

    # The component must be present, otherwise the source map is meaningless
    if ratio.max() < args.ssvep_min_ratio and not args.force:
        raise SystemExit(
            f'There is no {SSVEP_FREQ:g} Hz component in {EPOCHS_FNAME}, '
            f'the band to neighbour power ratio is {ratio.max():.2f} < '
            f'{args.ssvep_min_ratio}. Check the epochs are not notched and are '
            'aligned to the stimulus onset. Use --force to continue anyway.')

# %% ---- 2026-09-11 ------------------------
# 1. 准备模板数据（使用fsaverage模板）
print("下载fsaverage模板...")
fs_dir = fetch_fsaverage(verbose=False)
subjects_dir = os.path.dirname(fs_dir)

# 2. 加载模板的MRI数据
trans = args.trans  # 默认使用模板的trans
src_fname = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem_fname = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
logger.debug(f'{trans=}, {src_fname=}, {bem_fname=}')

# 3. 电极位置，EEG的溯源需要数字化电极位置
montage = evoked.get_montage()
if montage is None and MODE == 'EEG':
    logger.warning(
        'evoked没有电极位置信息，使用标准1020系统')
    montage = mne.channels.make_standard_montage('standard_1020')
    evoked.set_montage(montage)

# 4. 创建源空间和正向算子
print("计算正向解...")
src = mne.read_source_spaces(src_fname)

fwd = mne.make_forward_solution(
    evoked.info,
    trans=trans,
    src=src,
    bem=bem_fname,
    eeg=MODE == 'EEG',
    meg=MODE == 'MEG',
    n_jobs=n_jobs,
    verbose='ERROR',
)
print(f"正向解计算完成: {fwd}")

# 5. 计算噪声协方差矩阵
# 使用刺激前的时间窗估计噪声，噪声必须和参与溯源的数据在同一个频带内
if epochs_cov is None:
    epochs_cov = epochs
    if SSVEP_FREQ is not None:
        logger.debug(f'The noise is measured in the {TAG} band itself')

try:
    cov = mne.compute_covariance(
        epochs_cov, tmax=0., method=COV_METHOD, verbose='ERROR')
except (ValueError, ImportError) as e:
    logger.warning(f'{COV_METHOD} is not available, empirical is used, {e}')
    cov = mne.compute_covariance(
        epochs_cov, tmax=0., method='empirical', verbose='ERROR')
logger.debug(f'{cov=}')

# 6. 创建逆算子
print("创建逆算子...")
inverse_operator = make_inverse_operator(
    evoked.info,
    fwd,
    cov,
    loose=0.2,  # 松耦合约束
    depth=0.8,  # 深度加权
    verbose='ERROR',
)

# 7. 应用逆算子进行溯源
print("进行溯源计算...")
lambda2 = 1.0 / SNR ** 2
logger.debug(f'{SNR=}, {lambda2=}, {METHOD=}, {PICK_ORI=}')

stc = apply_inverse(
    evoked,
    inverse_operator,
    lambda2=lambda2,
    pick_ori=PICK_ORI,
    method=METHOD,
)

print(f"溯源结果: {stc}")
if SSVEP_FREQ is None:
    FNAME_STC = OUTPUT_DIR / f'{EPOCHS_FNAME}.ave.stc'
else:
    FNAME_STC = OUTPUT_DIR / f'{EPOCHS_FNAME}.{TAG}-evoked.stc'
stc.save(FNAME_STC, overwrite=True)
logger.debug(f'Saved into {FNAME_STC}')

# 8. 提取成分在源空间的功率
# 逐试次投影后取Hilbert包络再平均，得到的是该频率成分的功率，与试次间的相位无关
if SSVEP_FREQ is not None and not args.no_power:
    FNAME_POWER = OUTPUT_DIR / f'{EPOCHS_FNAME}.{TAG}-power.stc'
    FNAME_POWER_STD = OUTPUT_DIR / f'{EPOCHS_FNAME}.{TAG}-power-std.stc'
    component_power_stc(
        epochs, inverse_operator, lambda2, FNAME_POWER, FNAME_POWER_STD)

# %% ---- 2026-09-11 ------------------------
# Pending
# 用DICS波束形成器做10 Hz的频域溯源，需要刺激期和基线的对比


# %% ---- 2026-09-11 ------------------------
# Pending
