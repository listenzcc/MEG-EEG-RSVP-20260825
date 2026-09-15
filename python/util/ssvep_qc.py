"""
File: ssvep_qc.py
Author: Chuncheng Zhang
Date: 2026-09-15
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Shared helpers for the SSVEP component quality check and extraction.

    They are used by both python/ssvep-qc.py (sensor level QC only) and
    python/source-estimation.py (QC before the inverse solution), so the
    numbers and the figures of the two scripts are always consistent.

Functions:
    1. component_band_ratio
    2. extract_component
    3. save_component_qc_figure
"""

# %% ---- 2026-09-15 ------------------------
# Requirements and constants
from util.easy_imports import *
from scipy.signal import hilbert


# %% ---- 2026-09-15 ------------------------
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


def extract_component(epochs, freq, bw, n_jobs=n_jobs):
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
        n_jobs: The number of jobs for the filtering

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


def save_component_qc_figure(psd, evoked, ratio, freq, bw, fpath, mode='EEG'):
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
        mode: The recording mode, it selects the amplitude unit
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
    scale, unit = (1e15, 'fT') if mode == 'MEG' else (1e6, 'uV')
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


# %% ---- 2026-09-15 ------------------------
# Pending
