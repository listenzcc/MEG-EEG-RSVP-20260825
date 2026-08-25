"""
File: summary_data.py
Author: Chuncheng Zhang
Date: 2026-01-26
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary the dataset.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-01-26 ------------------------
# Requirements and constants
from .util.easy_imports import *


# %% ---- 2026-01-26 ------------------------
# Function and class
def summarize_dataset(data_path: Path) -> pd.DataFrame:
    """
    Summarize the dataset located at data_path.

    Parameters:
        data_path (Path): Path to the dataset directory.

    Returns:
        pd.DataFrame: Summary of the dataset.
    """
    data_path = Path(data_path)
    summary = []
    logger.debug(f"Summarizing dataset at {data_path}")
    for file in data_path.rglob('*ica-raw.fif'):
        raw = mne.io.read_raw_fif(file, preload=False)
        info = raw.info
        summary.append({
            'file_name': file.name,
            'n_channels': info['nchan'],
            'sfreq': info['sfreq'],
            'duration_sec': raw.n_times / info['sfreq']
        })
    return pd.DataFrame(summary)


# %% ---- 2026-01-26 ------------------------
# Play ground


# %% ---- 2026-01-26 ------------------------
# Pending


# %% ---- 2026-01-26 ------------------------
# Pending
