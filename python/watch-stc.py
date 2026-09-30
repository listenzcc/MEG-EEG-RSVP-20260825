"""
File: watch-stc.py
Author: Chuncheng Zhang
Date: 2026-09-30
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Watch the stc in 3d view.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-09-30 ------------------------
# Requirements and constants
import argparse
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(description='3D view for stc.plot')
parser.add_argument('-f', '--file', help='Select a .stc file')

args = parser.parse_args()
logger.info(f'Start with {args=}')

# %% ---- 2026-09-30 ------------------------
# Function and class



# %% ---- 2026-09-30 ------------------------
# Play ground
stc = mne.read_source_estimate(args.file, subject='fsaverage')
stc.plot(
    hemi='both',
)

input('')



# %% ---- 2026-09-30 ------------------------
# Pending



# %% ---- 2026-09-30 ------------------------
# Pending
