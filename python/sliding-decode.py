"""
File: sliding-decode.py
Author: Chuncheng Zhang
Date: 2026-09-07
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Sliding decode among the epochs.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-09-07 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Require SUBJ and MODE parameters')
parser.add_argument('-s', '--subject', default='S02',
                    help='Subject name like S02')
parser.add_argument('-m', '--mode', default='EEG', help='Mode name EEG | MEG')
args = parser.parse_args()
SUBJ = args.subject
MODE = args.mode

logger.info(f'Start with {SUBJ=}, {MODE=}')

# %% ---- 2026-09-07 ------------------------
# Function and class


# %% ---- 2026-09-07 ------------------------
# Play ground


# %% ---- 2026-09-07 ------------------------
# Pending


# %% ---- 2026-09-07 ------------------------
# Pending
