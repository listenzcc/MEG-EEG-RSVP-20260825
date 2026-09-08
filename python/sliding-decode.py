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
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from mne.decoding import SlidingEstimator, cross_val_multiscore, Vectorizer
from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Require SUBJ and MODE parameters')
parser.add_argument('-s', '--subject', default='S02',
                    help='Subject name like S02')
parser.add_argument('-m', '--mode', default='EEG', help='Mode name EEG | MEG')
parser.add_argument('-a', '--flag-remove-artificial', action='store_true', help='using the remove-artificial-epochs')
args = parser.parse_args()
SUBJ = args.subject
MODE = args.mode
FLAG_REMOVE_ARTIFICIAL = args.flag_remove_artificial
print(args)

logger.info(f'Start with {args=}')

# %%
DATA_DIR = Path(f'output/epochs/{MODE}-{SUBJ}')

OUTPUT_DIR = Path(f'output/sliding-decode/{MODE}-{SUBJ}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# %% ---- 2026-09-07 ------------------------
# Function and class


# %% ---- 2026-09-07 ------------------------
# Play ground
# Target (1) vs Non-target (2)

if FLAG_REMOVE_ARTIFICIAL:
    epochs_1 = mne.read_epochs(DATA_DIR / 'epochs-1-notch-removal-artificial-epo.fif')
    epochs_2 = mne.read_epochs(DATA_DIR / 'epochs-2-notch-removal-artificial-epo.fif')
else:
    epochs_1 = mne.read_epochs(DATA_DIR / 'epochs-1-notch-epo.fif')
    epochs_2 = mne.read_epochs(DATA_DIR / 'epochs-2-notch-epo.fif')

# epochs_3 = mne.read_epochs(DATA_DIR / 'epochs-3-notch-epo.fif')
epochs_all = mne.concatenate_epochs([epochs_1, epochs_2])
print(epochs_all)


# %% ---- 2026-09-07 ------------------------
# Pending
# 获取数据X和标签y
X = epochs_all.get_data()  # shape: (n_epochs, n_channels, n_times)
y = epochs_all.events[:, -1]  # 标签 1,2,3,4 ...
times = epochs_all.times  # 时间点

# 分类器（每个时间点都会用）
clf = make_pipeline(
    Vectorizer(),          # (n_channels, n_times) → (features)
    StandardScaler(),
    SVC(kernel='rbf')
)

time_decod = SlidingEstimator(
    clf,
    scoring='roc_auc',
    n_jobs=-1
)

# 10-fold CV
cv = StratifiedKFold(n_splits=10, shuffle=True)

scores = cross_val_multiscore(
    time_decod,
    X,
    y,
    cv=cv,
    n_jobs=-1
)

# 平均 across folds
scores_mean = scores.mean(axis=0)

print(scores)
print(scores_mean)
print(scores.shape)  # (n_splits, n_times)

if FLAG_REMOVE_ARTIFICIAL:
    scores_path = OUTPUT_DIR / 'scores-rma.txt'
else:
    scores_path = OUTPUT_DIR / 'scores.txt'
np.savetxt(scores_path, scores, fmt='%.6f')
logger.info(f'Saved scores to {scores_path}')

# %% ---- 2026-09-07 ------------------------
# Pending
