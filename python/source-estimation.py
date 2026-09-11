"""
File: source-estimation.py
Author: Chuncheng Zhang
Date: 2026-09-11
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Source estimation for multiple type MEG/EEG evoked.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-09-11 ------------------------
# Requirements and constants
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse, write_inverse_operator

from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Require SUBJ and MODE parameters')
parser.add_argument('-s', '--subject', default='S02',
                    help='Subject name like S02')
parser.add_argument('-m', '--mode', default='EEG', help='Mode name EEG | MEG')
parser.add_argument('-e', '--epochs_fname', default='epochs-1-notch-epo.fif',
                    help='Epochs fname, it should be inside the $DATA_DIR')

args = parser.parse_args()
SUBJ = args.subject
MODE = args.mode
EPOCHS_FNAME = args.epochs_fname

print(args)

logger.info(f'Start with {args=}')

# %%
DATA_DIR = Path(f'output/epochs/{MODE}-{SUBJ}')
OUTPUT_DIR = Path(f'output/source-estimation/{MODE}-{SUBJ}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-09-11 ------------------------
# Function and class


# %% ---- 2026-09-11 ------------------------
# Play ground
epochs = mne.read_epochs(DATA_DIR / EPOCHS_FNAME)

evoked = epochs.average()

if MODE == 'EEG':
    evoked.set_eeg_reference('average', projection=True)

times = evoked.times

print(epochs)
print(evoked)
print(times)

# %%
# fpath of the stc file
OUTPUT_FPATH = OUTPUT_DIR / f'{EPOCHS_FNAME}.ave.stc'
logger.debug(f'The stc file(s) will be saved in {OUTPUT_FPATH}')

# %%
# 1. 准备模板数据（使用fsaverage模板）
print("下载fsaverage模板...")
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)

# 2. 加载模板的MRI数据
trans = 'fsaverage'  # 使用模板的trans
src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

# 4. 创建正向算子（需要设置电极位置）
# 计算电极位置与模板的配准
# 使用标准电极位置配准到fsaverage
montage = evoked.get_montage()
if montage is None:
    print("evoked没有电极位置信息，使用标准1020系统")
    montage = mne.channels.make_standard_montage('standard_1020')
    evoked.set_montage(montage)

# 5. 计算正向解
print("计算正向解...")
# 创建源空间
src = mne.read_source_spaces(src)

# 计算导联场矩阵
fwd = mne.make_forward_solution(
    evoked.info,
    trans=trans,
    src=src,
    bem=bem,
    eeg=MODE == 'EEG',
    meg=MODE == 'MEG'
)
print(f"正向解计算完成: {fwd}")

# 6. 计算噪声协方差矩阵
# 使用0秒之前的数据计算协方差矩阵
baseline_data = evoked.copy().crop(evoked.tmin, 0).data
cov = mne.Covariance(
    data=np.cov(baseline_data),
    names=evoked.info['ch_names'],
    bads=[],
    projs=[],
    nfree=baseline_data.shape[1]
)

# 7. 创建逆算子
print("创建逆算子...")
inverse_operator = make_inverse_operator(
    evoked.info,
    fwd,
    cov,
    loose=0.2,  # 松耦合约束
    depth=0.8   # 深度加权
)

# 8. 应用逆算子进行溯源
print("进行溯源计算...")
snr = 3.0  # 信噪比
lambda2 = 1.0 / snr ** 2

stc = apply_inverse(
    evoked,
    inverse_operator,
    lambda2=lambda2,
    pick_ori='normal',
    method='MNE',  # 可以使用 'dSPM', 'sLORETA', 'eLORETA'
)

print(f"溯源结果: {stc}")
stc.save(OUTPUT_FPATH)
logger.debug(f'Saved into {OUTPUT_FPATH}')

# 选择峰值时间点
# stc.plot(hemi='both', subjects_dir=subjects_dir, subject='fsaverage')
# input('Press enter to escape.')
# exit(0)

# %% ---- 2026-09-11 ------------------------
# Pending


# %% ---- 2026-09-11 ------------------------
# Pending
