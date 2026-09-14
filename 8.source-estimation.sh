#!/usr/bin/env zsh

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/source-estimation.py

modes=(MEG EEG)
subjects=(S01 S02 S03 S04 S05 S06 S07 S08 S09 S10)

# 10 Hz SSVEP 的成分溯源。
# 1. 必须用没有被 notch 的 epochs，notch-epochs.py 已经把 10 / 20 Hz 去掉了，
#    epochs-{1,2}-epo.fif 是 notch 之前的文件。
# 2. events 1 / 2 都锁在刺激序列上，10 Hz 的相位在试次间是对齐的，可以平均；
#    events 3 是按键，和 10 Hz 序列不同相，平均之后成分会被抵消掉。
# 3. --cov_band control 用相邻频带(12-14 Hz)估计噪声，因为刺激前的
#    0.5 s 里 10 Hz 刺激序列一直在放，同频带自化会把要定位的成分压掉。
ssvep_freq=10
ssvep_epochs_fnames=(epochs-1-epo.fif epochs-2-epo.fif)

for mode in "${modes[@]}"; do
    for subj in "${subjects[@]}"; do
        for efname in "${ssvep_epochs_fnames[@]}"; do
            python $script --subject $subj --mode $mode --epochs_fname $efname \
                --ssvep_freq $ssvep_freq --cov_band control
        done
    done
done

# 宽带 ERP 的溯源，和上面分开跑，取消注释即可。
# erp_epochs_fnames=(epochs-1-notch-removal-artificial-epo.fif)
# for mode in "${modes[@]}"; do
#     for subj in "${subjects[@]}"; do
#         for efname in "${erp_epochs_fnames[@]}"; do
#             python $script --subject $subj --mode $mode --epochs_fname $efname
#         done
#     done
# done
