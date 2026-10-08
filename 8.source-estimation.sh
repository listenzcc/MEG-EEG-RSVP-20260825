#!/usr/bin/env zsh

# 文章图 4：源估计，分两组。
# A 组：10 Hz SSVEP 成分的溯源（图 2 的下游）。
#   1. 必须用没有被 notch 的 epochs，notch-epochs.py 已经把 10 / 20 Hz 去掉了，
#      epochs-{1,2}-epo.fif 是 notch 之前的文件。
#   2. events 1 / 2 都锁在刺激序列上，10 Hz 的相位在试次间是对齐的，可以平均；
#      events 3 是按键，和 10 Hz 序列不同相，平均之后成分会被抵消掉。
#   3. --cov_band control 用相邻频带(12-14 Hz)估计噪声，因为刺激前
#      0.5 s 里 10 Hz 刺激序列一直在放，同频带自化会把要定位的成分压掉。
# B 组：宽带 ERP 的溯源，按键投影前后各跑一遍（epochs-1-notch-epo.fif 与
#      epochs-1-notch-removal-artificial-epo.fif），这是"投影前峰值在运动皮层、
#      投影后转到 0.3 s 皮层源"这组对比图的数据来源。
#      另加 epochs-2-notch-removal-artificial-epo.fif，即投影后的非目标，
#      stage 13 的 target 减 non-target 对照读的就是它。
# 两组都建议先跑 9.ssvep-qc.sh 确认成分在（B 组不限频带，QC 主要针对 A 组）。
#
# --rt_group 是 stage 15 用的：把 target 试次按反应时切成 quick / slow 两组分别
# 溯源。那一段由 15.rt-source-contrast.sh 负责，不在这里跑。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/source-estimation.py

modes=(MEG EEG)
subjects=(S01 S02 S03 S04 S05 S06 S07 S08 S09 S10)

# ---- A 组：10 Hz SSVEP ----
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

# ---- B 组：宽带 ERP，按键投影前后 ----
erp_epochs_fnames=(epochs-1-notch-epo.fif epochs-1-notch-removal-artificial-epo.fif)

for mode in "${modes[@]}"; do
    for subj in "${subjects[@]}"; do
        for efname in "${erp_epochs_fnames[@]}"; do
            python $script --subject $subj --mode $mode --epochs_fname $efname
        done
    done
done

# ---- B 组之二：非目标（投影后），stage 13 的对照需要它 ----
# 13.group-source-map.sh 里 -c epochs-2-notch-removal-artificial-epo.fif 那一行
# 要的是非目标在投影后的 stc。少了这一段，那两条命令会因为找不到文件而中断，
# group-source-summary.csv 里也就不会出现 target 减 non-target 的行。
#
# 读的时候注意口径：投影算子是从非目标与按键试次上估出来的，所以非目标的 ERP
# 本身也被投掉了一部分。这个差是「两个条件在同一个投影下之差」，不是「非目标当
# 基线」——removal 之后的 evt=2 不能拿来当基线用。
for mode in "${modes[@]}"; do
    for subj in "${subjects[@]}"; do
        python $script --subject $subj --mode $mode \
            --epochs_fname epochs-2-notch-removal-artificial-epo.fif
    done
done
