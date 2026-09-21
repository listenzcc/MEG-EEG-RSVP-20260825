#!/usr/bin/env zsh

# 文章图 7：在峰窗口上做解码。三个分析：
#
# 1. windows   在 baseline / 早窗(0.24-0.36 s) / 晚窗(0.41-0.55 s) 三个窗口里
#              分别解 target vs non-target。如果晚窗掉到 chance，说明第二个峰
#              不携带目标身份，它就不是目标成分。
# 2. transfer  在早窗训练、晚窗测试（以及反向）。能迁移说明两个峰是同一过程的
#              两个视角，不能迁移说明是两套活动。
# 3. rt        在 target 试次内部解 quick vs slow，沿时间滑动。问的是"epoch 在哪个
#              时刻已经知道被试会答多快"。其中 0.10-0.20 s 的窗口单独汇报：所有
#              被试 RT 的 5% 分位都 >= 0.211 s，这一段早于任何真实按键，解码结果
#              不可能由按键本身解释。曲线峰值落在 0.4-0.5 s（按键时刻）所以不能
#              当证据用。
#
# 已跑出的结果里，晚窗 [0.41, 0.55] s 的 target 身份（MEG 0.650 / EEG 0.668）与按键
# 时间重合，而按键只出现在 target 试次里：投影去掉的是平均按键响应，单试次潜伏期
# 抖动的残余仍能造出 target vs non-target 差异。要排除这个备择假设，加跑一遍对照：
#
#     python ./python/peak-window-decode.py -m $mode -s $subj --analysis windows --min-rt 0.55
#     python ./python/check-peak-window-decode.py --min-rt 0.55
#
# --min-rt 0.55 只保留按键晚于 0.55 s 的 target 试次，晚窗完全落在按键之前。
# 两套结果按 min_rt 分行存在同一张 csv 里，互不覆盖。
#
# 依赖：stage 3 的投影后 epochs，以及 stage 10 的 quick-slow-trials-rma.csv
# 产出：output/peak-window-decode/**（window/transfer/rt 三张 csv + rt 曲线）
#       output/peak-window-decode/group-peak-window-decode-rma.png
#       对照那套另存为 *-minrt0.55.*，不覆盖主结果

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/peak-window-decode.py
subjects=(S01 S02 S03 S04 S05 S06 S07 S08 S09 S10)

# ---- 主结果：全部 target 试次，三个分析都跑 ----
# 已经在 output/peak-window-decode/ 里了，脚本没改过就不用重跑，
# 所以这一段默认注释掉。
# for mode in EEG MEG; do
#     for subj in $subjects; do
#         python $script -m $mode -s $subj
#     done
# done
# python ./python/check-peak-window-decode.py

# ---- 晚窗的残余按键对照：要跑 ----
# 只保留按键晚于 0.55 s 的 target 试次，此时晚窗 [0.41, 0.55] s 完全落在按键
# 之前。晚窗的 target 身份（MEG 0.650 / EEG 0.668）如果在这一批试次上还在，
# 就不能再用"残余按键成分"解释；如果掉到 chance，图 7 的 late 结论就得改成
# "与残余按键不可分离"。两套结果按 min_rt 分行存在同一张 csv 里，产物带
# -minrt0.55 后缀，不覆盖主结果。
min_rt=0.55

for mode in EEG MEG; do
    for subj in $subjects; do
        python $script -m $mode -s $subj --analysis windows --min-rt $min_rt
    done
done

python ./python/check-peak-window-decode.py --min-rt $min_rt
