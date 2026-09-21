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
#              不可能由按键本身解释。
#
# 依赖：stage 3 的投影后 epochs，以及 stage 10 的 quick-slow-trials-rma.csv
# 产出：output/peak-window-decode/**（window/transfer/rt 三张 csv + rt 曲线）
#       output/peak-window-decode/group-peak-window-decode-rma.png

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/peak-window-decode.py

for mode in EEG MEG; do
    for subj in S01 S02 S03 S04 S05 S06 S07 S08 S09 S10; do
        python $script -m $mode -s $subj
    done
done

python ./python/check-peak-window-decode.py
