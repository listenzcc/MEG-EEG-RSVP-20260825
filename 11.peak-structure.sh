#!/usr/bin/env zsh

# 文章图 6：目标响应的峰结构。
# 动机：投影后的 target 响应在 MEG 里是两个峰（约 0.29 s 和 0.47 s，中间 0.41 s
# 有一个谷），在 EEG 里是一个峰。quick-slow-analysis.py 只取窗内最大值，于是
# 它给一部分被试报第一个峰、给另一部分报第二个峰，MEG 的峰潜伏期分布因此变成
# 双峰，配对检验也就失去了检验力（见 output/quick-slow/group-summary-rma.csv,
# MEG p=0.23 但 EEG p=0.002）。这个 stage 把两个峰分开测量。
#
# 单被试的 GFP 有残余高频抖动（见 output/quick-slow/MEG-S04/quick-slow-rma.png 的
# 窄尖峰串），所以检测前先做 15 ms 高斯平滑，--smooth 0 可关掉；--min-ratio 0.35
# 要求第二峰至少达到第一峰的 35%，免得噪声鼓包被当成第二峰。稳健性检查建议把
# --smooth 取 0 / 10 / 15 / 25 ms 各跑一遍。
#
# 输入：epochs-{1,2}-notch-removal-artificial-ave.fif（来自 stage 3）
#       output/quick-slow/{MODE}-{SUBJ}/target-rma-{quick,slow}-ave.fif（stage 10）
# 产出：output/peak-structure/peak-structure-rma.csv
#       output/peak-structure/{MODE}-{SUBJ}-peak-structure-rma.png
#       output/peak-structure/group-peak-structure-rma.png + 三个 group csv

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/peak-structure.py

for mode in EEG MEG; do
    for subj in S01 S02 S03 S04 S05 S06 S07 S08 S09 S10; do
        python $script -m $mode -s $subj
    done
done

python ./python/check-peak-structure.py
