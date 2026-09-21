#!/usr/bin/env zsh

# 文章图 5：按反应时间把 TARGET 试次分成 quick / slow。
# 输入是 3.remove-keypress-artificial.sh 产出的投影后 epochs，
# 所以两组的差异反映的是目标特异活动，而不是反应锁时的按键成分。
#
# 分组规则按 --split auto 依次尝试：
#   1. median    该被试自己的反应时间中位数，两组都要够试次且中位 RT 相差 >= 30 ms
#   2. threshold 固定 0.4 s 阈值
#   3. tertile   最快与最慢各三分之一
# 每条被试实际用了哪条规则会记在 output/quick-slow/summary-rma.csv 里。
#
# 依赖：output/timeDelays/timeDelays-target.csv（来自 4.compute-time-delays.sh）

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/quick-slow-analysis.py

mode=EEG
python $script -m $mode -s S01
python $script -m $mode -s S02
python $script -m $mode -s S03
python $script -m $mode -s S04
python $script -m $mode -s S05
python $script -m $mode -s S06
python $script -m $mode -s S07
python $script -m $mode -s S08
python $script -m $mode -s S09
python $script -m $mode -s S10

mode=MEG
python $script -m $mode -s S01
python $script -m $mode -s S02
python $script -m $mode -s S03
python $script -m $mode -s S04
python $script -m $mode -s S05
python $script -m $mode -s S06
python $script -m $mode -s S07
python $script -m $mode -s S08
python $script -m $mode -s S09
python $script -m $mode -s S10

python ./python/check-quick-slow-analysis.py
