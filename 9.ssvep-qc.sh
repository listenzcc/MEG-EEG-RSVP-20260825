#!/usr/bin/env zsh

# 文章图 2：10 Hz SSVEP 的传感器水平质量检查。
# 只读 epochs，不需要 fsaverage，在跑 8.source-estimation.sh 之前先过一遍。
# --all_events 会同时检查 epochs-1（target）和 epochs-2（non-target）：
# 成分由刺激序列驱动，两类里都应该存在，两张图正好记录这一点。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/ssvep-qc.py

mode=EEG
python $script --subject S01 --mode $mode --all_events
python $script --subject S02 --mode $mode --all_events
python $script --subject S03 --mode $mode --all_events
python $script --subject S04 --mode $mode --all_events
python $script --subject S05 --mode $mode --all_events
python $script --subject S06 --mode $mode --all_events
python $script --subject S07 --mode $mode --all_events
python $script --subject S08 --mode $mode --all_events
python $script --subject S09 --mode $mode --all_events
python $script --subject S10 --mode $mode --all_events

mode=MEG
python $script --subject S01 --mode $mode --all_events
python $script --subject S02 --mode $mode --all_events
python $script --subject S03 --mode $mode --all_events
python $script --subject S04 --mode $mode --all_events
python $script --subject S05 --mode $mode --all_events
python $script --subject S06 --mode $mode --all_events
python $script --subject S07 --mode $mode --all_events
python $script --subject S08 --mode $mode --all_events
python $script --subject S09 --mode $mode --all_events
python $script --subject S10 --mode $mode --all_events
