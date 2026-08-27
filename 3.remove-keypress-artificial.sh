#!/usr/bin/env zsh

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/remove-keypress-artificial.py

mode=EEG
python $script --subject S01 --mode $mode
python $script --subject S02 --mode $mode
python $script --subject S03 --mode $mode
python $script --subject S04 --mode $mode
python $script --subject S05 --mode $mode
python $script --subject S06 --mode $mode
python $script --subject S07 --mode $mode
python $script --subject S08 --mode $mode
python $script --subject S09 --mode $mode
python $script --subject S10 --mode $mode

mode=MEG
python $script --subject S01 --mode $mode
python $script --subject S02 --mode $mode
python $script --subject S03 --mode $mode
python $script --subject S04 --mode $mode
python $script --subject S05 --mode $mode
python $script --subject S06 --mode $mode
python $script --subject S07 --mode $mode
python $script --subject S08 --mode $mode
python $script --subject S09 --mode $mode
python $script --subject S10 --mode $mode
