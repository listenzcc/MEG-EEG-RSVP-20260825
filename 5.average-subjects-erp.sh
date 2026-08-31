#!/usr/bin/env zsh

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/average-subjects-erp.py

python $script --mode EEG

python $script --mode MEG

