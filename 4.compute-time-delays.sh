#!/usr/bin/env zsh

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/compute-time-delays.py

python $script 


