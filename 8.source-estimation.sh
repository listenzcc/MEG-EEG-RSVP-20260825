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
epochs_fnames=(epochs-1-notch-epo.fif epochs-1-notch-removal-artificial-epo.fif epochs-2-epo.fif epochs-3-notch-epo.fif)

for mode in "${modes[@]}"; do
    for subj in "${subjects[@]}"; do
        for efname in "${epochs_fnames[@]}"; do
            python $script --subj $subj --mode $mode --epochs_fname $efname
        done
    done
done