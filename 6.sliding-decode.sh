#!/usr/bin/env zsh

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/sliding-decode.py
decoding_method=LR
argA=

mode=EEG
python $script --subject S01 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S02 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S03 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S04 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S05 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S06 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S07 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S08 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S09 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S10 --mode $mode $argA --decoding_method $decoding_method

mode=MEG
python $script --subject S01 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S02 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S03 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S04 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S05 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S06 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S07 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S08 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S09 --mode $mode $argA --decoding_method $decoding_method
python $script --subject S10 --mode $mode $argA --decoding_method $decoding_method

argA=-a