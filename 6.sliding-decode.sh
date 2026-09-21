#!/usr/bin/env zsh

# 文章图 3：滑动解码。逐被试在每个时间点上解 target vs non-target，
# 得到一条 AUC 曲线，看投影前后可区分的时刻与幅度怎么变。
#
# 两套分数，按有没有把按键成分投影掉分开：
#   scores.txt          投影前（epochs-1/2-notch-epo.fif）      <- 图 3 的"投影前"
#   scores-rma.txt      投影后（epochs-{1,2}-notch-removal-artificial-epo.fif）
# 脚本按 -a 开关选 epochs，输出文件名随之变。
#
# 注意：group-decode-summary.csv 目前只有投影后一行，说明之前只跑了 -a 那一半，
# 图 3 里"投影前 0.455 s / AUC 0.86-0.93"的数字没有 csv 支撑。要补的是**不带 -a**
# 的那一半；投影后的分数已经在 output/sliding-decode/*/scores-rma.txt 里，
# 除非解码脚本本身改过，否则不用再算一遍（下面的第二段循环已注释掉）。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/sliding-decode.py
decoding_method=LR
subjects=(S01 S02 S03 S04 S05 S06 S07 S08 S09 S10)

# ---- 投影前：要跑（图 3 缺的就是这一半）----
for mode in EEG MEG; do
    for subj in $subjects; do
        python $script --subject $subj --mode $mode \
            --decoding_method $decoding_method
    done
done

# ---- 投影后：分数已在 scores-rma.txt 里，要重算时取消注释 ----
# for mode in EEG MEG; do
#     for subj in $subjects; do
#         python $script --subject $subj --mode $mode -a \
#             --decoding_method $decoding_method
#     done
# done

python ./python/check-sliding-decode.py
