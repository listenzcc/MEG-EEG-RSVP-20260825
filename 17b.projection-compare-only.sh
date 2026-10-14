#!/usr/bin/env zsh

# 只重跑 17.projection-split.sh 的 C 段（四条件对照），不重做投影。
#
# 什么时候用这个：visual / keypress 两套 epochs 已经在 output/epochs/ 里（stage 17 的
# A、B 段跑过了），只是想用升级后的对照脚本重算读数。脚本本身是自包含的，只读
# epochs，不做逆解，也不碰 .fif 之外的东西。
#
# 依赖（每个被试每个模态四套，缺任何一套该被试会被跳过并告警）：
#   output/epochs/{MODE}-{SUBJ}/epochs-1-notch-epo.fif                     <- stage 2
#   output/epochs/{MODE}-{SUBJ}/epochs-1-notch-removal-visual-epo.fif      <- 17 A 段
#   output/epochs/{MODE}-{SUBJ}/epochs-1-notch-removal-keypress-epo.fif    <- 17 B 段
#   output/epochs/{MODE}-{SUBJ}/epochs-1-notch-removal-artificial-epo.fif  <- stage 3
#
# 本次重跑比上一版多出来的东西：
#   1. gfp_at_none_peak 列   固定潜伏期读数：每个条件都在该被试「未投影」窗内峰
#                            的时刻取 GFP。窗内最大比值把「拿掉了多少」和「残峰
#                            跑哪去了」混在一起——keypress 条件下窗内最大常常落在
#                            幸存的早期分量上（MEG 0.30 s），量的是早期分量而非
#                            0.4 s 峰。固定潜伏期读数把削减本身单独隔离出来。
#   2. ratio_at_none_peak_{tag}_vs_none 列  上面的逐被试比值。
#   3. {MODE}-paired-specificity.csv        两行（window-max / at-none-peak），
#                                            keypress 与 visual 的逐被试配对检验。
#
# 怎么读结果：
#   {MODE}-group-summary.csv 一行一个条件。要对比的关键是两个中位数：
#     median_ratio_vs_none          窗内最大比值
#     median_ratio_at_none_peak     固定潜伏期比值
#   预期 keypress 两个读数都远低于 visual，且固定潜伏期那一对比值差距更大。
#   {MODE}-paired-specificity.csv 给出配对检验（MEG 9/10、EEG 8/10 是上一版读数）。
#   {MODE}-gfp-comparison.png     四条 GFP 曲线 + 逐被试比值散点。
#
# 需要拷回本机的（本机是结果工作副本，只带 png/csv）：
#   output/projection-source-compare/{MODE}-group-summary.csv
#   output/projection-source-compare/{MODE}-per-subject.csv
#   output/projection-source-compare/{MODE}-paired-specificity.csv   <- 新增
#   output/projection-source-compare/{MODE}-gfp-comparison.png
# {MODE}-group-{tag}-ave.fif 不用拷（每个 1-2 MB，png/csv 已经概括了）。
#
# 成本：每个模态读 40 套 epochs（4 条件 x 10 被试）+ 滤波 + 出图，几分钟。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/projection-source-compare.py

echo -----------------------------
echo check the input epochs of every subject
for mode in EEG MEG; do
    for subj in S01 S02 S03 S04 S05 S06 S07 S08 S09 S10; do
        n=$(ls output/epochs/${mode}-${subj}/epochs-1-{notch,notch-removal-visual,notch-removal-keypress,notch-removal-artificial}-epo.fif 2>/dev/null | wc -l)
        if [ "$n" -ne 4 ]; then
            echo "  ${mode}-${subj}: only $n of 4 epochs present"
        fi
    done
done

echo -----------------------------
echo "compare (evt=1, peak window 0.30-0.55 s)"
python $script --mode EEG
python $script --mode MEG

echo -----------------------------
echo done, the readings are in output/projection-source-compare/
ls -la output/projection-source-compare/

# ---- 可选 ----
# 换事件（evt=3 是按键平均，本身就是按键响应，可当方法阳性对照）：
# python $script --mode MEG --evt 3
#
# 换读峰的窗（默认 0.30-0.55 覆盖投影前的 0.4 s 峰）：
# python $script --mode MEG --peak-window 0.25,0.45
