#!/usr/bin/env zsh

# S 级方案 §1（见 doc/projection-validation.md）：把投影拆成按键组与视觉组。
#
# 3.remove-keypress-artificial.sh 投的是 proj2 + proj3 两组投影器的并集：
#   proj2 <- 非目标平均（标准视觉响应子空间）
#   proj3 <- 按键平均  （运动相关子空间）
# 本脚本把它们拆开分别投，得到一个天然对照条件：
#   visual   只投 proj2        keypress 只投 proj3
# 两组用同一套 kwargs、同一批试次，成分数相同，差别只在有没有运动成分。
#
# 要读的东西（对照分析自动找这些文件，缺一个就跳过该被试并告警）：
#   epochs-{1,2,3}-notch-removal-visual-epo.fif        <- 本脚本 A 段生成
#   epochs-{1,2,3}-notch-removal-keypress-epo.fif      <- 本脚本 B 段生成
#   epochs-{1,2,3}-notch-epo.fif                       <- 已有，unprojected
#   epochs-{1,2,3}-notch-removal-artificial-epo.fif    <- 已有，当前 pipeline
# 最后一行如果不存在，先跑一遍 3.remove-keypress-artificial.sh。
#
# 产物：
#   output/epochs/{MODE}-{SUBJ}/epochs-*-notch-removal-{visual,keypress}-*.fif
#   output/epochs/{MODE}-{SUBJ}/proj-explained-var-{visual,keypress}.csv
#   output/projection-source-compare/{MODE}-gfp-comparison.png
#   output/projection-source-compare/{MODE}-group-summary.csv
#   output/projection-source-compare/{MODE}-per-subject.csv
#
# 怎么读结果（MODE-group-summary.csv 一行一个条件）：
#   peak_gfp_window 是 0.30-0.55 s 窗内的 GFP 峰。
#   visual 那行接近 none  -> 0.4 s 成分不在视觉子空间里
#   keypress 那行大幅下降 -> 拿掉它的是按键子空间，归因成立
#   visual 也一起下降     -> 投影是普遍的幅度压缩，贡献 1 的归因不成立，必须重写
# median_ratio_vs_none 与 wilcoxon_p_vs_one 是逐被试比值（对照 none，按被试配对）。
#
# 成本：A + B 两段各 2 模态 x 10 被试 = 40 次投影（只读 epochs 不做逆解，很快），
# 对照分析 2 次。原有的 -removal-artificial- 系列不受影响（both 的 tag 不变）。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/remove-keypress-artificial.py

# ---- A 段：只投视觉组（proj2）----
# 输出 tag 是 visual，覆盖不到 -removal-artificial- 系列。
mode=EEG
python $script --subject S01 --mode $mode --proj-source visual
python $script --subject S02 --mode $mode --proj-source visual
python $script --subject S03 --mode $mode --proj-source visual
python $script --subject S04 --mode $mode --proj-source visual
python $script --subject S05 --mode $mode --proj-source visual
python $script --subject S06 --mode $mode --proj-source visual
python $script --subject S07 --mode $mode --proj-source visual
python $script --subject S08 --mode $mode --proj-source visual
python $script --subject S09 --mode $mode --proj-source visual
python $script --subject S10 --mode $mode --proj-source visual

mode=MEG
python $script --subject S01 --mode $mode --proj-source visual
python $script --subject S02 --mode $mode --proj-source visual
python $script --subject S03 --mode $mode --proj-source visual
python $script --subject S04 --mode $mode --proj-source visual
python $script --subject S05 --mode $mode --proj-source visual
python $script --subject S06 --mode $mode --proj-source visual
python $script --subject S07 --mode $mode --proj-source visual
python $script --subject S08 --mode $mode --proj-source visual
python $script --subject S09 --mode $mode --proj-source visual
python $script --subject S10 --mode $mode --proj-source visual

# ---- B 段：只投按键组（proj3）----
mode=EEG
python $script --subject S01 --mode $mode --proj-source keypress
python $script --subject S02 --mode $mode --proj-source keypress
python $script --subject S03 --mode $mode --proj-source keypress
python $script --subject S04 --mode $mode --proj-source keypress
python $script --subject S05 --mode $mode --proj-source keypress
python $script --subject S06 --mode $mode --proj-source keypress
python $script --subject S07 --mode $mode --proj-source keypress
python $script --subject S08 --mode $mode --proj-source keypress
python $script --subject S09 --mode $mode --proj-source keypress
python $script --subject S10 --mode $mode --proj-source keypress

mode=MEG
python $script --subject S01 --mode $mode --proj-source keypress
python $script --subject S02 --mode $mode --proj-source keypress
python $script --subject S03 --mode $mode --proj-source keypress
python $script --subject S04 --mode $mode --proj-source keypress
python $script --subject S05 --mode $mode --proj-source keypress
python $script --subject S06 --mode $mode --proj-source keypress
python $script --subject S07 --mode $mode --proj-source keypress
python $script --subject S08 --mode $mode --proj-source keypress
python $script --subject S09 --mode $mode --proj-source keypress
python $script --subject S10 --mode $mode --proj-source keypress

# ---- C 段：四条件对照 ----
# 读 evt=1 的四个条件，出群体 GFP 曲线、逐被试比值与群体汇总表。
# 只想换事件时加 --evt 2 或 --evt 3；只想换测峰的窗加 --peak-window 0.35,0.50。
python ./python/projection-source-compare.py --mode EEG
python ./python/projection-source-compare.py --mode MEG
