#!/usr/bin/env zsh

# 新加的分析：把投影后（rma）的 target epochs 按 quick / slow 标签做 sliding
# decoding，逐个被试、逐个模态。两类都是 target 试次，non-target 不参与。
#
# 为什么单独开一个 stage：stage 12 的 --analysis rt 分支解的正是这个对比，但它把
# 逐折曲线写成 .txt 塞在 output/peak-window-decode/{MODE}-{SUBJ}/ 里，导出结果集
# 只带 png 和 csv，所以那些曲线拿不回来，也就没东西可以跨被试平均。本 stage 是
# 这个对比的专用版本：
#   - 曲线存成 csv，能过导出这一关；
#   - 分类器、折数、随机种子、通道选择都是参数；
#   - 每个被试的读数另存一行 summary.csv；
#   - 新增跨被试的群体时间曲线与统计（stage 12 只报一个峰值数字，没有检验）。
# peak-window-decode.py 一个字没改；默认参数与它一致，所以
#   --win-peak -0.2 0.8 能逐位复现它的 auc_peak 与 peak_t，可当一致性校验。
#
# 怎么读结果（曲线峰在按键时刻，别只看峰）：
#   投影去掉了按键伪迹里与反应锁时的部分，单试次的残余还在，而 quick 试次按得早、
#   slow 按得晚，所以读到运动准备或残余按键的分类器一样会高于 chance。
#   唯一不能用按键解释的是 0.10-0.20 s 这一段：本批所有被试 RT 的 5% 分位都
#   >= 0.211 s，这一段早于任何真实按键。结论只能挂在这一段上（auc_preresp），
#   0.41-0.55 s 的峰与按键时刻重合，不能单独当证据。
#
# 依赖：stage 3 的投影后 epochs（epochs-1-notch-removal-artificial-epo.fif）
#       与 stage 10 的 quick-slow-trials-rma.csv
# 产物：output/rt-sliding-decode/{MODE}-{SUBJ}-LR/
#         curves-rma.csv        逐折 + 均值的 AUC 时间曲线（csv，可导出）
#         summary.csv           该被试的读数（峰值、三个窗、RT 中位数、试次数）
#         trials-rma.csv        进入解码的试次（index / delay / group）
#         rt-sliding-rma.png    该被试的曲线
#       output/rt-sliding-decode/group-rt-sliding-{summary,windows,clusters}-rma.csv
#       output/rt-sliding-decode/group-rt-sliding-rma.png
#
# 成本：每个被试一次滑窗交叉验证（10 折 x 400 时间点 x 2 类），EEG 稍快，
# MEG 通道多一倍。20 个被试，可与 stage 6 同一量级。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/rt-sliding-decode.py
subjects=(S01 S02 S03 S04 S05 S06 S07 S08 S09 S10)

# ---- 主结果：每个被试、每个模态一条曲线 ----
for mode in EEG MEG; do
    for subj in $subjects; do
        python $script -m $mode -s $subj
    done
done

# ---- 群体汇总：跨被试曲线 + 窗内检验 + 簇置换检验 ----
# 一次把两个模态都读进来，不需要参数。
python ./python/check-rt-sliding-decode.py

# ---- 可选：把图 5 的对照做全 ----
# ① 分类器换成 SVC，看曲线形状是否依赖线性模型（写进 {MODE}-{S}-SVC/，不覆盖 LR）
# for mode in EEG MEG; do for subj in $subjects; do
#     python $script -m $mode -s $subj -d SVC
# done; done
# python ./python/check-rt-sliding-decode.py --decoding SVC
#
# ② 两组人数差得多的被试（MEG-S03 是 86 比 282）加 --balance 抽平再解一次，
#    看是不是类别不平衡带出来的差异。
# python $script -m MEG -s S03 --balance -o output/rt-sliding-decode-balanced
#
# ③ 对齐 stage 12 的峰值读数（它的搜索范围含刺激前基线）：
# python $script -m EEG -s S01 --win-peak -0.2 0.8
#
# ④ 早窗与晚窗的差异是否稳健：换窗重跑不影响 curves-rma.csv，
#    换窗只改 summary.csv 里的读数，群体那边用 --win-early / --win-late 对齐。
