#!/usr/bin/env zsh

# 文章图 4 的群体版本：全部被试的群体 z-map。
#
# stage 8 把每名被试的 stc 都算在 fsaverage 模板上，所以 10 名被试的顶点
# 是对齐的，可以直接平均。这个 stage 做三件事：
#   1. 先按各自的基线把每名被试标准化（--normalize baseline，默认）。
#      MNE 逆解出来的幅度逐被试差很多，不标准化的话群体图就是某一名被试的图。
#   2. 平均成群体图，并对零做逐被试的单样本检验，得到 t 图和 p 图。
#   3. 报告群体图的峰时刻、峰的 MNI 坐标、以及哪些脑区（aparc 标签）带响应；
#      同时把群体图写回 stc，之后画 3D 图不必再翻单被试文件。
#
# 三组图，对应 stage 8 的 A / B 两组：
#   B 组（宽带 ERP）  投影前 / 投影后 / 投影后 target 减 non-target
#   A 组（10 Hz SSVEP）功率图，不加 --normalize，保留物理量
#
# 统计有两种，回答的问题不同，正文报的时候要说清楚是哪一种：
#   - 顶点级 FDR：保住空间细节，效应本身是局部的，这个读数最诚实；
#   - 标签×时间矩阵上的最大统计量置换检验（--n-perm）：把 300 万个
#     顶点时间点压到几千个测试，才谈得上校正。直接对顶点做最大统计量
#     是没用的，光是基线窗里的最大值就能到 |t| > 200，那是多重比较的
#     产物，不是响应。
#
# 依赖：stage 8 的 output/source-estimation/{MODE}-{SUBJ}/*.stc-lh.stc
#       （不需要重跑 stage 8，stc 已经在）
# 产出：output/group-source-map/group-{MODE}-{epochs}[-minus-{epochs}].{tag}-{z,t}.stc
#                                                  .{tag}-labels.csv
#                                                  .{tag}-summary.csv
#                                                  .{tag}.png
#       output/group-source-map/group-source-summary.csv（所有条件的汇总，一行一个条件）

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/group-source-map.py

# 10 名被试的名字是自动从 output/source-estimation/ 里扫出来的，不用传 -s。
# 希望只跑一部分被试时：python $script -m MEG -e ... -t ave -s S01 S02
subjects_arg=

# 标签×时间的置换检验，10 名被试共有 1024 种符号翻转，抽满即穷举
n_perm=1024

# ---- B 组：宽带 ERP，投影前后 ----
for mode in MEG EEG; do
    for efname in epochs-1-notch-epo.fif epochs-1-notch-removal-artificial-epo.fif; do
        python $script -m $mode -e $efname -t ave --n-perm $n_perm $subjects_arg
    done
done

# ---- B 组：投影后的 target 减 non-target ----
# 这是"0.3 s 的成分在额下回一带"这句话的原始出处：差分之后做的单样本检验，
# 检验的是 target 与 non-target 的差是否为 0，而不是响应是否为 0。
for mode in MEG EEG; do
    python $script -m $mode \
        -e epochs-1-notch-removal-artificial-epo.fif \
        -c epochs-2-notch-removal-artificial-epo.fif \
        -t ave --n-perm $n_perm $subjects_arg
done

# ---- A 组：10 Hz SSVEP 的源功率 ----
# 功率图是个正的量，基线里 10 Hz 刺激序列一直在放，所以不做基线标准化，
# 保留功率本身（--normalize none）。要换成相对基线的变化就删掉这个选项。
for mode in MEG EEG; do
    python $script -m $mode -e epochs-1-epo.fif -t ssvep10-power \
        --normalize none --n-perm $n_perm $subjects_arg
done

# 想把群体图直接渲成脑图（需要装好 3D 后端）就在上面任意一条加上 --brain，
# 例如：
#   python $script -m MEG -e epochs-1-notch-removal-artificial-epo.fif -t ave --brain
