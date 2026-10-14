#!/usr/bin/env zsh

# stage 19 的三个对照：基线窗（状态/漂移）、1 Hz 高通（漂移）、--balance（类别不平衡）。
#
# 为什么需要它们：19 的主结果是 pre-motor 窗（0.10-0.20 s）的 AUC 0.52-0.53，很小，
# 而且曲线在 0.95-1.05 s 还有一段抬升（MEG 0.541，p=0.0039），后者按运动解释不成立。
# 在那段抬升被解释掉、pre-motor 读数被对照检验过之前，pre-motor 只能写成「弱且方向
# 一致」，不能归因给 target 加工。这三个对照就是去检验它。
#
# ---------------------------------------------------------------------------
# 对照 A：基线段（不需要重跑解码，一条命令）
# ---------------------------------------------------------------------------
# epochs 是 0.1-40 Hz、不带基线校正的，分类器在每个时刻看到的是绝对头皮分布，慢漂移
# 还在数据里。刺激前 [-0.2, -0.05] s 没有任何 target 相关的刺激，所以那里偏离 0.5 的
# 部分只能是漂移、觉醒度或试次顺序。这条窗的读数现在由 check-rt-sliding-decode.py
# 直接从已有的 curves-rma.csv 算出（列名 auc_baseline），不需要重新解码。
#
# 本机先用已有的 20 条曲线算过（2026-10-14）：
#   EEG  基线 0.5073（6/10 高于 0.5，p=0.28）   pre-motor 0.5274（p=0.014）
#   MEG  基线 0.4976（5/10，p=0.63）            pre-motor 0.5227（p=0.027）
#   EEG  更长的基线 [-0.45,-0.05] 0.5073       MEG 0.4976
# 基线在 chance，晚段不在——所以晚段的抬升不是「刺激前就存在的状态差」，
# 而是刺激之后才出现的东西。
#
# ---------------------------------------------------------------------------
# 对照 B：1 Hz 高通（漂移的正面处理）
# ---------------------------------------------------------------------------
# 不信任「基线在 chance」这条间接推断时用它：直接把 1 Hz 以下滤掉重解一遍。
# 若 pre-motor 的 0.52 在 1 Hz 高通后消失，那它来自慢漂移，不能归因给 target。
# 滤波器的实际参数：MNE 默认 FIR，200 Hz 采样下 1 Hz 高通是 661 点 = 3.3 s，
# 比 2 s 的 epoch 还长，MNE 用 reflect 填充补齐边界，实测各窗 RMS 保留 0.995-1.036，
# 边界（基线窗与尾部）多出 3-4% 的功率，读数时记得这一点。
#
# 对照的锋利度已在两个合成沙盒量化（scratch/rt-sliding-test/，2026-10-14）：
#   效应沙盒（0.25-0.45 s 真效应，无漂移）：hp1 后效应簇还在（0.275-0.310 /
#     0.320-0.340 / 0.365-0.435 三段，p 均 <0.05），前运动窗保持 chance
#     （0.5044，p=0.66）-> 高通不制造假阳性、也不杀死真效应；
#   漂移沙盒（0.3 Hz 组间漂移，无真效应）：base 前运动 0.840 -> hp1 后 0.539，
#     基线 0.743 -> 0.506。高于 chance 的部分被削 8.7 倍，但阻带泄漏留下
#     ~0.04 的均匀残差，簇检验仍会捡起来。**所以判读规则是比较而非绝对**：
#     真实数据 pre-motor 只有 0.527/0.523，且基线窗已在 chance（对照 A），
#     真实漂移的功率上限远低于这个卡通漂移，若 hp1 后 pre-motor 回到
#     ~0.50-0.51 即可排除漂移主导；若仍停在 0.53 附近则不能下结论。
#
# ---------------------------------------------------------------------------
# 对照 C：--balance（类别不平衡）
# ---------------------------------------------------------------------------
# stage 10 的规则大多给出大小接近的两组，但阈值切法会留下悬殊的一对：MEG-S03 是
# 86 比 282，MEG-S08 146 比 307。AUC 对先验不敏感，分类器不敏感不到那个程度。
# --balance 把大的那组随机抽到小的那组同样大小（种子与交叉验证一致，seed 0）。
# MEG-S03 的峰值 0.8531 是全批最高，值得单独确认它不是不平衡带出来的。
# 已在合成沙盒验证抽平动作本身：40/100 的标签表 -> 40/40（40 保留 40）；
# 两组本来相等时它是空转，输出与 base 逐位一致。
#
# ---------------------------------------------------------------------------
# 依赖：stage 3 的投影后 epochs + stage 10 的 quick-slow-trials-rma.csv
#       （都是 19 主结果已经用过的输入，不新增依赖）
# 产物：
#   output/rt-sliding-decode/            对照 A：group-* 表里多出 auc_baseline 列
#   output/rt-sliding-decode-hp1/        对照 B：curves-rma-hp1.csv 等 + group-* 表
#   output/rt-sliding-decode-balance/    对照 C：curves-rma-bal.csv 等 + group-* 表
#
# 注意 --qs-tag：标签文件属于 epochs，不属于打分方式。对照跑改了 --tag（只影响输出
# 文件名），所以要把 --qs-tag 指回 rma，否则找不到 quick-slow-trials-rma.csv。
#
# 需要拷回本机的（本机是结果工作副本，只带 png/csv）：
#   output/rt-sliding-decode/group-rt-sliding-{summary,windows,clusters}-rma.csv
#   output/rt-sliding-decode/group-rt-sliding-rma.png
#   output/rt-sliding-decode-hp1/group-rt-sliding-*-rma-hp1.csv
#   output/rt-sliding-decode-hp1/group-rt-sliding-rma-hp1.png
#   output/rt-sliding-decode-balance/group-rt-sliding-*-rma-bal.csv
#   output/rt-sliding-decode-balance/group-rt-sliding-rma-bal.png
#   （逐被试的 curves/summary 不用再拷，主结果那批已经在）
#
# 成本：对照 A 几秒（只读曲线）；B、C 各是一次完整的滑窗解码，与 19 主结果同一量级。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/rt-sliding-decode.py
check=./python/check-rt-sliding-decode.py
subjects=(S01 S02 S03 S04 S05 S06 S07 S08 S09 S10)

# ===========================================================================
# 对照 A：基线段（只重跑汇总，不重跑解码）
# ===========================================================================
echo -----------------------------
echo "control A: the pre stimulus baseline window (no decoding rerun)"
python $check

# ===========================================================================
# 对照 B：1 Hz 高通
# ===========================================================================
echo -----------------------------
echo "control B: refilter to 1 Hz high pass"
for mode in EEG MEG; do
    for subj in $subjects; do
        python $script -m $mode -s $subj \
            -t rma-hp1 --qs-tag rma --highpass 1 \
            -o output/rt-sliding-decode-hp1
    done
done
python $check -d output/rt-sliding-decode-hp1 -t rma-hp1

# ===========================================================================
# 对照 C：--balance
# ===========================================================================
echo -----------------------------
echo "control C: draw the larger group down to the smaller one"
for mode in EEG MEG; do
    for subj in $subjects; do
        python $script -m $mode -s $subj \
            -t rma-bal --qs-tag rma --balance \
            -o output/rt-sliding-decode-balance
    done
done
python $check -d output/rt-sliding-decode-balance -t rma-bal

echo -----------------------------
echo done
echo "A: output/rt-sliding-decode/group-rt-sliding-windows-rma.csv     <- auc_baseline 列"
echo "B: output/rt-sliding-decode-hp1/                                 <- 与 A 比 auc_pre-motor"
echo "C: output/rt-sliding-decode-balance/                             <- 与 A 比 auc_peak 与 auc_pre-motor"

# ---- 可选 ----
# 只跑不平衡的那几个被试（省时间，但群体表就只剩这几个被试）：
# python $script -m MEG -s S03 -t rma-bal --qs-tag rma --balance \
#     -o output/rt-sliding-decode-balance
# python $script -m MEG -s S08 -t rma-bal --qs-tag rma --balance \
#     -o output/rt-sliding-decode-balance
#
# 更强的高通（漂移越慢越该被拿走，代价是 2-6 Hz 的成分一起掉）：
# python $script -m MEG -s S01 -t rma-hp2 --qs-tag rma --highpass 2 \
#     -o output/rt-sliding-decode-hp2
#
# 对齐 stage 12 的峰值读数（它的搜索范围含刺激前基线）：
# python $script -m EEG -s S01 --win-peak -0.2 0.8
