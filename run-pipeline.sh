#!/usr/bin/env zsh

# RSVP 目标识别 - 主线 A（时间进程视角）总控流水线
# 按 doc/results.docx 的图序组织。
#
# ============================================================================
# 本轮只需要重跑下面这几段（其余已跑完，见各自的注释）：
#
#   1. stage 6   滑动解码，补投影前那一半分数        -> 图 3 缺的对比数字
#   2. stage 10  quick / slow 分层，修幅度列舍入      -> 图 5 的幅度比较
#   3. stage 11  只重画群体图（逐被试结果没变）      -> 图 6 标题重叠的修正
#   4. stage 12  晚窗的残余按键对照（--min-rt 0.55） -> 图 7 的 late 结论
#   5. stage 13  全被试群体 z-map（新）              -> 图 4 缺的群体源图
#
# 依赖关系：13 只读 stage 8 的 stc，而 stage 8 不用重跑；12 的对照要 stage 10
# 的分组表，所以 10 排在 12 前面；11 的群体图读 stage 10 的峰数据。
# ============================================================================
#
#   图 1  ERP：按键伪迹掩盖目标响应             -> stage 1-5
#   图 2  10 Hz SSVEP 传感器水平质检            -> stage 9
#   图 3  滑动解码：0.3 s 成分承担解码          -> stage 6
#   图 4  溯源：投影前运动皮层，投影后 0.3 s 皮层源 -> stage 8 + 13
#   图 5  反应时分层：quick / slow 的目标响应    -> stage 10
#   图 6  峰结构：两个模态都有第二峰            -> stage 11
#   图 7  峰窗口解码：晚窗还认不认 target        -> stage 12
#
# 每个 stage 都可以单独跑（各自有编号脚本），本文件是按顺序的入口。
# 断点续跑：脚本级产物已存在时多数脚本会跳过或覆盖写，重跑是安全的。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

run_sh() {
    echo "============================="
    echo "STAGE $1: $2"
    echo "============================="
    ./"$1"
}

run_check() {
    echo "============================="
    echo "CHECK $1"
    echo "============================="
    shift
    python "$@"
}

# ---- 图 1：ERP 与按键伪迹（已跑完，不用重跑）----
# epochs、notched epochs、投影后的 epochs、反应时表、群体 ERP 图都在 output/ 里。
# 只有在原始数据或前处理参数变过之后才需要重跑这五段。
# run_sh 1.raw-to-epochs.sh "raw-to-epochs"
# run_sh 2.notch-epochs.sh "notch-epochs"
# run_sh 3.remove-keypress-artificial.sh "remove-keypress-artificial"
# run_sh 4.compute-time-delays.sh "compute-time-delays"
# run_sh 5.average-subjects-erp.sh "average-subjects-erp"

# ---- 图 2：10 Hz SSVEP 传感器水平质检（已跑完，不用重跑）----
# run_sh 9.ssvep-qc.sh "ssvep-qc (sensor level, events 1 & 2)"

# ---- 图 3：滑动解码（要重跑：投影前那一半分数没写进 csv）----
# 之前只跑了带 -a 的投影后版本，group-decode-summary.csv 因此只有一行。
# 这一段补投影前（不带 -a）的分数，再重新出群体曲线。
run_sh 6.sliding-decode.sh "sliding-decode (the run without RMA)"
run_check "group decoding curves" ./python/check-sliding-decode.py

# ---- 图 4：源估计（stage 8 已跑完，不用重跑）----
# A 组：10 Hz SSVEP 溯源；B 组：宽带 ERP 投影前后各一遍。
# stc 已经在 output/source-estimation/{MODE}-{S01..S10}/ 里，13 直接读它们。
# run_sh 8.source-estimation.sh "source-estimation (SSVEP + ERP before/after removal)"

# ---- 图 4：全被试群体 z-map（要跑，新）----
# 10 名被试标准化后平均 + 单样本检验，输出群体 z / t 图、峰时刻、峰 MNI、
# 带响应的 aparc 标签，以及顶点级 FDR 和标签级的置换检验。
# 三组条件：投影前 / 投影后 / 投影后 target 减 non-target；另加 SSVEP 功率图。
# 交互查看单被试图的旧入口（python/check-source-estimation.py）保留，
# 群体图用这个新 stage，不必再一张张看。
run_sh 13.group-source-map.sh "group-source-map (all subjects, z map)"

# ---- 图 5：按反应时间分层的目标响应（要重跑：幅度列被舍成 6 位小数）----
# 导出的 summary-rma.csv 里 gfp_peak_a_* / amp_* 是固定 6 位小数，MEG 全是 0、
# EEG 只剩一位有效数字，图 5 的幅度比较现在是空的。当前脚本已改成 4 位有效
# 数字（sig4 / r6），重跑这一段即可，分组本身不会变。
# 输入是投影后的 epochs-1-notch-removal-artificial-epo.fif，所以 quick / slow
# 的差异不是反应锁时的按键成分。
run_sh 10.quick-slow-analysis.sh "quick-slow-analysis (split target trials by RT)"

# ---- 图 6：目标响应的峰结构（逐被试结果没变，只重画群体图）----
# peak-structure-rma.csv 的数值不受本轮改动影响（改的只是 upsert 的容错），
# 所以 20 次逐被试计算不必再跑；需要重跑的只有群体汇总，它修好了标题重叠，
# 并且多算了 n_peak1_early（第一峰落在早期视觉成分上的被试数）。
# 要连逐被试一起重算时取消下面那段的注释。
# run_sh 11.peak-structure.sh "peak-structure (first vs second peak)"
run_check "group peak structure" ./python/check-peak-structure.py

# ---- 图 7：峰窗口解码（主结果已跑完，要跑的是晚窗对照）----
# 主结果（windows / transfer / rt，全部 target 试次）在
# output/peak-window-decode/ 里，逐段计算已在 12.peak-window-decode.sh 里注释掉。
# 要跑的是 --min-rt 0.55 的对照：只保留按键晚于 0.55 s 的 target 试次，
# 此时晚窗完全落在按键之前，用来排除"晚窗的 target 身份只是残余按键成分"。
run_sh 12.peak-window-decode.sh "peak-window-decode (late window control)"

# ---- 可选：主线 B（单试次性能），与图 1-7 无关 ----
# ./7.eegnetv4-decode.sh
