#!/usr/bin/env zsh

# RSVP 目标识别 - 主线 A（时间进程视角）总控流水线
# 按 doc/results.docx 的图序组织，重跑一遍即可复现全部结果图。
#
#   图 1  ERP：按键伪迹掩盖目标响应      -> stage 1-4
#   图 2  10 Hz SSVEP 传感器水平质检     -> stage 5
#   图 3  滑动解码：0.3 s 成分承担解码    -> stage 6
#   图 4  溯源：投影前运动皮层，投影后 0.3 s 皮层源 -> stage 7-8
#   图 5  反应时分层：quick / slow 的目标响应        -> stage 10
#   图 6  峰结构：MEG 双峰 vs EEG 单峰               -> stage 11
#   图 7  峰窗口解码：晚窗还认不认 target            -> stage 12
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

# ---- 图 1：ERP 与按键伪迹 ----
# 1. 原始数据 -> epochs（-0.5 ~ 1.5 s，200 Hz，1/2/3 三类事件）
run_sh 1.raw-to-epochs.sh "raw-to-epochs"

# 2. 10 / 20 Hz notch，得到 ERP 用的 notched epochs
run_sh 2.notch-epochs.sh "notch-epochs"

# 3. 按键伪迹投影：用非目标(2)和按键(3)的平均构建 SSP，
#    同时输出 target-non-target 差分 ERP（投影前后各一张）
run_sh 3.remove-keypress-artificial.sh "remove-keypress-artificial"

# 4. 行为反应时表（stage 5 的 quick/slow 分组要用；只看 'all' 也要跑）
run_sh 4.compute-time-delays.sh "compute-time-delays"

# 5. 跨被试 ERP 群体平均：all-{1,2,3}-notch[-removal-artificial]-ave.png
run_sh 5.average-subjects-erp.sh "average-subjects-erp"

# ---- 图 2：10 Hz SSVEP 传感器水平质检 ----
# 只读 epochs-{1,2}-epo.fif（notch 之前），不依赖 fsaverage。
# 输出 output/ssvep-qc/{MODE}-{SUBJ}/*.ssvep10-qc.png 和 qc-summary.csv
run_sh 9.ssvep-qc.sh "ssvep-qc (sensor level, events 1 & 2)"

# ---- 图 3：滑动解码 ----
# 逐被试解码（10 折 x 401 时点，投影前后两套 scores）
run_sh 6.sliding-decode.sh "sliding-decode"

# 跨被试群体曲线 + 峰值时间表
echo "============================="
echo "STAGE check: group decoding curves"
echo "============================="
python ./python/check-sliding-decode.py

# ---- 图 4：源估计 ----
# A 组：10 Hz SSVEP 溯源（epochs-1/2，notch 之前，--cov_band control）
# B 组：宽带 ERP 溯源（按键投影前后各一遍）
run_sh 8.source-estimation.sh "source-estimation (SSVEP + ERP before/after removal)"

# 群体 z-map 交互查看；加 --png 可免交互出图
# 例：python python/check-source-estimation.py -m MEG -t ave -e epochs-1-notch-epo.fif
#     python python/check-source-estimation.py -m MEG -t ave -e epochs-1-notch-removal-artificial-epo.fif
echo "============================="
echo "STAGE check: source maps (interactive, run per condition)"
echo "============================="
echo "python python/check-source-estimation.py -m MEG -t ave -e epochs-1-notch-epo.fif"
echo "python python/check-source-estimation.py -m MEG -t ave -e epochs-1-notch-removal-artificial-epo.fif"

# ---- 图 5：按反应时间分层的目标响应 ----
# 输入是投影后的 epochs-1-notch-removal-artificial-epo.fif，
# 所以 quick / slow 的差异不是反应锁时的按键成分。
# 每名被试先按自己的 RT 中位数分，分不开就退到 0.4 s 固定阈值，
# 再不行就用最快/最慢各三分之一。实际用的规则记在 summary-rma.csv。
run_sh 10.quick-slow-analysis.sh "quick-slow-analysis (split target trials by RT)"

# ---- 图 6：目标响应的峰结构 ----
# 投影后的 target 响应在 MEG 里是两个峰（约 0.29 s 与 0.47 s，中间 0.41 s 有谷），
# 在 EEG 里是一个峰。quick-slow-analysis.py 只取窗内 argmax，会给一部分被试报
# 第一个峰、另一部分报第二个峰，MEG 的配对检验因此失去检验力。
run_sh 11.peak-structure.sh "peak-structure (first vs second peak)"

# ---- 图 7：在峰窗口上解码 ----
# windows：早窗 / 晚窗 / baseline 各自解 target vs non-target；
# transfer：早窗训练晚窗测试（及反向）；
# rt：target 试次内部解 quick vs slow，0.10-0.20 s 窗口早于任何真实按键。
run_sh 12.peak-window-decode.sh "peak-window-decode (windows / transfer / rt)"

# ---- 可选：主线 B（单试次性能），与图 1-7 无关 ----
# ./7.eegnetv4-decode.sh
