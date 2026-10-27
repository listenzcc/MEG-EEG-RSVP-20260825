#!/usr/bin/env zsh
# ---------------------------------------------------------------------------
# stage 22：target ERP 与 target - non-target 差异波，投影前后的对照图
#
# 为什么要新做一套：
#   stage 5 的图（output/ERP/{MODE}/all-{evt}-*.png）用 MNE plot_joint，
#   拿来做「投影前后的比较」有四个问题——
#     1. 地形图帧时刻按 evt 各自选：target 硬编码 5 帧
#        [0.15, 0.22, 0.28, 0.35, 0.46]，non-target 与 keypress 走 'peaks'
#        自动选峰（实际落在 0.255 / 0.88 / 1.465 s），三者不可比；
#     2. 下排是蝴蝶图（全部通道叠画），没有全局场功率，读不出成分幅度；
#     3. 只有 5 帧，N200 / P300 时段的演化看不到；
#     4. 投影前后分两张图、各自缩放色标，目测不出削减了多少。
#
# 本 stage 固定时间网格（默认 0.10–0.58 s，步长 0.08，7 帧）、
# 投影前与投影后共用同一色标、下排换成 GFP 与高负荷通道波形。
#
# 依赖：stage 5 的 ave.fif（output/ERP/{MODE}/all-{1,2}-{suffix}-ave.ave.fif）；
#       若缺，脚本自动从 output/epochs 的逐被试 epochs 按试次数加权重算。
#       不需要 stage 3 的重投影，耗时以读文件为主。
#
# 产物：
#   output/ERP/EEG/erp-contrast-target-EEG.png    target ERP 本身
#   output/ERP/EEG/erp-contrast-diff-EEG.png      target - non-target
#   output/ERP/MEG/erp-contrast-target-MEG.png
#   output/ERP/MEG/erp-contrast-diff-MEG.png
# 日志里逐帧打印 GFP 与 max|ch|，写稿直接引用。
#
# 口径提醒：
#   - 带通 0.1–5 Hz、基线 (-0.2, 0) s，沿用 stage 5，保证与已有图可比。
#     10 Hz SSVEP 已被低通滤掉，看到的是慢成分的包络。
#   - 本范式是 10 Hz 无间隔呈现，每张图只有 100 ms 曝光，
#     经典 oddball P300 的前提（长间隔、低概率偏差刺激）不成立。
#     差异波里 0.25–0.45 s 的成分只作描述，不冠 P300 / N200 的名字。
#   - 0.4 s 之后的差异波落在按键窗内，按 3.2 的归因口径读。
#
# 用法：
#   zsh 22.erp-contrast.sh                      # 两模态、两张图都出
#   python python/erp-contrast.py --mode EEG --analysis diff \
#       --times-start 0.08 --times-stop 0.56 --times-step 0.06
#   python python/erp-contrast.py --mode MEG --analysis both --no-sensors
# ---------------------------------------------------------------------------

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/erp-contrast.py

# EEG：62 通道，传感器点可以画
python $script --mode EEG --analysis both

# MEG：272 条磁强计，传感器点太密，关掉
python $script --mode MEG --analysis both --no-sensors

echo -----------------------------
echo done. figures in output/ERP/{EEG,MEG}/erp-contrast-*.png
