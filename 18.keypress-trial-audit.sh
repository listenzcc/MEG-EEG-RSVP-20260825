#!/usr/bin/env zsh

# S 级方案 §2（见 doc/projection-validation.md）：数据里有没有「非目标试次的按键」。
#
# 只在原始 annotations 层面统计，不做 epoching、不做任何投影：
#   每个按键事件前面最近的一张图是 target 还是非目标？
#   每个 target 图后面有没有按键？
#
# 为什么值得跑：若存在足量的「非目标 + 按键」试次，它们构成一个完全不依赖投影的
# 内部对照——有按键、没有 target。目标特异成分不可能出现在那里，运动相关成分可以。
# 这一对照比任何投影前后比较都直接，因为它不动数据。
#
# 已有的线索（来自阶段 1 的产物）：EEG 按键事件有效平均 511 次、target 504 次
# （按键多 7 个），MEG 按键 406 次、target 481 次（少 75 个，像漏按）。
# 但那些数字来自 epochs（已经过事件重标注与 reject），本脚本给的是 raw 层面的真值。
#
# 产物：
#   output/keypress-trial-audit/{MODE}-{SUBJ}.csv    每个被试一行
#   output/keypress-trial-audit/summary-{MODE}.csv   汇总
#
# 怎么读结果（summary-{MODE}.csv 末段会打印各列总计）：
#   kp_with_target_in_window      按键前 0.15-0.8 s 窗口内存在 target 图的按键数
#   kp_without_target_in_window   窗口内没有任何 target 的按键数
#                                 -> 内部对照的可用试次数（保守下限）
#   target_without_keypress       漏按的 target 数 -> epochs-1 与 epochs-3 的差额来源
#   delay_median_to_target_ms     按键与窗内最近 target 的延迟，应与反应时中位数
#                                 同量级（EEG ~0.35 s、MEG ~0.41 s）
#   delay_median_to_nearest_image_ms 按键与最近一张图（不管类型）的延迟；10 Hz 下
#                                 这张图通常不是触发图，这个数应集中在 0-100 ms
#   若 kp_without_target_in_window 有几十个以上：可以做 B1 对照，见 doc/projection-validation.md §2
#   若接近 0：这条内部对照不可用，转而依赖 17.projection-split.sh 的拆投影对照
#
# 先跑一个被试确认事件标号对不对（日志里会打印 event_id 字典）：
#   python ./python/keypress-trial-audit.py --mode EEG --subject S01
# 若标号不是 1 / 2 / 3，用 --target-code / --nontarget-code / --keypress-code 指定。
#
# 成本：只读头信息（preload=False），每个被试几秒。EEG 与 MEG 各 10 个被试。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/keypress-trial-audit.py

# ---- EEG：逐被试统计 ----
mode=EEG
python $script --mode $mode --subject S01
python $script --mode $mode --subject S02
python $script --mode $mode --subject S03
python $script --mode $mode --subject S04
python $script --mode $mode --subject S05
python $script --mode $mode --subject S06
python $script --mode $mode --subject S07
python $script --mode $mode --subject S08
python $script --mode $mode --subject S09
python $script --mode $mode --subject S10
python $script --mode $mode --summarize

# ---- MEG：逐被试统计 ----
mode=MEG
python $script --mode $mode --subject S01
python $script --mode $mode --subject S02
python $script --mode $mode --subject S03
python $script --mode $mode --subject S04
python $script --mode $mode --subject S05
python $script --mode $mode --subject S06
python $script --mode $mode --subject S07
python $script --mode $mode --subject S08
python $script --mode $mode --subject S09
python $script --mode $mode --subject S10
python $script --mode $mode --summarize
