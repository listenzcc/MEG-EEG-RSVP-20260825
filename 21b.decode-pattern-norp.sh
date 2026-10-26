#!/usr/bin/env zsh

# 投影前的 target 对 non-target 地形图（--tag norp）。
#
# stage 21 已经把投影后（--tag rma）的 target 与 rt 两套空间侧算完了，正文的
# 图 9a/9b（逐时刻模式）与 9e/9f（一致性检验）、表 7–9 的 target 行都出自那里。
# 缺的是投影前那一份：同一批被试、同一套逐时间点分类器、同样的抽平与平滑，
# 唯一差别是 epochs 换成未投影的 epochs-{1,2}-notch-epo.fif。
#
# 为什么值得跑：整套论文的归因结论（3.2）是时间轴上的——0.4 s 峰被运动子空间
# 削到两成到三成。投影前后的地形图并排，把这条结论补上空间侧：
#   · 投影前 0.42–0.46 s（解码峰所在时刻）的判别拓扑若落在双侧中央区，
#     与运动执行的预期位置一致，就是归因结论在空间上的独立证据；
#   · 投影后同一时刻的拓扑若转到枕区、且早窗拓扑在 0.33–0.36 s 出现，
#     则与解码峰前移（图 5）互相印证。
# 注意：这不是新的统计检验，是同一结论的空间侧旁证。判据仍然是 3.2 的配对
# 比值，地形图只回答「被投掉的是哪一块地形」。
#
# 与 rma 版的可比性（不要改这几项）：
#   --balance     target 对比默认抽平（non-target 抽到与 target 等样本），
#                 与 rma 版一致，否则两侧拓扑不可直接对照。
#   --smooth-ms 40   同 rma；--t-min/-t-max/-step-ms 全部默认。
#   --seed 0     同 rma。
# 文件名按 tag 区分（patterns-target-norp.csv 等），不会覆盖 rma 的产物。
#
# 依赖：stage 1 / 2 的未投影 epochs（output/epochs/{MODE}-{SUBJ}/
#       epochs-1-notch-epo.fif 与 epochs-2-notch-epo.fif）
# 产物：output/decode-pattern/
#         {MODE}-{SUBJ}/patterns-{,unit-,diff-,norm-}target-norp.csv
#         {MODE}-{SUBJ}/channels-norp.csv、summary-target-norp.csv
#         group-decode-pattern-target-norp-{EEG,MEG}.png        时刻序列地形图
#         group-decode-pattern-windows-target-norp-{EEG,MEG}.png  四窗平均
#         group-decode-pattern-stability-target-norp-{EEG,MEG}.png 分裂半与强度
#         group-decode-pattern-target-norp-{EEG,MEG}.csv、-summary.csv
#         group-pattern-agreement-target-norp-{EEG,MEG}.png     一致长度 + 窗间余弦
#         group-pattern-agreement-target-norp-{EEG,MEG}.csv、group-pattern-window-cos-*.csv
#
# 成本：20 个被试 × 1 个对比，每个被试在每个时间点拟合一次 logistic regression
#       并算一次通道协方差。MEG 通道多，量级约为 stage 21 的一半。
#
# 拷回本机：把 output/decode-pattern 下 *norp* 的文件与 {MODE}-{SUBJ} 目录里
#           的 *norp.csv 一起拷回，本机只做读图与写稿，不重算。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/decode-pattern.py
check=./python/check-decode-pattern.py
agree=./python/check-pattern-agreement.py
subjects=(S01 S02 S03 S04 S05 S06 S07 S08 S09 S10)

# ---- 逐被试：target 对比、两模态、投影前 ----
for mode in EEG MEG; do
    for subj in $subjects; do
        python $script -m $mode -s $subj --analysis target --tag norp
    done
done

# ---- 群体：地形图时间序列、窗平均、稳定性 ----
python $check --analysis target --tag norp --kind pattern

# ---- 群体：一致性检验与窗间余弦 ----
python $agree --analysis target --tag norp

# ---- 可选 ----
# ① 平均差版（无分类器，两类平均直接相减），与 pattern 并排看分类器读的是不是平均差：
# python $check --analysis target --tag norp --kind diff
#
# ② quick / slow 的投影前版本。标签是纯反应时列表，与投影无关，故指向 rma 的标签文件：
# for mode in EEG MEG; do for subj in $subjects; do
#     python $script -m $mode -s $subj --analysis rt --tag norp --qs-tag rma
# done; done
# python $check --analysis rt --tag norp --kind pattern
# python $agree --analysis rt --tag norp
#
# ③ 只挑关键时刻出图（默认 12 个时刻），把投影前后同一时刻并排：
# python $check --analysis target --tag norp --kind pattern \
#     --times 0.20,0.30,0.36,0.42,0.46,0.50
