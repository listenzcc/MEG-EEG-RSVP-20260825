#!/usr/bin/env zsh

# 文章新图：两条滑窗解码的空间侧。
#
# stage 6（target 对 non-target）与 stage 19（quick 对 slow）各自只给出一条
# AUC 随时间的曲线，曲线不告诉信息长在头上哪个位置。本 stage 用与它们相同的
# 逐时间点分类器，但把分类器背后的空间模式留下来，于是同一条时间轴可以读成
# 一串地形图。
#
# 每个时间点出两样东西：
#   pattern   分类器权重的 Haufe 变换 a = cov(X) @ (w / sd)。权重只说分类器听
#             哪些通道，乘上通道协方差之后才是"承载判别活动的拓扑"，这才是
#             地形图该画的东西。
#   diff      两类平均的差，无分类器，单变量。与 pattern 并排存，便于对照：
#             两者一致的地方说明分类器读的就是平均差，不一致的地方要另找解释。
#
# 读图的三条限制（写进图注）：
#   1. 每个时间点单独归一化到单位长度，所以颜色只表达拓扑形状，不表达强度。
#      强度看 norm 列与本 stage 的稳定性图，AUC 看 stage 6 / 19 的曲线。
#   2. pattern 是逐时间点拟合的，默认沿时间做 40 ms 滑动平均（--smooth-ms
#      可调，0 为不平滑）。解码的 ERP 成分本身就是几十 ms 宽，这一步不损失
#      真实的时间分辨率。
#   3. 逐时间点估计的被试间一致性由 split-half 相关给出，图上每个时刻下方
#      标注；一致性低的时刻，地形图不宜单独解读。`check-pattern-agreement.py`
#      把这条一致性做成检验：逐时刻算被试间平均拓扑的长度，用逐被试翻转符号
#      造零分布（随机符号下这个长度落在 1/sqrt(N) 而不是 0），沿线做簇置换。
#      它同时给窗与窗之间的分裂半余弦——同一批被试的平均会共享噪声、把余弦
#      抬向 1，分半之后才看得出两个窗是不是同一个形状（负值＝翻号）。
#
# 依赖：stage 3 的投影后 epochs（epochs-{1,2}-notch-removal-artificial-epo.fif）
#       与 stage 10 的 quick-slow-trials-rma.csv
# 产物：output/decode-pattern/{MODE}-{SUBJ}/
#         patterns-{target,rt}-rma.csv         逐时刻的 Haufe pattern（幅度版）
#         patterns-unit-*.csv                  同上，逐时刻单位长度（画图的版本）
#         diff-*.csv                           两类平均差
#         norm-*.csv                           pattern 的模，强度读数
#         channels-rma.csv                     通道名与类型
#         summary-*.csv                        该被试的一行读数
#       output/decode-pattern/
#         group-decode-pattern-{target,rt}-rma-{EEG,MEG}.png        时刻序列地形图
#         group-decode-pattern-windows-*-{EEG,MEG}.png              四个窗的平均
#         group-decode-pattern-stability-*-{EEG,MEG}.png            split-half 与强度
#         group-decode-pattern-{target,rt}-rma-{EEG,MEG}.csv        群体 pattern
#         group-decode-pattern-{target,rt}-rma-summary.csv          窗读数
#       --kind diff 另出一套同名文件，画的是平均差而不是 pattern。
#       check-pattern-agreement.py 另出：
#         group-pattern-agreement-{analysis}-rma-{MODE}.png        一致性 + 窗间余弦
#         group-pattern-agreement-*.csv                            逐时刻一致性、簇、p
#         group-pattern-window-cos-*.csv                           四窗之间的分裂半余弦
#
# 成本：每个被试在每个时间点拟合一次 logistic regression，并算一次通道协方差。
#       MEG 通道多，20 被试 x 2 对比的量级与 stage 6 的滑窗交叉验证相当。

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

# ---- 逐被试：两个对比、两个模态 ----
for mode in EEG MEG; do
    for subj in $subjects; do
        for analysis in target rt; do
            python $script -m $mode -s $subj --analysis $analysis --tag rma
        done
    done
done

# ---- 群体：地形图时间序列、窗平均、稳定性 ----
for analysis in target rt; do
    for kind in pattern diff; do
        python $check --analysis $analysis --tag rma --kind $kind
    done
done

# ---- 群体：一致性检验与窗间余弦 ----
for analysis in target rt; do
    python $agree --analysis $analysis --tag rma
done

# ---- 可选 ----
# ① 投影前的 target 对比（与投影后的地形图直接对照，看投影改了哪里）：
# for mode in EEG MEG; do for subj in $subjects; do
#     python $script -m $mode -s $subj --analysis target --tag norp
# done; done
# python $check --analysis target --tag norp --kind pattern
#
# ② 不平滑，看逐点估计的原始噪声水平（与默认 40 ms 平滑并排看）：
# python $script -m MEG -s S01 --analysis target --tag rma --smooth-ms 0 \
#     -o output/decode-pattern-nosmooth
# python $check --analysis target --tag rma --kind pattern \
#     --input-dir output/decode-pattern-nosmooth
#
# ③ 换时刻（默认 -0.10 到 0.80 s 共 14 个）与换窗：
# python $check --analysis rt --tag rma --kind pattern \
#     --times 0.05,0.10,0.15,0.20,0.25,0.30,0.35
# python $check --analysis rt --tag rma --kind pattern --win-early 0.20 0.32
#
# ④ 只读一个模态，或换窗（窗同时决定簇检验之外的窗读数）：
# python $agree --analysis rt --tag rma --modes EEG
# python $agree --analysis rt --tag rma --win-early 0.20 0.32
#
# ⑤ target 对比改为不抽平（non-target 试次远多于 target，默认抽平）：
# python $script -m EEG -s S01 --analysis target --no-balance \
#     -o output/decode-pattern-full
