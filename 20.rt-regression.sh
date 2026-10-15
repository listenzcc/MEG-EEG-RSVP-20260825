#!/usr/bin/env zsh

# 连续反应时回归：不再把试次切成 quick / slow 两组，而是把反应时当连续变量，
# 在每一个时间点上跨试次回归，同时把按键锁时的响应作为一组基函数放进同一个
# 设计矩阵。目的是把「与加工速度相关的成分」和「按键动作本身」在统计上分开。
#
# 为什么需要这个 stage：quick / slow 二分丢掉了反应时里的大部分信息，而且两组的
# 分界线落在实际 RT 分布的空窗里（MEG 十名被试最概然 RT 都是 0.4117 s，18/20 名
# 被试在 0.36-0.40 s 之间几乎没有试次），二分后的标签接近二值。回归用连续 RT，
# 不依赖这条分界线。
#
# 设计矩阵（每个时间点一列一套）:
#   const               截距，吃掉整个刺激锁时响应
#   rt                  中心化的反应时，本分析要估计的那一项
#   button_±0.xx        按键后 0-0.35 s 的分箱基函数，每箱的取值取决于该试次在
#                       这个时间点上的 lag = t - RT
# 关键性质：在最早一次按键之前，所有 button 列恒为 0，rt 列是干净的刺激锁时调制；
# 最早一次按键之后，button 列活过来，rt 的估计只在 RT 分布允许的程度内可分。
# 脚本把这条线算出来（clean_until_s），并分段报告可识别性。
#
# 可识别性是本 stage 的核心，不是附加检查：RT 是 target 到按键的间隔，RT 分布越窄，
# button 基函数越能复现 rt 列本身，两者描述同一份方差。诊断报三个数：
#   cond         整个设计能不能求逆
#   vif_rt       rt 列的标准误被放大多少倍
#   cca_rt_button rt 列与 button 基函数的典型相关，决定这个系数到底有没有意义
# 合成沙盒（scratch/rt-regression-test）已验证：
#   real 场景 clean 段 gain 1.00 / corr 0.96（真效应能恢复）
#   所有场景 clean 段 leak == null（按键响应不泄漏进最早按键之前）
#   mixed 段无偏（gain 1.00）但 corr 降到 0.56，vif 15.8、cca 0.968
#   RT 量化到 5 个值时 mixed 段 cond = inf，被判不可识别
#
# 怎么读结果：
#   1. 先看 audit-*.csv 的 clean_until_s 与 clean_identifiable / mixed_identifiable。
#      mixed 段几乎必然是 NOT identifiable，这是结论的一部分，不是脚本失败。
#   2. 唯一能写进论文的是 clean 段（t < clean_until_s）的 rt 系数曲线。
#   3. recovery_* 是每个被试自己的恢复检验：gain 接近 1 才说明这个被试的设计
#      能把真效应找回来；leak_over_null 接近 1 才说明 clean 段没有按键泄漏。
#
# 依赖：stage 3 的投影后 epochs（epochs-1-notch-removal-artificial-epo.fif）
#       与 stage 10 的 quick-slow-trials-rma.csv
# 产物：output/rt-regression/{MODE}-{SUBJ}/
#         audit-rma.csv            反应时分布 + 设计诊断 + 恢复检验
#         identifiability-rma.csv  若干探测时间点的 cond / vif / cca
#         beta-curve-rma.csv       rt 系数的时间曲线（投影到自身靶拓扑上）
#         beta-windows-rma.csv     三个预指定窗的读数
#         rt-regression-rma.png    该被试的系数与 t 值
#       output/rt-regression/group-rt-regression-{summary,windows,clusters}-rma.csv
#       output/rt-regression/group-rt-regression-rma.png
#
# 成本：每个被试一次逐时间点最小二乘（约 400 个时间点 x 12-306 个通道），
# 加上每组 10 次模拟恢复。20 个被试，比 stage 19 略重。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/rt-regression.py
subjects=(S01 S02 S03 S04 S05 S06 S07 S08 S09 S10)

# ---- 主结果：投影后 epochs 上的连续 RT 回归 ----
for mode in EEG MEG; do
    for subj in $subjects; do
        python $script -m $mode -s $subj --model epochs
    done
done

# ---- 群体汇总：跨被试曲线 + 窗内检验 + 簇置换 ----
python ./python/check-rt-regression.py

# ---- 可选：投影前的同一条回归，用来看投影带走的是哪一部分 ----
# 对比两套系数：若投影前 clean 段已有 RT 关联而投影后消失，说明那段关联依赖
# 与运动子空间共线的成分；若两段都在，说明它是独立于按键的加工速度效应。
# for mode in EEG MEG; do for subj in $subjects; do
#     python $script -m $mode -s $subj --model epochs -t norp \
#         --epochs-fname epochs-1-notch-epo.fif
# done; done
# python ./python/check-rt-regression.py --tag norp

# ---- 可选：连续 raw 上的 FIR 去卷积（--model continuous）----
# 不切 epoch，直接在拼接后的 raw 上同时估计 image / target / target×RT / button
# 四组有限冲激响应基。好处是没有 epoch 重叠问题，坏处是设计矩阵很大
# （数十万行），只能走正规方程，且读 raw 需要 stage 1 的 block_*_ica-raw.fif。
# python $script -m EEG -s S01 --model continuous
