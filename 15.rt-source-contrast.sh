#!/usr/bin/env zsh

# 文章图 5 的溯源版本：quick / slow 两组目标试次的源空间对比。
#
# 三段，顺序不能换：
#   A. 逐被试溯源（stage 8 的扩展）——用 --rt_group 把 target 试次按反应时切成
#      quick / slow 两组，各自平均后走同一套逆解，输出 ave-quick / ave-slow 两个
#      stc。分组来自 stage 10 写好的 quick-slow-trials-rma.csv，所以这里用的分组和
#      传感器水平的幅度比较是**同一个分组**，不会因为重新分组而对不上。
#      噪声协方差仍然量在完整的 target 条件上，两组共用同一个逆算子——若各组用
#      自己的协方差，两组之间差的就不只是数据，还有正则化程度的差别。
#   B. 群体统计（stage 13 的扩展）——先各出一份 quick、slow 的群体 z 图，再用
#      --contrast-tag 出 quick 减 slow 的配对差图。两组是同一批被试的两种条件，
#      所以对差值做单样本检验就是配对检验，这是这里正确的检验。
#   C. 渲染（stage 14 的扩展）——把两张图渲染出来。关键在 --times：默认每张图各
#      自取自己的峰时刻，那样两个 panel 的差别里混着时间差；对比用的这一版必须
#      传同一个时刻，读数是 stage 11 的群体第一峰（MEG 0.270 s / EEG 0.350 s）。
#
# 与传感器水平结论的对应关系（写作时别记错）：
#   幅度：EEG quick/slow = 1.377（9/10，p=0.0098，稳健）；
#         MEG = 1.243（9/10，p=0.0098，剔除试次不平衡者后 5/6、p=0.156，不稳健）。
#         源空间这一步要回答的是「这个幅度差有没有一个可定位的皮层对应物」，
#         而不是重新检验幅度效应。
#   潜伏期：三个读数都不显著（EEG +26 ms、MEG +38 ms，p=0.275）。所以源空间
#         不要写成「quick 的峰更早」。
#
# 成本：A 段是 10 被试 × 2 模态 × 2 组 = 40 次逆解，每次都会重算正演算子与协方差，
# MEG 的正演比较慢，整段按小时计。可以拆开跑（比如先只跑 MEG）。
#
# 依赖：stage 3 的 epochs-1-notch-removal-artificial-epo.fif（服务器上有）
#       stage 10 的 output/quick-slow/{MODE}-{SUBJ}/quick-slow-trials-rma.csv
# 产出：output/source-estimation/{MODE}-{SUBJ}/*.ave-quick.stc、*.ave-slow.stc
#       output/group-source-map/group-{MODE}-{epochs}.ave-quick[-minus-ave-slow]-*.{stc,csv,png}
#       output/group-source-map/group-{epochs}.ave-quick-and-ave-slow-mean-*.png
#
# 渲染需要 3D 后端（pip install pyvista pyvistaqt），无显示器用 xvfb-run -a 跑整段。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

src_script=./python/source-estimation.py
map_script=./python/group-source-map.py
plot_script=./python/group-source-plot.py

modes=(MEG EEG)
subjects=(S01 S02 S03 S04 S05 S06 S07 S08 S09 S10)

# 投影后的 epochs，两组都在它上面切分（投影过所以 quick/slow 的差别不是按键成分）
efname=epochs-1-notch-removal-artificial-epo.fif

# 标签×时间的置换检验，1024 种符号翻转
n_perm=1024

# ---- A 段：逐被试的 quick / slow 溯源 ----
# 要省时间就先把 modes 改成 (MEG)，或者只留几号被试。
for mode in "${modes[@]}"; do
    for subj in "${subjects[@]}"; do
        for rt_group in quick slow; do
            python $src_script --subject $subj --mode $mode \
                --epochs_fname $efname --rt_group $rt_group
        done
    done
done

# ---- B 段：群体统计 ----
# 顺序：先单条件（各自出峰时刻与峰位置），再配对差。
for mode in "${modes[@]}"; do
    # 单条件，用于报告「quick 组自己的峰在哪、slow 组自己的峰在哪」
    python $map_script -m $mode -e $efname -t ave-quick --n-perm $n_perm
    python $map_script -m $mode -e $efname -t ave-slow  --n-perm $n_perm

    # 配对差：quick 减 slow。标签级置换检验 + 顶点级 FDR 都会写到
    # group-source-summary.csv 的 contrast_tag=ave-slow 那一行。
    python $map_script -m $mode -e $efname \
        -t ave-quick --contrast-tag ave-slow --n-perm $n_perm
done

# ---- C 段：渲染 ----
# C1. 各自在自己的峰时刻，用于看每组自己的空间分布
python $plot_script -e $efname -t ave-quick
python $plot_script -e $efname -t ave-slow

# C2. 同一时刻的两组对比：这是要放进文章的那一版。
#     时刻取 stage 11 的群体第一峰，MEG 与 EEG 分开跑，因为同一个 --times 会
#     同时作用到两个模态，而两个模态的第一峰不在同一个时刻。
python $plot_script -m MEG -e $efname -t ave-quick ave-slow --times 0.270
python $plot_script -m EEG -e $efname -t ave-quick ave-slow --times 0.350

# C3. 差值脑图：quick 减 slow，色标强制以零为中心（差值看的是符号）
python $plot_script -e $efname -t ave-quick --contrast-tag ave-slow

# 想换视角或半球：--views lat med dor ven、--hemi both。
# 只想验证平均与选峰、不碰渲染器：python $plot_script -e $efname \
#     -t ave-quick ave-slow --dry-run
