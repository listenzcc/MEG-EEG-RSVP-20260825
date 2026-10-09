#!/usr/bin/env zsh

# 投影前后 × quick / slow 的 2×2 溯源对照。
#
# 为什么要有这一段：主线 A 的核心主张是「0.4 s 的峰是按键伪迹，投影掉反应子空间
# 之后 0.3 s 的目标特异成分才浮现」。图 5 的 RT 效应（quick 比 slow 大）是在
# 投影**后**的 epochs 上算的。那么顺理成章的追问是：
#
#   投影前的那份 target 数据里，quick 和 slow 有没有源空间差别？如果有，
#   它落在哪——是不是就落在按键那一片（中央前回一带）？
#
# 这一段就是回答这个问题的对照：四个条件（投影前/后 × quick/slow）各自溯源，
# 三种差各出一份。
#
# ── 四个条件共用哪一个逆算子（这一段唯一的硬技术决定） ────────────────────
#
# 噪声协方差一律量在**投影前**的整个 target 条件上，也就是 epochs-1-notch-epo.fif。
# 四个条件因此共用同一个逆算子，任何两张图之差里都不含正则化程度的差别。
#
# 为什么不能反过来量在投影后的文件上——这不是口味问题，是实测结论：
# mne.compute_covariance 会自动把 epochs.info['projs'] 里的投影算子吃进协方差，
# 于是协方差本身就带上了这一个投影。白化时那两个维度被当作零方差丢掉，等价于
# 把「投影」这个操作也施加到了投影**前**的数据上。结果是 投影前 − 投影后 的
# 差值图恒等于 0，要对照的东西被白化自己抹掉了。
#   （合成数据实测：共用投影后协方差时差值图 |max| = 0.00；
#     共用投影前协方差时 |max| = 21.9，同一条件下各自用自己的协方差是 58.8。)
#
# 代价要说清楚，写作时别装作没有：投影后的数据带着 2 个投影算子，而协方差里
# 没有它们。mne 不检查这个不一致（逆解模块里根本没有这道检查），会静默算完。
# 因为被投掉的那两个维度上投影后的数据本来就没有能量，影响是有限的——
# 合成数据里这一项让投影后的图幅度变约 10%（|max| 413.8 → 458.2）。
# 换来的好处是「投影前」那一张与 stage 8 已有的图完全一致（同一个文件、
# 同一个协方差、同一个逆算子），文章里两张图可以并排放。
#
# ── tag 后缀 ──────────────────────────────────────────────────────────────
#
# 四个条件都加 --tag_suffix sharedcov，有两个作用：
#   1. 与 stage 15 的 ave-quick / ave-slow 不打架。stage 15 用的是投影后文件
#      自己的协方差，两者数值不同，不能互相覆盖。
#   2. 投影前和投影后是两个文件，-c 做跨文件差值时两边的 tag 必须一致，
#      所以同一个后缀要加在两个条件上。
#
# ── 三种差，别混 ──────────────────────────────────────────────────────────
#
#   B2 投影状态内的 RT 配对：quick − slow，投影前后各一次。
#      这是「RT 效应在投影前是什么样、投影后是什么样」。
#   B3 投影本身拿掉了什么：投影前 − 投影后，quick / slow 各一次。
#      正负号按直觉走，投影前减投影后，得到的**正**值就是被拿掉的那部分。
#
# 引读数之前的提醒（沿 stage 15）：
#   幅度：EEG quick/slow = 1.377（9/10，p=0.0098，稳健）；
#         MEG = 1.243（9/10，p=0.0098，平衡子集只剩 7/10、p=0.078，不稳健）。
#   潜伏期：三个读数都不显著，不要写成「quick 的峰更早」。
#
# 成本：A 段是 10 被试 × 2 模态 × 4 条件 = 80 次逆解，每次都要重算正演算子，
# MEG 的正演很慢，整段按小时计（比 stage 15 再翻一倍）。**强烈建议先只跑 MEG**，
# 确认分组和产物都对了再补 EEG。省时间的办法：modes 只留 MEG，或 subjects 只留
# 两三个人先跑通。
#
# 依赖：stage 2 与 stage 3 的两个 epochs 文件
#         output/epochs/{MODE}-{SUBJ}/epochs-1-notch-epo.fif                  （投影前，stage 2）
#         output/epochs/{MODE}-{SUBJ}/epochs-1-notch-removal-artificial-epo.fif（投影后，stage 3）
#       stage 10 的 output/quick-slow/{MODE}-{SUBJ}/quick-slow-trials-rma.csv
#       fsaverage 模板（mne 会自动取）
# 产出：output/source-estimation/{MODE}-{SUBJ}/{epochs}.ave-{quick,slow}-sharedcov.stc
#       output/group-source-map/group-{MODE}-{epochs}[-minus-{epochs}].ave-*-sharedcov-{z,t}.stc
#       output/group-source-map/group-{epochs}.*-{mean,diff}-*.png
#       output/group-source-map/group-source-summary.csv（追加，一行一个条件）
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

# 投影前 / 投影后
proj_efname=epochs-1-notch-epo.fif
removal_efname=epochs-1-notch-removal-artificial-epo.fif

# 四个条件共用的协方差来源，必须是投影前的这一份，理由见文件头
cov_ref=$proj_efname

# 四个条件共用的 tag 后缀，见文件头
suffix=sharedcov

# 标签×时间的置换检验，10 名被试共 1024 种符号翻转，抽满即穷举
n_perm=1024

# 渲染用的时刻：stage 11 的群体第一峰，两个模态不在同一时刻
peak_MEG=0.270
peak_EEG=0.350
# 按键所在的窗口，看「被拿掉的东西」长什么样要用它
keypress_time=0.450

# ---- A 段：逐被试四个条件的溯源 ----
# 顺序无所谓，但两类跑法不一样，别把 --cov_epochs_fname 加错了：
#   投影前 —— 不加 --cov_epochs_fname。它的「自然协方差」就是 cov_ref 本身，
#            也就是整个 target 条件，与投影后那份共享的是同一个协方差。
#   投影后 —— 必须加 --cov_epochs_fname $cov_ref，否则它量在自己身上，
#            四个条件就不是一个逆算子了。
for mode in "${modes[@]}"; do
    for subj in "${subjects[@]}"; do
        for rt_group in quick slow; do
            # 投影前
            python $src_script --subject $subj --mode $mode \
                --epochs_fname $proj_efname \
                --rt_group $rt_group --tag_suffix $suffix

            # 投影后，协方差仍然量在投影前的整个 target 条件上
            python $src_script --subject $subj --mode $mode \
                --epochs_fname $removal_efname \
                --rt_group $rt_group --cov_epochs_fname $cov_ref \
                --tag_suffix $suffix
        done
    done
done

# ---- B 段：群体统计 ----
for mode in "${modes[@]}"; do
    # B1. 四个条件各自的峰时刻与峰位置。
    #     这是后面所有解读的底数，先出它，别跳过。
    for efname in $proj_efname $removal_efname; do
        for rt_group in quick slow; do
            python $map_script -m $mode -e $efname \
                -t ave-$rt_group-$suffix --n-perm $n_perm
        done
    done

    # B2. 投影状态内的 RT 配对：quick 减 slow，投影前后各一次。
    #     两组是同一批被试的两种条件，对差值做单样本检验就是配对检验。
    for efname in $proj_efname $removal_efname; do
        python $map_script -m $mode -e $efname \
            -t ave-quick-$suffix --contrast-tag ave-slow-$suffix \
            --n-perm $n_perm
    done

    # B3. 投影拿掉了什么：投影前 减 投影后，跨文件，用 -c。
    #     两边的 tag 相同（同一个 $suffix），-c 才找得到另一边。
    for rt_group in quick slow; do
        python $map_script -m $mode \
            -e $proj_efname -t ave-$rt_group-$suffix \
            -c $removal_efname --n-perm $n_perm
    done
done

# ---- C 段：渲染 ----
# C1. 2×2 的四个条件放在同一时刻、共用色标。这是要放进文章的那一张：
#     行是投影状态，列是 RT 组，幅度可以直接横向看。
#     时刻取投影后的峰（stage 11 的群体第一峰），投影前在那一刻也有内容，
#     用同一个时刻是为了让四格之差是幅度差而不是时间差。
python $plot_script -m MEG -e $proj_efname $removal_efname \
    -t ave-quick-$suffix ave-slow-$suffix --times $peak_MEG --clim-shared
python $plot_script -m EEG -e $proj_efname $removal_efname \
    -t ave-quick-$suffix ave-slow-$suffix --times $peak_EEG --clim-shared

# C2. 投影拿掉了什么：投影前 减 投影后，一行一个 RT 组。
#     多给一个按键时刻，被拿掉的那一块应该在 0.45 s 附近最清楚。
python $plot_script -m MEG -e $proj_efname -c $removal_efname \
    -t ave-quick-$suffix ave-slow-$suffix \
    --times $peak_MEG $keypress_time --clim-shared
python $plot_script -m EEG -e $proj_efname -c $removal_efname \
    -t ave-quick-$suffix ave-slow-$suffix \
    --times $peak_EEG $keypress_time --clim-shared

# C3. 投影状态内的 RT 差值：quick 减 slow，投影前后各一张。
#     这两张并排看就是这一段要的结论——差值有没有换地方。
for efname in $proj_efname $removal_efname; do
    python $plot_script -e $efname \
        -t ave-quick-$suffix --contrast-tag ave-slow-$suffix
done

# 想换视角或半球：--views lat med dor ven、--hemi both。
# 只验证平均与选峰、不碰渲染器（本机也能跑，因为它只读 stc）：
#   python $plot_script -e $proj_efname $removal_efname \
#       -t ave-quick-$suffix ave-slow-$suffix --times $peak_MEG --dry-run
#
# 只想跑通流程、不跑满 10 名被试：把上面的 subjects 改成 (S02 S03)，
# 或者给 $src_script 那两行加上 --subject 之外的被试名手工跑。
