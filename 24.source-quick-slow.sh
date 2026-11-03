#!/usr/bin/env zsh

# quick 与 slow 两组目标试次的源空间**差值**长图：一组差值占两行（俯视 + 后视），
# 从左到右 0-1 s，受试间平均。
#
# ── 这一支和 stage 15 / 23 的分工 ────────────────────────────────────────
#   15  rt-source-contrast：统计 + 单时刻渲染。标签×时间的符号翻转置换检验
#       （两模态都没有通过 FDR 的顶点）、quick / slow 各一张峰时刻图、差值 csv。
#   23  source-long-figure：四个条件各自一行，想看的是「各自长什么样」。
#   24  这一支：把 quick 与 slow **相减**，看差别落在哪一段、俯视与后视是不是同一个
#       故事。0-1 s 连续十一帧，是 23 缺的那一半。
#
# **图不替代统计**。哪一块差值是站得住的，仍然回 15 的置换检验读数；这里只画形态
# 与时间进程。正文里两者要分开放，别用图上的红块代替 p 值。
#
# ── 读的是什么 ────────────────────────────────────────────────────────────
# 默认读 stage 16 的 ave-quick-sharedcov / ave-slow-sharedcov：这两份量在投影前那一份
# 文件的协方差上，两组共用同一个逆算子，所以两组之差里不含正则化差异。换成 stage 15
# 的 ave-quick / ave-slow（各自协方差）会让差值混进一点正则化差别，只有明确要复现
# 那一路读数时才用。
#
# ── 归一化成百分比 ────────────────────────────────────────────────────────
# 默认 --normalize ref：差值除以该被试的参考幅度（两组 99 分位的均值，一个标量），
# 结果读作「相对本人整体激活水平的百分比」。逐顶点做分母会在 |q|+|s| 近零处爆掉，
# 所以分母取标量。屏幕上看到的 ±20 就是「快组比慢组高两成参考幅度」。
# 想看绝对量就 --normalize none --data-scale 1e15 --unit fAm。
#
# ── 两个 measure 别混 ────────────────────────────────────────────────────
#   --measure absdiff（默认）：逐被试 |quick| − |slow|。源估计取皮层法向分量，同一块
#       源在两组的符号可以相反，先取绝对值再相减不会把翻号读成减小。
#   --measure diff：逐被试 quick − slow 的带符号差，就是 stage 15 做置换检验用的那个量。
#       要和 15 的统计对齐时用它，但读图时记住蓝色可能是「翻号」而不是「更小」。
#
# ── 幅度口径（写作时别记错）──────────────────────────────────────────────
# 传感器水平：EEG quick/slow = 1.377（9/10，p = 0.0098，稳健）
#             MEG = 1.243（9/10，p = 0.0098；剔除试次不平衡者后 5/6、p = 0.156，不稳健）
# 这两个比值与这张图上的百分比**不是同一口径**，不能并列写。潜伏期差（EEG +26 ms、
# MEG +38 ms）三个读数都不显著，源空间不要写成「quick 的峰更早」。
#
# ── 成本 ──────────────────────────────────────────────────────────────────
# 两模态 × 1 组 × 2 视角 × 11 帧 = 44 次渲染，十几分钟量级；加 --with-members 是
# 三组六行，翻三倍。本机无 pyvista 只能跑 --dry-run 与 --self-test。
#
# 产出：output/source-quick-slow/source-quick-slow-{MEG,EEG}.png
#       output/source-quick-slow/source-quick-slow-{summary,timeline,persubject}.csv
#
# 渲染需要 3D 后端（pip install pyvista pyvistaqt），无显示器用 xvfb-run -a 跑整段。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/source-quick-slow.py

# 视角：俯视看中央区，后视看枕叶。数组，不要写成字符串——zsh 不对未加引号的变量分词
views=(dor cau)

# 1. 先看版式与色标上限（不需要 3D 后端，本机也能跑）
python $script --dry-run -v "${views[@]}" || {
    echo "!! dry-run 失败，先修上面的报错，不出正式图"
    exit 1
}
echo "---- dry-run 完了，看 output/source-quick-slow/source-quick-slow-*-DRYRUN.png ----"
echo "---- 日志里每行的 vmax ±xx% 与数据范围决定要不要调 --clim ----------"

# 2. 正式出图：两个模态各一张，只画差值（2 行）
python $script -v "${views[@]}"

# 3. 想把 quick、slow 各自的幅度也放进来对照（三组六行，渲染时间三倍）
# python $script -v "${views[@]}" --with-members

# 4. 要与 stage 15 的置换检验对齐（带符号配对差），或要绝对量
# python $script -v "${views[@]}" --measure diff
# python $script -v "${views[@]}" --normalize none --data-scale 1e15 --unit fAm

# 5. 觉得个别顶点把色标顶太高（大片偏淡），压低分位数
# python $script -v "${views[@]}" --clim-percent 95

# 无显示器时整段前面加 xvfb-run -a：
#   xvfb-run -a ./24.source-quick-slow.sh
