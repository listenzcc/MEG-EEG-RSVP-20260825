#!/usr/bin/env zsh

# 投影**后**的 quick / slow 源空间长图，形式与 stage 24 完全一致（默认六行）：
#   |quick| 均值（俯视 + 后视）、|slow| 均值（俯视 + 后视）、差值（俯视 + 后视）
# 从左到右 0-1 s，受试间平均。数据换成了投影后（remove artificial / rma）那一份。
#
# ── 24 与 25 是一对，别混 ────────────────────────────────────────────────
#   24  epochs-1-notch-epo.fif                   投影前：运动本体完整
#   25  epochs-1-notch-removal-artificial-epo.fif 投影后：运动子空间被去掉
# 同一批试次的同一套 quick / slow 划分（标签来自按键时刻的 RT，与投影无关），
# 只是数据走不走那一步投影。两张图并排放，读的是「投影拿掉了哪一块」。
#
# 两个脚本调的是同一个 python 脚本，25 只多了 --after-projection：
#   python/source-quick-slow.py --after-projection -o output/source-quick-slow-rma
# 产物前缀也随之变成 source-quick-slow-rma-*，两支即使指到同一目录也不互相覆盖。
#
# ── 为什么这一支值得单独出 ────────────────────────────────────────────────
# 24 的图里 quick 与 slow 的差别主要就是动作本身（中央区那一片），这与主线「0.4 s 的
# 峰是按键锁时运动响应」一致。投影后再看一次，回答的是另一半问题：去掉运动子空间之后，
# quick 与 slow 还剩不剩差别、剩在哪。如果剩在视觉区，说明 RT 调制有一部分不是运动本身；
# 如果几乎没剩下什么，那 quick/slow 这条轴在这个数据里就是运动轴。
#
# ── 统计口径（引读数前必看）──────────────────────────────────────────────
# stage 15 的置换检验就是跑在投影后这一份上（各自协方差，tag 不带 sharedcov），
# 标签×时间的符号翻转检验**两模态都没有通过 FDR**。所以：
#   · 这一支的图与 15 的 p 值同源，可以并排放，比 24 更适合与统计一起写。
#   · 但图上看到的差别仍然不能当显著：没有过 FDR 就是没有过。
#   · 逆算子仍是两组共用（量在投影前整个 target 条件上），所以 quick 与 slow 之差里
#     不含正则化差异；与 24 的同一条件相比也不含逆算子的差别。
#
# 幅度口径（沿 stage 15 / 16 / 24，写作时别记错）：
#   传感器水平 EEG quick/slow = 1.377（9/10，p = 0.0098，稳健）
#              MEG = 1.243（9/10，p = 0.0098；平衡子集 5/6、p = 0.156，不稳健）
# 这两个比值与图上的百分比**不是同一口径**，不能并列写。潜伏期差三个读数都不显著，
# 投影后更不该写成「quick 的峰更早」。
#
# ── 色标 ──────────────────────────────────────────────────────────────────
# 与 24 一致：成员行单侧 magma（0 在最下端）、差值行双向 coolwarm（0 居中），
# 三组各一根，深浅只组内可比。注意**不要跨 24 / 25 比深浅**——两支各自按自己的数据
# 定 vmax，行标签上写的是各自的 max。
#
# ── 成本 ──────────────────────────────────────────────────────────────────
# 两模态 × 3 组 × 2 视角 × 11 帧 = 132 次渲染，半小时往上；--no-members 减到 44 次。
# 本机无 pyvista 只能跑 --dry-run 与 --self-test。
#
# 产出：output/source-quick-slow-rma/source-quick-slow-rma-{MEG,EEG}.png
#       output/source-quick-slow-rma/source-quick-slow-rma-{summary,timeline,persubject}.csv
#
# 渲染需要 3D 后端（pip install pyvista pyvistaqt），无显示器用 xvfb-run -a 跑整段。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/source-quick-slow.py
outdir=output/source-quick-slow-rma

# 视角：俯视看中央区，后视看枕叶。数组，不要写成字符串——zsh 不对未加引号的变量分词
views=(dor cau)

# 1. 先看版式与色标上限（不需要 3D 后端，本机也能跑）
python $script --dry-run --after-projection -o $outdir -v "${views[@]}" || {
    echo "!! dry-run 失败，先修上面的报错，不出正式图"
    exit 1
}
echo "---- dry-run 完了，看 $outdir/source-quick-slow-rma-*-DRYRUN.png ----"
echo "---- 日志里每行的 vmax 与数据范围决定要不要调 --clim ----------"

# 2. 正式出图：两个模态各一张，六行（3 组 × 2 视角）
python $script --after-projection -o $outdir -v "${views[@]}"

# 3. 只要差值那两行（渲染量减三分之二，与 15 的统计对照时够用）
# python $script --after-projection -o $outdir -v "${views[@]}" --no-members

# 4. 要与 stage 15 的置换检验的符号口径完全对齐（带符号配对差）
# python $script --after-projection -o $outdir -v "${views[@]}" --measure diff

# 5. 觉得个别顶点把色标顶太高（大片偏淡），压低分位数
# python $script --after-projection -o $outdir -v "${views[@]}" --clim-percent 95

# 无显示器时整段前面加 xvfb-run -a：
#   xvfb-run -a ./25.source-quick-slow-rma.sh
