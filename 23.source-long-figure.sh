#!/usr/bin/env zsh

# 文章用的源空间长图：一行一个条件，从左到右 0-1 s，俯视图，受试间平均。
#
# 四行（默认）：
#   before proj      epochs-1-notch-epo.fif                    ave
#   after proj       epochs-1-notch-removal-artificial-epo.fif  ave
#   quick (movement) epochs-1-notch-epo.fif                    ave-quick-sharedcov
#   slow  (movement) epochs-1-notch-epo.fif                    ave-slow-sharedcov
#
# quick / slow 取投影前的文件：这两行看的是运动本体（快慢两组各自的按键锁时运动
# 响应长什么样），投影之后那一块已经被投掉了。想要投影后的 quick / slow 就把
# -r 换成 epochs-1-notch-removal-artificial-epo.fif:ave-quick（stage 15 的 tag，
# 不带 sharedcov）。
#
# 两个模态各出一张图（单位不同，不能同图）：
#   output/source-long-figure/source-long-MEG.png
#   output/source-long-figure/source-long-EEG.png
#
# ── 四行是不是共用同一个逆算子 ────────────────────────────────────────────
# 不是。第 1/3/4 行都量在 epochs-1-notch-epo.fif 的整个 target 条件上（before 的
# ave 自然就是它，quick/slow 由 stage 16 用 --cov_epochs_fname 指过去），所以这
# 三行共用一个算子。第 2 行（投影后的 ave）是 stage 8 用投影后文件自己的协方差
# 算的，算子与前三行不同，跨行读 (0.xx x row1) 时会把正则化差异混进比值里。
#
# 要让四行真正共用一个算子，补跑投影后的 ave（每人一次，比 stage 16 便宜）：
#   for mode in MEG EEG; do for subj in S01 S02 S03 S04 S05 S06 S07 S08 S09 S10; do
#     python python/source-estimation.py --subject $subj --mode $mode \
#       --epochs_fname epochs-1-notch-removal-artificial-epo.fif \
#       --cov_epochs_fname epochs-1-notch-epo.fif --tag_suffix sharedcov
#   done; done
# 然后把第 2 行换成：
#   epochs-1-notch-removal-artificial-epo.fif:ave-sharedcov
# 第 1 行不用动：它的协方差本来就量在 epochs-1-notch-epo.fif 上，与上面这条命令
# 用的是同一个，加不加 --tag_suffix 数值都一样。
#
# ── 先跑 dry-run，再跑真的 ────────────────────────────────────────────────
# dry-run 不碰渲染器，会出一张空格的版式图，用来看行标签、时刻、色标位置对不对，
# 顺便把每行的 vmax 与数据范围打到日志里。看完再决定 --clim / --data-scale：
#   python python/source-long-figure.py --dry-run
#
# ── 色标怎么调 ────────────────────────────────────────────────────────────
# 默认 --clim-mode row：每行按自己显示的 11 帧定 vmax，投影后那一行才看得见形状；
# 削减量由行标签上的「(0.30x row1)」给出。要目测幅度换 --clim-mode shared。
# 数值太小不好读时乘个量级：
#   python python/source-long-figure.py -m MEG --data-scale 1e15 --unit fA
# 觉得色标被个别顶点顶得太高（大片偏淡）就压低分位数：
#   python python/source-long-figure.py --clim-percent 95
# 想固定色标（跨模态、跨版本同图对比）就直接给数：
#   python python/source-long-figure.py -m MEG --clim 1.5e-13
#
# ── 其它常用开关 ──────────────────────────────────────────────────────────
#   --times 0.2 0.3 0.4 0.5          只出这几帧（默认 0-1 s 步长 0.1，共 11 帧）
#   --view dor | lat | med | ven     俯视图是 dor（默认）
#   --hemi lh | rh                   只看一侧
#   --measure abs                    看激活幅度（逐被试取绝对值再平均）
#   --render-mode reuse              每行只建一次脑图，快；发现截图不变会自动回退
#   --save-panels                    逐格 png 也留下来，方便单独放大
#   --font-scale 1.3                 图里字太小就调大
#   -s S01 S02 S03                   先跑几个人试排版
#
# ── 依赖 ──────────────────────────────────────────────────────────────────
# 读 output/source-estimation 下 stage 8 / 15 / 16 写好的 stc；渲染需要 3D 后端
# （pip install pyvista pyvistaqt），无显示器的机器用 xvfb-run -a 跑整段。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/source-long-figure.py

# 1. 先看版式与 vmax（本机也能跑，不需要 3D 后端）
python $script --dry-run
echo "---- dry-run 完了，看 output/source-long-figure/source-long-*-DRYRUN.png ----"

# 2. 正式出图。--data-scale 与 --unit 按 dry-run 日志里的数据范围定，这里留给
#    你改：MNE 逆解是 Am 量级，乘 1e15 得到 fA 量级的刻度。
python $script --data-scale 1e15 --unit fA --clim-percent 99

# 3. 想按幅度读（共用色标）时补一张，文件名不一样，不会覆盖上面的
python $script --clim-mode shared --data-scale 1e15 --unit fA \
    -o output/source-long-figure/shared

# 无显示器时整段前面加 xvfb-run -a：
#   xvfb-run -a ./23.source-long-figure.sh
