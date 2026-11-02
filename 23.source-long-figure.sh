#!/usr/bin/env zsh

# 文章用的源空间长图：一行一个条件的一个视角，从左到右 0-1 s，受试间平均。
#
# 两个视角：俯视 dor（默认看运动/中央区）+ 后视 cau（看枕叶视觉区）。每个条件出两行，
# 四行变八行。行序是「条件为组、组内按视角」，同一组共用一根色标，所以俯视与后视的
# 深浅在组内可以直接比。只想要四行就把 views 改成 "dor"。
#
# 四行（默认条件）：
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
# ── 画什么量 ──────────────────────────────────────────────────────────────
# 默认 --measure abs：逐被试取绝对值再平均，读激活幅度，不看方向。色标是单侧的
# 0~vmax，配色 magma（暗→亮）。想看正负极性就换 --measure signed（配色自动变
# RdBu_r）。配色不喜欢就给 --colormap：viridis inferno cividis turbo hot mne。

# ── 色标怎么调 ────────────────────────────────────────────────────────────
# 默认 --clim-mode row：每个条件各自定 vmax、各自一根色标（多个视角共用这一根）。
# 这几行幅度不在一个量级上
# （投影后只有投影前的两三成），统一色标会把投影后那行压成一片黑，所以不统一；
# 每行的 vmax 写在行标签上，量级关系从那里读。想让深浅直接比幅度再换 shared。
# 数值太小不好读时乘个量级：
#   python python/source-long-figure.py -m MEG --data-scale 1e15 --unit fA
# 觉得色标被个别顶点顶得太高（大片偏淡）就压低分位数：
#   python python/source-long-figure.py --clim-percent 95
# 想固定色标（跨模态、跨版本同图对比）就直接给数：
#   python python/source-long-figure.py -m MEG --clim 1.5e-13
# 嫌行标签上的「(0.30x row1)」啰嗦就 --no-ratio
#
# ── 其它常用开关 ──────────────────────────────────────────────────────────
#   --times 0.2 0.3 0.4 0.5          只出这几帧（默认 0-1 s 步长 0.1，共 11 帧）
#   --view dor cau                   视角，可给多个；dor 俯视（默认）、cau 后视、
#                                    lat 侧面、med 内侧、ven 腹面、fro 正面
#                                    写法 dor cau / "dor cau" / dor,cau 都认
#   --hemi lh | rh                   只看一侧
#   --measure signed                 看正负极性（默认 abs，只看幅度、magma 配色）
#   --colormap inferno               换配色（magma viridis inferno cividis turbo hot）
#   --no-ratio                       行标签不写相对第一行的倍数
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

# 视角：俯视 + 后视。后视看枕叶视觉区，两个视角让每个条件占两行。
# 这里是 zsh 数组，不是字符串：zsh 默认不对未加引号的变量做分词，
# 写成 views="dor cau" 再 -v $views 会把整串当一个参数递进去（会报“不认识的视角
# dor cau”）。数组配 "${views[@]}" 才逐个传。只想要四行就改成 views=(dor)。
views=(dor cau)

# 1. 先看版式与 vmax（本机也能跑，不需要 3D 后端）
python $script --dry-run -v "${views[@]}" || { echo "!! dry-run 失败，先修上面的报错，不出正式图"; exit 1; }
echo "---- dry-run 完了，看 output/source-long-figure/source-long-*-DRYRUN.png ----"

# 2. 正式出图（默认 abs + magma + 每行一根色标）。--data-scale 与 --unit 按 dry-run
#    日志里的数据范围定，这里留给你改：MNE 逆解是 Am 量级，乘 1e15 得到 fA 量级。
python $script -v "${views[@]}" --data-scale 1e15 --unit fA --clim-percent 99

# 3. 四行共用一根色标的版本（要直接比幅度时才跑，多一倍渲染时间，默认关掉）
# python $script -v "${views[@]}" --clim-mode shared --data-scale 1e15 --unit fA \
#     -o output/source-long-figure/shared

# 无显示器时整段前面加 xvfb-run -a：
#   xvfb-run -a ./23.source-long-figure.sh
