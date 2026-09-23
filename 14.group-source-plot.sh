#!/usr/bin/env zsh

# 文章图 4 的脑图截图：MEG 与 EEG 的群体 mean 源图。
#
# stage 13 已经把群体图算成了 stc 并写回磁盘，这个 stage 只做渲染：把 stc 平均
# 成群体 mean，用 stc.plot 渲在 fsaverage 表面上截图，一次同时出 MEG 和 EEG。
# 与 stage 13 的分工是「数字」和「图」，两者用的是同一个峰时刻读数，所以表格和
# 图永远说的是同一件事。
#
# 渲染需要 3D 后端，服务器上没有的话先装一个：
#   pip install pyvista pyvistaqt
#   conda install -c conda-forge mayavi      （旧的 PySurfer 路线）
# 无显示器的机器（ssh 上去的那种）加 xvfb-run：
#   xvfb-run -a ./14.group-source-plot.sh
#
# 只想先验证平均和选峰对不对、不碰渲染器，加 --dry-run：
#   python ./python/group-source-plot.py --dry-run
#
# 依赖：stage 8 的 output/source-estimation/{MODE}-{SUBJ}/*.stc-lh.stc
#       （不需要重跑 stage 8）
# 产出：output/group-source-map/group-{MODE}-{epochs}.{tag}-mean.stc
#                                       -mean-{hemi}-{t}s-{views}.png
#       output/group-source-map/group-{epochs}.{tag}-mean-contact-sheet.png（拼好的总图）
#       output/group-source-map/group-{epochs}.{tag}-mean-summary.csv
#
# 拷回本地只看图的话，只需要上面那几个 .png 和一个 .csv，stc 不用下载。

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

script=./python/group-source-plot.py

# ---- B 组：宽带 ERP，投影前 / 投影后 ----
# 这一对是"投影前峰值在运动皮层、投影后转到 0.3 s 皮层源"那句话的图。
# 默认只截群体图的峰时刻；要固定看几个已知时刻就用 --times，例如：
#   python $script -e epochs-1-notch-removal-artificial-epo.fif --times 0.29 0.35 0.41
python $script -e epochs-1-notch-removal-artificial-epo.fif
python $script -e epochs-1-notch-epo.fif

# ---- A 组：10 Hz SSVEP 的源功率 ----
# 功率图是全正的量，色标会自动切成正的一侧（pos_lims），不用手动调 clim。
python $script -e epochs-1-epo.fif -t ssvep10-power

# ---- 视角与半球 ----
# 默认半球分开出图（--hemi split，lh / rh 各一张），每张两个视角（lat + med），
# 这是论文单栏的常规排法。要一次看全就用：
#   python $script --hemi both --views lat
# 四视角（加背侧和腹侧）：
#   python $script --views lat med dor ven
