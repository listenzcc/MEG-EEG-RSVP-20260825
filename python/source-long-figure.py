"""
File: source-long-figure.py
Author: Chuncheng Zhang
Date: 2026-10-30

Purpose:
    一行一个条件的源空间激活长图：从左到右 0-1 s 的一排，俯视图，受试间平均。

    默认四行，两个模态各出一张：

      before proj       epochs-1-notch-epo.fif                    ave
      after proj        epochs-1-notch-removal-artificial-epo.fif  ave
      quick (movement)  epochs-1-notch-epo.fif                    ave-quick-sharedcov
      slow  (movement)  epochs-1-notch-epo.fif                    ave-slow-sharedcov

    为什么 quick / slow 取投影前的文件：这两行要看的是运动本身（按键锁时的运动
    响应在快慢两组里长什么样），投影后的文件里那一块已经被投掉了。行由 -r 指定，
    不写就用上面这四行。

Inputs:
    output/source-estimation/{MODE}-{SUBJ}/{epochs_fname}.{tag}-lh.stc / -rh.stc
    这些文件由 python/source-estimation.py（stage 8 / 15 / 16）写出，本脚本只读。
    所有被试都在 fsaverage 模板上，顶点一一对应，所以平均就是逐顶点取均值。

Outputs:
    output/source-long-figure/source-long-{MODE}.png           长图
    output/source-long-figure/source-long-summary.csv          每行的 vmax / 被试 / 时刻
    output/source-long-figure/source-long-{MODE}-panels/*.png  逐格原图（--save-panels）

色标（这是留给你调的接口）:
    投影后的幅度只有投影前的两三成，共用色标会把投影后那一行压成一片白。所以默认
    --clim-mode row：每行按自己显示的那些帧定 vmax，削减量改由行标题上的 vmax 与
    「相对第一行的比值」给出，形状与量级都不丢。要目测幅度就换 --clim-mode shared。

      --clim X          直接指定 vmax（与数据同单位，MNE 逆解是 Am 量级）
      --clim-percent P  vmax 取 |值| 的哪个分位数（默认 99）
      --clim-mode       row | shared
      --colormap        默认 RdBu_r（有符号，蓝负红正）；--measure abs 时默认 hot
      --colorbar        auto | row | shared | none
      --data-scale S    画之前把数据乘 S，色标刻度跟着变（例如 1e15 得到 fA 量级）
      --unit U          色标标题里的单位文字

Usage:
    python python/source-long-figure.py --dry-run
    python python/source-long-figure.py -m MEG
    python python/source-long-figure.py -m MEG --clim 1e-13 --colorbar row
    python python/source-long-figure.py --self-test
    python python/source-long-figure.py -m EEG --clim-mode shared --clim-percent 98
    python python/source-long-figure.py -r epochs-1-notch-epo.fif:ave \
        epochs-1-notch-removal-artificial-epo.fif:ave --row-labels before after

Notes:
    - 四行默认**不共用**同一个逆算子：第 1/3/4 行都量在 epochs-1-notch-epo.fif 的整个
      target 条件上（before 的 ave 自然就是它，quick/slow 由 stage 16 指过去），
      第 2 行（投影后的 ave）是 stage 8 用投影后文件自己的协方差算的。所以跨行读
      「(0.xx x row1)」时含一点正则化差异。要严格共用，补跑一次
      `source-estimation.py --epochs_fname epochs-1-notch-removal-artificial-epo.fif
      --cov_epochs_fname epochs-1-notch-epo.fif --tag_suffix sharedcov`，
      第 2 行改用 tag ave-sharedcov（23.source-long-figure.sh 的文件头有循环写法）。
    - 渲染走 stc.plot，需要 3D 后端（pyvista）；无显示器的机器用 xvfb-run -a。
    - 图内文字用英文，避免服务器缺中文字体。
    - 俯视图是 --view dor（dorsal），hemi=both 时一格里面左右半球并排。
      正/侧/腹面改 --view fro lat med ven。
    - 时间网格 0-1 s 步长 0.1 s 共 11 帧，用 --start / --stop / --step 或 --times 改。
    - 两个模态 × 四行 × 11 帧 = 88 次渲染，按每格两三秒估，整段十几分钟。
"""


# %% ---- 2026-10-30 ------------------------
# Requirements and constants
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mne
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger('source-long-figure')

# 默认四行：投影前后的 target ERP，和投影前的 quick / slow（运动本体）
ROWS_DEFAULT = [
    'epochs-1-notch-epo.fif:ave',
    'epochs-1-notch-removal-artificial-epo.fif:ave',
    'epochs-1-notch-epo.fif:ave-quick-sharedcov',
    'epochs-1-notch-epo.fif:ave-slow-sharedcov',
]

parser = argparse.ArgumentParser(
    description='一行一个条件的源空间激活长图（0-1 s，俯视图，受试间平均）')

# 画什么
parser.add_argument('-r', '--rows', nargs='+', default=list(ROWS_DEFAULT),
                    help='每一行写成 epochs_fname:tag，例如 '
                         'epochs-1-notch-epo.fif:ave。默认四行见文件头')
parser.add_argument('--row-labels', nargs='+', default=None,
                    help='每行的标签，顺序与 -r 一致；不写就从文件名与 tag 推')
parser.add_argument('-m', '--modes', nargs='+', default=['MEG', 'EEG'],
                    help='要画的模态，默认两个都画，各出一张图')
parser.add_argument('-s', '--subjects', nargs='*', default=None,
                    help='被试名，默认扫 output/source-estimation 下所有被试')
parser.add_argument('--allow-partial', action='store_true',
                    help='允许各行用不同的被试集合。默认取各行的交集，这样行与行之间'
                         '的幅度才可比，被丢掉的人会打日志')

# 时间轴
parser.add_argument('--start', type=float, default=0.0, help='时间网格起点（s）')
parser.add_argument('--stop', type=float, default=1.0, help='时间网格终点（s），含')
parser.add_argument('--step', type=float, default=0.1,
                    help='时间网格步长（s）；0-1 s 步长 0.1 共 11 帧')
parser.add_argument('--times', type=float, nargs='*', default=None,
                    help='直接给时刻列表，给了就不看 --start/--stop/--step')

# 画成什么量
parser.add_argument('--measure', default='signed', choices=['signed', 'abs'],
                    help='signed：逐被试带符号的群体均值（默认）；'
                         'abs：逐被试取绝对值再平均，读的是激活幅度')
parser.add_argument('--data-scale', type=float, default=1.0,
                    help='画之前把数据乘这个数，只影响色标刻度与数字')
parser.add_argument('--unit', default='',
                    help='色标标题里的单位文字，例如 fA / nAm')

# 渲染
parser.add_argument('--view', default='dor',
                    help='视角，默认 dor（俯视）；lat med ven fro par 亦可')
parser.add_argument('--hemi', default='both', choices=['both', 'lh', 'rh'],
                    help='both 两个半球并排一格，默认；lh / rh 只画一侧')
parser.add_argument('--cortex', default='classic',
                    help='皮层渲染方式，classic | low_contrast | high_contrast | bone')
parser.add_argument('--smoothing-steps', type=int, default=7, help='表面平滑步数')
parser.add_argument('--panel-size', type=int, nargs=2, default=[700, 330],
                    help='单格的像素尺寸（宽 高），hemi=both 时宽要够放两个半球')
parser.add_argument('--render-mode', default='rebuild',
                    choices=['rebuild', 'reuse'],
                    help='rebuild：每格重建一次脑图（默认，稳）；'
                         'reuse：每行建一次、逐帧 set_time 截屏（快，但一旦发现'
                         '相邻帧完全相同就自动回退 rebuild）')
parser.add_argument('--no-crop', action='store_true',
                    help='不裁掉每格四周的白边（默认裁，长图更紧凑）')
parser.add_argument('--dpi', type=int, default=150)
parser.add_argument('--font-scale', type=float, default=1.0,
                    help='字号整体缩放，默认 1.0（列宽 2 英寸时时间标签 11 pt）。'
                         '列数多、图被压小时调大')

# 色标
parser.add_argument('--clim', type=float, default=None,
                    help='直接指定 vmax（与数据同单位，已含 --data-scale）。'
                         '给了就不看 --clim-percent')
parser.add_argument('--clim-percent', type=float, default=99.0,
                    help='vmax 取 |值| 的哪个分位数，默认 99')
parser.add_argument('--clim-mode', default='row', choices=['row', 'shared'],
                    help='row：每行各自定 vmax（默认，投影后那行才看得见形状）；'
                         'shared：一个模态的所有行共用一个 vmax，深浅可直接比幅度')
parser.add_argument('--colormap', default=None,
                    help='默认 signed 用 RdBu_r、abs 用 hot')
parser.add_argument('--colorbar', default='auto',
                    choices=['auto', 'row', 'shared', 'none'],
                    help='auto：row 模式每行一根、shared 模式一根；也可强制')
parser.add_argument('--colorbar-symmetric', action='store_true',
                    help='强制色标以零为中心（全正的数据也想要双侧刻度时用）')

# 其它
parser.add_argument('-t', '--title', default=None,
                    help='图的标题，默认自动写模态')
parser.add_argument('-d', '--stc-dir', default='output/source-estimation',
                    help='源估计结果的根目录，默认 output/source-estimation。'
                         '合成沙盒里用它可以指到别处')
parser.add_argument('-o', '--output-dir', default='output/source-long-figure',
                    help='产物目录')
parser.add_argument('--save-panels', action='store_true',
                    help='把每格原图也存下来（文件名带行号与时刻）')
parser.add_argument('--dry-run', action='store_true',
                    help='只读 stc、算平均与色标，不碰渲染器')
parser.add_argument('--self-test', action='store_true',
                    help='用合成图跑一遍排版与色标，不需要 stc 与 3D 后端')

args = parser.parse_args()

MODES = [m.upper() for m in args.modes]
DATA_DIR = Path(args.stc_dir)
OUTPUT_DIR = Path(args.output_dir)

# 这两个在 main 里按参数定，函数里当全局读
CMAP = args.colormap or ('hot' if args.measure == 'abs' else 'RdBu_r')
SUBJECTS_DIR = None


# %% ---- 2026-10-30 ------------------------
# Function and class
def time_grid(times_all):
    '''
    要画的时刻，吸附到 stc 自己的采样点上。

    存的是吸附后的值：stc.plot 会把 initial_time 取到最近的样本，图上标的时刻必须
    就是数据里的时刻，否则 0.300 实际画的是 0.295。

    Args:
        times_all: stc 的时间轴

    Returns:
        times: 吸附后的时刻列表
    '''
    if args.times:
        wanted = list(args.times)
    else:
        if args.step <= 0:
            raise SystemExit('--step 必须大于 0')
        n = int(round((args.stop - args.start) / args.step)) + 1
        wanted = [args.start + i * args.step for i in range(n)]

    times = []
    for t in wanted:
        snapped = float(times_all[int(np.argmin(np.abs(times_all - t)))])
        if snapped in times:
            continue
        if abs(snapped - t) > 1e-4:
            logger.debug(f'{t:.3f} s 吸附到 {snapped:.3f} s')
        times.append(snapped)
    if not times:
        raise SystemExit('时间网格是空的，检查 --start / --stop / --step')
    return times


def parse_row(spec):
    '''把 epochs_fname:tag 拆成两半。'''
    if ':' not in spec:
        raise SystemExit(f'-r 的每一项要写成 epochs_fname:tag，读到的是 {spec}')
    epochs_fname, tag = spec.split(':', 1)
    return epochs_fname.strip(), tag.strip()


def default_label(epochs_fname, tag):
    '''
    行标签。写英文，服务器上不一定有中文字体。

    投影状态由文件名说，分组由 tag 说：quick / slow 那一行不是「target ERP」，
    而是「快或慢那一半试次的平均」。
    '''
    state = 'before proj' if 'removal-artificial' not in epochs_fname else 'after proj'
    if 'quick' in tag:
        return f'quick\n{state}'
    if 'slow' in tag:
        return f'slow\n{state}'
    if 'ssvep' in tag:
        return f'SSVEP\n{state}'
    return f'target ERP\n{state}'


def stc_stem(mode, subj, epochs_fname, tag):
    '''
    找一名被试的 stc。stage 8 存成 {epochs}.{tag}.stc，mne 读的时候自己加 -lh/-rh。
    '''
    folder = DATA_DIR / f'{mode}-{subj}'
    for stem in (folder / f'{epochs_fname}.{tag}.stc',
                 folder / f'{epochs_fname}.{tag}'):
        if Path(str(stem) + '-lh.stc').exists():
            return stem
    return None


def subject_names(mode):
    '''被试名单，-s 优先，否则扫盘。'''
    if args.subjects:
        return list(args.subjects)
    return sorted(p.name.split('-')[-1]
                  for p in DATA_DIR.glob(f'{mode}-*') if p.is_dir())


def load_rows(mode, rows):
    '''
    读入每一行、每一名被试的 stc 数据。

    不做平均：被试集合要先在行之间对齐（见 --allow-partial），色标也要看过所有行
    的样本才能定，所以先只把数据拿全。

    Returns:
        loaded: 每行一个 dict(epochs, tag, blocks, missing)
        times: 时间轴
        template: 第一个读到的 stc，只借它的顶点编号与时间轴，数据不用
    '''
    wanted = subject_names(mode)
    if not wanted:
        raise SystemExit(f'{DATA_DIR} 下没有 {mode}-* 目录，stage 8 还没跑？')

    loaded, times, template = [], None, None
    for epochs_fname, tag in rows:
        blocks, missing = {}, []
        for subj in wanted:
            stem = stc_stem(mode, subj, epochs_fname, tag)
            if stem is None:
                missing.append(subj)
                continue
            stc = mne.read_source_estimate(str(stem))
            if template is None:
                # 群体图要借 fsaverage 的顶点编号，自己新建一个 SourceEstimate
                # 拿不到那套编号，所以留一份模板，后面只换 data
                template = stc.copy()
                times = np.asarray(stc.times, float)
            block = stc.data.astype(np.float64)
            if args.measure == 'abs':
                block = np.abs(block)
            if args.data_scale != 1.0:
                block = block * args.data_scale
            blocks[subj] = block
        if not blocks:
            raise SystemExit(
                f'{mode} 的这一行没有任何 stc：{epochs_fname} tag={tag}\n'
                f'  找的是 {DATA_DIR / (mode + "-*")}/'
                f'{epochs_fname}.{tag}-lh.stc\n'
                '先确认这个条件由哪个 stage 出：\n'
                '  ave                       -> 8.source-estimation.sh\n'
                '  ave-quick / ave-slow      -> 15.rt-source-contrast.sh\n'
                '  ave-quick / ave-slow + -sharedcov -> 16.rt-source-projection.sh')
        loaded.append(dict(epochs=epochs_fname, tag=tag, blocks=blocks,
                           missing=missing))
    return loaded, times, template


def align_subjects(loaded):
    '''
    决定每一行用哪几名被试。

    默认取交集：行与行之间要比幅度，被试集合不同就会把「换了几个人」混进差别里。
    --allow-partial 时各行各用自己的人，N 也就逐行不同。

    Returns:
        messages: 几行日志，说明谁被丢掉了
    '''
    if args.allow_partial:
        return []
    sets = [set(r['blocks']) for r in loaded]
    common = set.intersection(*sets)
    if not common:
        raise SystemExit('各行没有共同的被试，检查 -r 与 -s')
    messages = []
    for r in loaded:
        gone = sorted(set(r['blocks']) - common)
        if gone:
            messages.append(f'{r["epochs"]}:{r["tag"]} 缺 {gone}，'
                            f'这一行按共同的 {len(common)} 人平均')
            for subj in gone:
                r['blocks'].pop(subj)
        if r['missing']:
            logger.debug(f'{r["epochs"]}:{r["tag"]} 盘上没有 '
                         f'{r["missing"]}')
    return messages


def group_mean(row):
    '''逐顶点、逐时刻的平均。'''
    keys = sorted(row['blocks'])
    return np.mean([row['blocks'][k] for k in keys], axis=0), keys


def pick_vmax(frames):
    '''
    定 vmax。

    取要显示的那些帧里 |值| 的分位数，不是整条时间轴：0.6 s 之后的漂移和基线极值
    会把色标顶上去，真正要看的这十几帧就全平了。

    Args:
        frames: (n_vertices, n_frames) 的数组列表

    Returns:
        vmax: 上限，数据单位；0 表示交给 mne
    '''
    if args.clim is not None:
        return float(args.clim)
    vals = np.concatenate([np.abs(f).ravel() for f in frames])
    if vals.size == 0:
        return 0.
    vmax = float(np.nanpercentile(vals, args.clim_percent))
    if not np.isfinite(vmax) or vmax <= 0:
        logger.warning(f'vmax 算出来是 {vmax}，色标交给 mne')
        return 0.
    return vmax


def is_symmetric(data_min):
    '''
    色标要不要以零为中心。

    有负值就双侧（蓝负红正）；全正的数据（abs、SSVEP 功率）用单侧，否则整张图会
    挤在色标的上半段。--colorbar-symmetric 可以强制双侧。
    '''
    return bool(data_min < 0 or args.colorbar_symmetric)


def clim_dict(vmax, data_min):
    '''stc.plot 认识的色标，单双侧的判断与色标条一致。'''
    if not vmax or vmax <= 0:
        return 'auto'
    if is_symmetric(data_min):
        return dict(kind='value', lims=[-vmax, 0., vmax])
    return dict(kind='value', pos_lims=[0., vmax / 2., vmax])


def crop_box(img):
    '''非白像素的包围盒 (row0, row1, col0, col1)，全白时 None。'''
    gray = np.asarray(img[..., :3], float).mean(axis=-1)
    mask = gray < 250
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def apply_crop(images, boxes, pad=2):
    '''
    用同一个包围盒裁所有格子。

    同一个盒子是关键：各裁各的，缩放就不一致，相邻两格会把同一块皮层画成不同大小。
    '''
    if args.no_crop:
        return images
    ok = [b for b in boxes if b is not None]
    if not ok:
        return [np.full_like(im, 255) for im in images]
    r0 = max(min(b[0] for b in ok) - pad, 0)
    c0 = max(min(b[2] for b in ok) - pad, 0)
    r1 = max(b[1] for b in ok) + pad
    c1 = max(b[3] for b in ok) + pad
    return [np.ascontiguousarray(im[r0:r1 + 1, c0:c1 + 1]) for im in images]


def text_width_in(text, fontsize):
    '''
    一段文字有多宽（英寸），用来定左边的留白。

    用 TextPath 量，比按字数估准：中文、·、负号这些宽度都不一样，估小了标签会被
    左边界裁掉。多行取最宽的一行。
    '''
    wp = FontProperties(size=fontsize)
    return max((TextPath((0, 0), line, prop=wp).get_extents().width / 72.
                for line in str(text).split('\n')), default=0.)


def draw_bar(fig, cax, vmax, symmetric, cmap, fontsize, unit=''):
    '''
    画一根色标条。

    单双侧必须与 stc.plot 的 clim 一致，否则色条与脑图对不上：单侧时 mne 是把
    [0, vmax] 线性铺满整张颜色表，双侧是 [-vmax, vmax]，两者不能混。

    刻度只标两端与零点，并且逐刻度写成科学计数法——MNE 逆解是 1e-14 量级，默认
    的「1e-15 偏移 + 5/0/-5」读不出量级。
    '''
    if symmetric:
        norm = Normalize(vmin=-vmax, vmax=vmax)
        ticks = [-vmax, 0., vmax]
    else:
        norm = Normalize(vmin=0., vmax=vmax)
        ticks = [0., vmax / 2., vmax]
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 orientation='vertical')
    cax.set_yticks(ticks)
    cax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:.2g}'))
    cax.tick_params(labelsize=fontsize, length=2)
    if unit:
        cax.set_title(unit, fontsize=fontsize)


def assemble(mode, panels, times, labels, notes, bars, cmap, fname,
             subtitle=None):
    '''
    把逐格截图拼成一张长图。

    手工排布（fig.add_axes）而不用 gridspec：列数由时刻决定、行高由截图的宽高比
    决定，用相对坐标最不容易出意外，改列数也不用重算布局。

    Args:
        mode: MEG / EEG，进标题
        panels: panels[row][col]，每项是 HxWx3 的 ndarray
        times: 每列的时刻（s）
        labels: 每行的大字标签
        notes: 每行的小字（N 与 vmax）
        bars: 每行一个 dict(vmax, symmetric)，画色标用
        cmap: 与 stc.plot 用同一个 colormap
        fname: 输出文件名
        subtitle: 第二行小字，默认由参数拼
    '''
    n_rows, n_cols = len(panels), len(panels[0])
    h, w = panels[0][0].shape[:2]

    # 字号跟着单格宽度缩放：11 列的长图缩到一页宽之后，9 pt 会小到看不见
    f = args.font_scale
    fs_time, fs_row, fs_note = 11 * f, 11 * f, 9 * f
    fs_title, fs_sub, fs_bar = 14 * f, 10 * f, 8 * f

    col_w = 2.0                     # 一列脑图的宽度（英寸）
    row_h = col_w * h / w
    # 左边留白按最长的那行标签算，固定值短了会把标签裁掉
    left = max([1.25] +
               [text_width_in(v, fs) + 0.20
                for v, fs in list(zip(labels, [fs_row] * len(labels)))
                + list(zip(notes, [fs_note] * len(notes)))])
    top, bottom = 0.80, 0.06
    bar_w = 0.10
    cbar_mode = args.colorbar
    if cbar_mode == 'auto':
        cbar_mode = 'none' if args.clim_mode == 'shared' else 'row'
    nbar = n_rows if cbar_mode == 'row' else (1 if cbar_mode == 'shared' else 0)
    right = (bar_w + 0.62) if nbar else 0.10

    W = left + n_cols * col_w + right
    H = top + n_rows * row_h + bottom
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor('white')

    for r in range(n_rows):
        y = (H - top - (r + 1) * row_h) / H
        for c in range(n_cols):
            x = (left + c * col_w) / W
            ax = fig.add_axes([x, y, col_w / W, row_h / H])
            ax.imshow(panels[r][c], aspect='auto', interpolation='nearest')
            ax.set_axis_off()
            if r == 0:
                fig.text(x + col_w / W / 2, (H - top + 0.07) / H,
                         f'{times[c] * 1000:.0f} ms', ha='center', va='bottom',
                         fontsize=fs_time)

        yc = (H - top - (r + 0.5) * row_h) / H
        fig.text((left - 0.18) / W, yc + 0.11 * row_h / H, labels[r],
                 ha='right', va='center', fontsize=fs_row, linespacing=1.45)
        fig.text((left - 0.18) / W, yc - 0.17 * row_h / H, notes[r],
                 ha='right', va='center', fontsize=fs_note, color='0.35')

        if cbar_mode == 'row' and bars[r]['vmax']:
            cax = fig.add_axes([(left + n_cols * col_w + 0.12) / W,
                                y + row_h / H * 0.16,
                                bar_w / W, row_h / H * 0.68])
            draw_bar(fig, cax, bars[r]['vmax'], bars[r]['symmetric'],
                     cmap, fs_bar, args.unit)

    if cbar_mode == 'shared' and bars[0]['vmax']:
        cax = fig.add_axes([(left + n_cols * col_w + 0.12) / W,
                            (H - top - n_rows * row_h + 0.12) / H,
                            bar_w / W, (n_rows * row_h - 0.24) / H])
        draw_bar(fig, cax, bars[0]['vmax'], bars[0]['symmetric'],
                 cmap, fs_bar, args.unit)

    title = args.title or f'{mode} group mean source estimate'
    fig.text(0.5, (H - 0.10) / H, title, ha='center', va='top', fontsize=fs_title)
    fig.text(0.5, (H - 0.34) / H,
             subtitle or (f'{args.measure} | view {args.view} | {n_rows} rows x '
                          f'{n_cols} frames | colour scale: {args.clim_mode}'),
             ha='center', va='top', fontsize=fs_sub, color='0.35')

    fig.savefig(fname, dpi=args.dpi, facecolor='white')
    plt.close(fig)
    logger.info(f'Saved into {fname}, {W:.1f} x {H:.1f} in')


def check_backend():
    '''渲染前先确认 3D 后端在，免得读完数据才失败。'''
    try:
        backend = mne.viz.get_3d_backend()
    except Exception as e:
        raise SystemExit(
            f'没有 3D 后端，{e}。装一个：\n'
            '  pip install pyvista pyvistaqt\n'
            '无显示器的机器用 xvfb-run -a 跑整条命令。')
    logger.info(f'The 3d backend is {backend}')
    return backend


def make_brain(stc, clim, t0):
    '''
    建一个脑图窗口用来截屏。

    标题留空：每格的时刻标在拼图时写，不靠 mne 的 title；colorbar 关掉，色标统一
    在拼图里画，这样整张长图的色标样式一致，也方便调。
    '''
    return stc.plot(subject='fsaverage', subjects_dir=SUBJECTS_DIR,
                    hemi=args.hemi, views=[args.view], initial_time=float(t0),
                    time_viewer=False, clim=clim, colormap=CMAP,
                    colorbar=False, size=tuple(args.panel_size),
                    background='white', foreground='black',
                    cortex=args.cortex,
                    smoothing_steps=args.smoothing_steps,
                    title='', verbose='ERROR',
                    brain_kwargs=dict(show=False))


def render_row(stc, clim, times, render_mode):
    '''
    渲染一行的所有帧。

    rebuild 每帧重建一次脑图（stage 14 走的就是这条路，稳）；reuse 建一次然后
    set_time 逐帧截屏，快得多，但重绘链要是没生效就会截出一排同样子的图，所以这里
    自己验一下相邻帧，一旦相同就回退 rebuild。
    '''
    if render_mode == 'rebuild':
        images = []
        for t in times:
            brain = make_brain(stc, clim, t)
            images.append(np.asarray(brain.screenshot(mode='rgb')))
            brain.close()
        return images

    images = []
    brain = make_brain(stc, clim, times[0])
    try:
        for t in times:
            brain.set_time(float(t))
            images.append(np.asarray(brain.screenshot(mode='rgb')))
    finally:
        brain.close()

    same = sum(np.array_equal(images[i], images[i + 1])
               for i in range(len(images) - 1))
    if same:
        logger.warning(f'reuse 下有 {same}/{len(images) - 1} 对相邻帧完全相同，'
                       '重绘链没生效，这一行改用 rebuild 重截')
        return render_row(stc, clim, times, 'rebuild')
    return images


def run_self_test():
    '''
    合成图跑一遍排版、色标、裁边。

    不需要 stc、不需要 fsaverage、不需要 3D 后端，任何机器上都能跑，用来确认
    「拼图这一段」本身是对的——真正渲染的部分要上服务器才知道。
    '''
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    times = [round(t, 3) for t in np.arange(0., 1.0001, 0.1)]
    labels = ['target ERP\nbefore proj', 'target ERP\nafter proj',
              'quick\nbefore proj', 'slow\nbefore proj']
    vmaxes = [1.0, 0.3, 0.8, 0.7]

    panels = []
    rr, cc = np.ogrid[:110, :240]
    for r, v in enumerate(vmaxes):
        row = []
        for c, _ in enumerate(times):
            img = np.full((110, 240, 3), 255, np.uint8)
            for k, (yy, xx) in enumerate([(40, 70), (58, 150), (78, 200)]):
                amp = v * (1 + 0.5 * np.sin(c / 2. + k))
                mask = (rr - yy) ** 2 + (cc - xx) ** 2 < 19 ** 2
                shade = int(255 - 200 * min(amp, 1.0))
                img[mask] = (shade, shade, 255 - (255 - shade) // 3)
            row.append(img)
        panels.append(row)

    notes = [f'N=10 · vmax {v:.2f}' + ('' if i == 0 else
                                       f' ({v / vmaxes[0]:.2f}x row1)')
             for i, v in enumerate(vmaxes)]
    if args.colorbar == 'auto':
        args.colorbar = 'row' if args.clim_mode == 'row' else 'shared'
    rows_seen = [dict(vmax=v, symmetric=True) for v in
                 (vmaxes if args.clim_mode == 'row'
                  else [vmaxes[0]] * len(vmaxes))]

    flat = [im for row in panels for im in row]
    cropped = apply_crop(flat, [crop_box(im) for im in flat])
    n = len(times)
    panels = [cropped[r * n:(r + 1) * n] for r in range(len(vmaxes))]

    assemble('SELF', panels, times, labels, notes, rows_seen,
             plt.get_cmap(CMAP), OUTPUT_DIR / 'source-long-SELF-TEST.png',
             subtitle='self test, synthetic panels')
    print(f'time grid: {times}')
    print(f'clim (signed) {clim_dict(1.0, -0.5)}   (one sided) {clim_dict(1.0, 0.0)}')
    print(f'colormap: {CMAP}, colorbar: {args.colorbar}, mode: {args.clim_mode}')


def main():
    '''
    主流程：读 stc -> 对齐被试 -> 群体平均 -> 定色标 -> 渲染 -> 拼图。

    色标一定要在看完全部行之后才能定（--clim-mode shared），所以分两段：先把每行
    的数据与要显示的帧拿全，再逐行渲染。
    '''
    global SUBJECTS_DIR

    rows_spec = [parse_row(s) for s in args.rows]
    if args.row_labels and len(args.row_labels) != len(rows_spec):
        raise SystemExit(f'--row-labels 有 {len(args.row_labels)} 项，'
                         f'-r 有 {len(rows_spec)} 行，对不上')
    labels = (list(args.row_labels) if args.row_labels
              else [default_label(e, t) for e, t in rows_spec])

    try:
        fs_dir = mne.datasets.fetch_fsaverage(verbose='ERROR')
        SUBJECTS_DIR = str(Path(fs_dir).parent)
        logger.info(f'The template is in {SUBJECTS_DIR}')
    except Exception as e:
        if not args.dry_run:
            raise SystemExit(f'fsaverage 模板取不到，{e}。渲染离不开它。')
        logger.warning(f'fsaverage 取不到，dry-run 不需要：{e}')

    if not args.dry_run:
        check_backend()

    summary = []
    for mode in MODES:
        loaded, times_all, template = load_rows(mode, rows_spec)
        for msg in align_subjects(loaded):
            logger.warning(msg)

        prepared = []
        for row in loaded:
            data, subjects = group_mean(row)
            prepared.append(dict(row=row, data=data, subjects=subjects))

        times = time_grid(times_all)
        idx = [int(np.argmin(np.abs(times_all - t))) for t in times]
        logger.info(f'{mode}: {len(times)} 帧 {times[0]:.3f}-{times[-1]:.3f} s，'
                    f'各行 N = ' +
                    ', '.join(str(len(p['subjects'])) for p in prepared))

        # 每行自己的 vmax 与符号范围
        for p in prepared:
            frames = p['data'][:, idx]
            p['vmax_own'] = pick_vmax([frames])
            p['data_min'] = float(np.nanmin(frames))
            p['data_max'] = float(np.nanmax(frames))
            p['symmetric'] = is_symmetric(p['data_min'])

        if args.clim_mode == 'shared':
            shared = pick_vmax([p['data'][:, idx] for p in prepared])
            logger.info(f'{mode} 全图共用一个 vmax = {shared:.4g}')
            for p in prepared:
                p['vmax_used'] = shared
        else:
            for p in prepared:
                p['vmax_used'] = p['vmax_own']

        first = prepared[0]['vmax_used'] or np.nan
        for i, p in enumerate(prepared):
            ratio = p['vmax_used'] / first
            p['ratio'] = ratio
            logger.info(f'{mode} row {i + 1} '
                        f'{labels[i].replace(chr(10), "/")}: '
                        f'{len(p["subjects"])} 被试, '
                        f'vmax {p["vmax_used"]:.4g}, '
                        f'{ratio:.2f}x row1, 数据范围 '
                        f'[{p["data_min"]:.4g}, {p["data_max"]:.4g}]')

        panels = []
        for i, p in enumerate(prepared):
            if p['data'].shape != template.data.shape:
                raise SystemExit(
                    f'{mode} row {i + 1} 的数据是 {p["data"].shape}，'
                    f'模板是 {template.data.shape}，顶点或时刻对不上')
            stc = template.copy()
            stc.data = p['data']
            stc.subject = 'fsaverage'
            clim = clim_dict(p['vmax_used'], p['data_min'])
            logger.info(f'{mode} row {i + 1} clim = {clim}')

            if args.dry_run:
                # 浅灰占位格，只验版式：行标签、时刻、色标位置都能先看一眼，
                # 不必等到服务器上跑完渲染才发现排版不对
                hh, ww = args.panel_size[1], args.panel_size[0]
                blank = np.full((hh, ww, 3), 248, np.uint8)
                blank[[0, -1], :, :] = 190
                blank[:, [0, -1], :] = 190
                panels.append([blank.copy() for _ in times])
                continue

            imgs = render_row(stc, clim, times, args.render_mode)
            panels.append(imgs)
            if args.save_panels:
                folder = OUTPUT_DIR / f'source-long-{mode}-panels'
                folder.mkdir(parents=True, exist_ok=True)
                for im, t in zip(imgs, times):
                    plt.imsave(folder / f'row{i + 1}-{t:.3f}s.png', im)

            summary.append(dict(
                mode=mode, row=i + 1, epochs_fname=p['row']['epochs'],
                tag=p['row']['tag'], label=labels[i].replace('\n', ' '),
                n_subjects=len(p['subjects']),
                subjects=' '.join(p['subjects']),
                measure=args.measure, view=args.view, hemi=args.hemi,
                times=' '.join(f'{t:.3f}' for t in times),
                vmax=f'{p["vmax_used"]:.6g}',
                vmax_vs_row1=f'{p["ratio"]:.4f}'
                if np.isfinite(p['ratio']) else '',
                data_min=f'{p["data_min"]:.6g}',
                data_max=f'{p["data_max"]:.6g}',
                clim_mode=args.clim_mode, colormap=CMAP))

        # 全图用同一张颜色表，色标按行画，幅度之间的差别由行标签上的 vmax 给出
        flat = [im for row in panels for im in row]
        cropped = apply_crop(flat, [crop_box(im) for im in flat])
        n = len(times)
        panels = [cropped[r * n:(r + 1) * n] for r in range(len(prepared))]

        notes = [f'N={len(p["subjects"])} · vmax {p["vmax_used"]:.3g}'
                 + ('' if i == 0 else f' ({p["ratio"]:.2f}x row1)')
                 for i, p in enumerate(prepared)]
        suffix = '-DRYRUN' if args.dry_run else ''
        bars = [dict(vmax=p['vmax_used'], symmetric=p['symmetric'])
                for p in prepared]
        assemble(mode, panels, times, labels, notes, bars,
                 plt.get_cmap(CMAP),
                 OUTPUT_DIR / f'source-long-{mode}{suffix}.png')
        if args.dry_run:
            logger.info(f'{mode} dry-run 到此为止：没有渲染，上面这张是空格的版式图')

    if not summary:
        logger.info('dry-run 没有写 summary，去掉 --dry-run 才有')
        return
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
    fname = OUTPUT_DIR / 'source-long-summary.csv'
    df.to_csv(fname, index=False)
    logger.info(f'Saved into {fname}')


# %% ---- 2026-10-30 ------------------------
# Play ground
if args.self_test:
    run_self_test()
else:
    main()


# %% ---- 2026-10-30 ------------------------
# Pending
# 1. 俯视图看的是朝上的那一面，沟里的活动读不到；要看深部得换 --cortex 或
#    改成 volume 源。
# 2. 色标只按分位数定，没有统计掩膜：哪块「激活」是真的还得回
#    group-source-map.py 的置换检验读数。
