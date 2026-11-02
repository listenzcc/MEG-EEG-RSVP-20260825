"""
File: source-long-figure.py
Author: Chuncheng Zhang
Date: 2026-10-30

Purpose:
    一行一个条件的源空间激活长图：从左到右 0-1 s 的一排，受试间平均。

    默认四行，两个模态各出一张：

      before proj       epochs-1-notch-epo.fif                    ave
      after proj        epochs-1-notch-removal-artificial-epo.fif  ave
      quick (movement)  epochs-1-notch-epo.fif                    ave-quick-sharedcov
      slow  (movement)  epochs-1-notch-epo.fif                    ave-slow-sharedcov

    为什么 quick / slow 取投影前的文件：这两行要看的是运动本身（按键锁时的运动
    响应在快慢两组里长什么样），投影后的文件里那一块已经被投掉了。行由 -r 指定，
    不写就用上面这四行。

    视角可以给多个（-v dor cau）。给两个视角时每个条件出两行，四行变八行：同一
    条件的俯视与后视并排，后视看的是枕叶视觉区。行序是「条件为组、组内按视角」，
    同一组共用一根色标，所以俯视与后视的深浅在组内可以直接比。

Inputs:
    output/source-estimation/{MODE}-{SUBJ}/{epochs_fname}.{tag}-lh.stc / -rh.stc
    这些文件由 python/source-estimation.py（stage 8 / 15 / 16）写出，本脚本只读。
    所有被试都在 fsaverage 模板上，顶点一一对应，所以平均就是逐顶点取均值。

Outputs:
    output/source-long-figure/source-long-{MODE}.png           长图
    output/source-long-figure/source-long-summary.csv          每行（含视角）的 vmax / 被试 / 时刻
    output/source-long-figure/source-long-{MODE}-panels/*.png  逐格原图（--save-panels）

画什么量:
    默认 --measure abs：逐被试取绝对值再平均，读的是激活幅度，不看方向。色标因此
    是单侧的 0~vmax，配色走 magma（暗→亮）。要看正负极性（蓝负红正）就换
    --measure signed，配色自动换成 RdBu_r。

色标（这是留给你调的接口）:
    每个条件各自定色标（--clim-mode row，默认）。这几行的幅度本来就不在一个量级上——
    投影后只有投影前的两三成——共用一根色标会把投影后那一行压成一片黑，所以不统一；
    同一条件的多个视角共用一根（vmax 按条件定，跨视角不变）。
    每行的 vmax 写在组标签上，组与组的量级关系由那里读，形状由各组的色标读。
    确实想让深浅直接比幅度时再换 --clim-mode shared。

    单侧色标下 mne 对 0~vmax/2 做透明度斜坡（0 全透明），皮层灰底会透出来，
    所以低激活处不会糊成一片黑。

      --clim X          直接指定 vmax（与数据同单位，MNE 逆解是 Am 量级）
      --clim-percent P  vmax 取 |值| 的哪个分位数（默认 99）
      --clim-mode       row | shared（默认 row）
      --colormap        默认 magma（abs）/ RdBu_r（signed）；
                        备选 viridis inferno cividis turbo hot mne
      --colorbar        auto | row | shared | none（默认 auto：每行一根）
      --data-scale S    画之前把数据乘 S，色标刻度跟着变（例如 1e15 得到 fA 量级）
      --unit U          色标标题里的单位文字

Usage:
    python python/source-long-figure.py --dry-run -v dor cau
    python python/source-long-figure.py -m MEG -v dor cau
    python python/source-long-figure.py -m MEG --clim 1e-13 --colorbar row
    python python/source-long-figure.py --self-test -v dor cau
    python python/source-long-figure.py -m EEG --colormap inferno --clim-percent 98
    python python/source-long-figure.py -m MEG --measure signed --colormap RdBu_r
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
    - 俯视是 -v dor（默认），后视看枕叶用 -v cau（posterior / back 也认），
      正/侧/腹面用 fro lat med ven。hemi=both 时一格里面左右半球并排。
    - 时间网格 0-1 s 步长 0.1 s 共 11 帧，用 --start / --stop / --step 或 --times 改。
    - 两个模态 × 四行 × 两个视角 × 11 帧 = 176 次渲染，按每格两三秒估，整段半小时
      上下；只跑一个视角就减半。--render-mode reuse 能再快一些。
"""


# %% ---- 2026-10-30 ------------------------
# Requirements and constants
import argparse
import logging
import re
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

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger('source-long-figure')


def split_tokens(values):
    '''把 nargs='+' 收上来的东西拆成单个 token。

    三种写法都得认：`-v dor cau`（nargs 已拆开）、`-v "dor cau"`（引号里带空格，
    zsh 默认不对未加引号的变量分词，所以脚本里给变量加引号时就是这一种）、
    `-v dor,cau`。不拆的话整串会被当成一个名字，报“不认识的视角 dor cau”。
    '''
    out = []
    for val in values:
        for tok in re.split(r'[,;/|\s]+', str(val).strip()):
            if tok:
                out.append(tok)
    return out

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
parser.add_argument('--measure', default='abs', choices=['signed', 'abs'],
                    help='abs：逐被试取绝对值再平均，读激活幅度（默认，方向不看）；'
                         'signed：逐被试带符号的群体均值，看正负极性')
parser.add_argument('--data-scale', type=float, default=1.0,
                    help='画之前把数据乘这个数，只影响色标刻度与数字')
parser.add_argument('--unit', default='',
                    help='色标标题里的单位文字，例如 fA / nAm')

# 渲染
parser.add_argument('-v', '--view', '--views', dest='views', nargs='+',
                    default=['dor'],
                    help='视角，可以给多个：dor（俯视，默认）、cau（后视，看枕叶视觉区）、'
                         'lat、med、ven、ros、fro、par。给两个就每个条件出两行，'
                         '行数翻倍。也认 dorsal / posterior / back 这类全名。'
                         '写法 -v dor cau / -v "dor cau" / -v dor,cau 都认')
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
                    help='默认 abs 用 magma、signed 用 RdBu_r；'
                         '备选 viridis inferno cividis turbo hot mne')
parser.add_argument('--colorbar', default='auto',
                    choices=['auto', 'row', 'shared', 'none'],
                    help='auto：row 模式每行一根、shared 模式一根；也可强制')
parser.add_argument('--colorbar-symmetric', action='store_true',
                    help='强制色标以零为中心（全正的数据也想要双侧刻度时用）')
parser.add_argument('--no-ratio', action='store_true',
                    help='行标签上不写「相对第一行的倍数」（四行各自定色标时，'
                         '这行小字只作量级参考，嫌啰嗦就关掉）')

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

MODES = [m.upper() for m in split_tokens(args.modes)]
DATA_DIR = Path(args.stc_dir)
OUTPUT_DIR = Path(args.output_dir)

# 视角：用户写全名也认，内部一律换成 mne 的三字母写法
VIEW_ALIASES = {
    'dor': 'dor', 'dorsal': 'dor', 'top': 'dor', 'axi': 'dor', 'axial': 'dor',
    'cau': 'cau', 'caudal': 'cau', 'posterior': 'cau', 'back': 'cau',
    'ven': 'ven', 'ventral': 'ven', 'bottom': 'ven',
    'lat': 'lat', 'lateral': 'lat',
    'med': 'med', 'medial': 'med',
    'ros': 'ros', 'rostral': 'ros',
    'fro': 'fro', 'frontal': 'fro', 'front': 'fro',
    'par': 'par', 'parietal': 'par',
}
VIEW_NAMES = {'dor': 'dorsal', 'cau': 'posterior', 'ven': 'ventral',
              'lat': 'lateral', 'med': 'medial', 'ros': 'rostral',
              'fro': 'frontal', 'par': 'parietal'}


def split_view_tokens(values):
    '''把 --view 收上来的东西拆成单个视角名。'''
    return split_tokens(values)


def normalize_view(name):
    '''把 --view 的写法统一成 mne 认识的三字母视角名。'''
    key = str(name).lower()
    if key not in VIEW_ALIASES:
        raise SystemExit(f'不认识的视角 {name}，可选：'
                         + '、'.join(sorted(VIEW_ALIASES)))
    return VIEW_ALIASES[key]


VIEWS = []
for _v in split_view_tokens(args.views):
    _v = normalize_view(_v)
    if _v not in VIEWS:
        VIEWS.append(_v)
if not VIEWS:
    raise SystemExit('--view 没给视角')
VIEW_TAGS = [VIEW_NAMES[v] for v in VIEWS]

# 这两个在 main 里按参数定，函数里当全局读
CMAP = args.colormap or ('magma' if args.measure == 'abs' else 'RdBu_r')
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
        return list(split_tokens(args.subjects))
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


_MEASURE_FIG = {}


def text_width_in(text, fontsize):
    '''
    一段文字有多宽（英寸），用来定左边的留白。

    用 Agg 的渲染器实测 get_window_extent。TextPath 按字形度量估出来的宽度比实际
    窄，偏窄就意味着一行「N=10 · max 0.3 (0.30x row1)」的左端会被图边界裁掉。
    多行取最宽的一行。
    '''
    fig = _MEASURE_FIG.get('fig')
    if fig is None:
        fig = plt.figure(figsize=(1, 1), dpi=200)
        _MEASURE_FIG['fig'] = fig
    fig.clf()
    renderer = fig.canvas.get_renderer()
    widest = 0.
    for line in str(text).split('\n'):
        t = fig.text(0.5, 0.5, line, fontsize=fontsize)
        widest = max(widest, t.get_window_extent(renderer=renderer).width)
    fig.clf()
    return widest / fig.dpi


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


def row_note(n_subjects, vmax, ratio=None, is_first=False):
    '''
    行标签下面的那行小字。

    四行各自定色标，深浅不能跨行读，所以量级写在这里：N 与 vmax，外加相对第一行
    的倍数。--no-ratio 之后只留 N 与 vmax。
    '''
    word = 'max' if args.measure == 'abs' else 'vmax'
    text = f'N={n_subjects} · {word} {vmax:.3g}'
    if ratio is not None and not is_first and not args.no_ratio \
            and np.isfinite(ratio):
        text += f' ({ratio:.2f}x row1)'
    return text


def assemble(mode, panels, times, labels, notes, bars, cmap, fname,
             subtitle=None, tags=None, groups=None):
    '''
    把逐格截图拼成一张长图。

    手工排布（fig.add_axes）而不用 gridspec：列数由时刻决定、行高由截图的宽高比
    决定，用相对坐标最不容易出意外，改列数也不用重算布局。

    Args:
        mode: MEG / EEG，进标题
        panels: panels[row][col]，每项是 HxWx3 的 ndarray
        times: 每列的时刻（s）
        labels: 每行的大字标签（groups 给了就不用）
        notes: 每行的小字（N 与 max）（groups 给了就不用）
        bars: 每行一个 dict(vmax, symmetric, span, show)，画色标用；
            span 是这根色标覆盖几行（同一条件的多个视角共用一根）
        cmap: 与 stc.plot 用同一个 colormap
        fname: 输出文件名
        subtitle: 第二行小字，默认由参数拼
        tags: 每行的视角名（多视角时才有，单视角传 None 就不占位）
        groups: 多视角时每组的 dict(start, span, label, note)。标签与量级只写在
            组的中线上一份，视角名写在各自行上——八行各写一遍 label 会挤成一团
    '''
    n_rows, n_cols = len(panels), len(panels[0])
    h, w = panels[0][0].shape[:2]

    # 字号跟着单格宽度缩放：11 列的长图缩到一页宽之后，9 pt 会小到看不见
    f = args.font_scale
    fs_time, fs_row, fs_note = 11 * f, 11 * f, 9 * f
    fs_tag = 9.5 * f
    fs_title, fs_sub, fs_bar = 14 * f, 10 * f, 8 * f
    tags = list(tags) if tags else [''] * n_rows
    if groups:
        left_texts = [(g['label'], fs_row) for g in groups] \
            + [(g['note'], fs_note) for g in groups]
    else:
        left_texts = list(zip(labels, [fs_row] * len(labels))) \
            + list(zip(notes, [fs_note] * len(notes)))

    col_w = 2.0                     # 一列脑图的宽度（英寸）
    row_h = col_w * h / w
    # 左边留白按最长的那行标签实测宽度算，留 0.30 英寸的边距
    widest_label = max([text_width_in(v, fs) for v, fs in left_texts] or [0.])
    # 多视角时左边排成两列：条件标签一列、视角名一列。两列的宽度都要先量出来，
    # 否则标签的第一行会伸进视角名那一栏里（自检里「dorsal」压着「target ERP」）
    tag_w = max([text_width_in(t, fs_tag) for t in tags if t] or [0.])
    tag_w = tag_w + 0.02 if tag_w else 0.
    left = max(1.25, 0.30 + widest_label + (tag_w + 0.14 if tag_w else 0.))
    top, bottom = 0.80, 0.06
    bar_w = 0.10
    cbar_mode = args.colorbar
    if cbar_mode == 'auto':
        cbar_mode = 'none' if args.clim_mode == 'shared' else 'row'
    nbar = n_rows if cbar_mode == 'row' else (1 if cbar_mode == 'shared' else 0)
    right = 0.10
    if nbar:
        # 右边留白按最长的刻度文字实测：MEG 的刻度是 9.6e-15 这种七个字符，
        # 写死 0.6 英寸会被右边界切掉最后一位
        vals = [b['vmax'] for b in bars if b.get('vmax')] or [0.]
        texts = [f'{v:.2g}' for v in vals] + [f'{v / 2:.2g}' for v in vals] \
            + ['0', args.unit]
        widest = max(text_width_in(t, fs_bar) for t in texts)
        right = 0.12 + bar_w + 0.08 + widest + 0.26

    W = left + n_cols * col_w + right
    H = top + n_rows * row_h + bottom
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor('white')

    def row_y(r):
        '''第 r 行的下边界（图坐标，0 在下）。'''
        return (H - top - (r + 1) * row_h) / H

    for r in range(n_rows):
        y = row_y(r)
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
        if groups:
            # 多视角：视角名单独一列，紧贴脑图；条件标签在它左边一列
            if tags[r]:
                fig.text((left - 0.06) / W, yc, tags[r], ha='right',
                         va='center', fontsize=fs_tag, color='0.10')
        else:
            x_label = (left - 0.18) / W
            fig.text(x_label, yc + 0.11 * row_h / H, labels[r],
                     ha='right', va='center', fontsize=fs_row, linespacing=1.45)
            fig.text(x_label, yc - 0.17 * row_h / H, notes[r],
                     ha='right', va='center', fontsize=fs_note, color='0.35')

        if cbar_mode == 'row' and bars[r].get('vmax') and bars[r].get('show', True):
            # 同一条件的几个视角共用一根色标：bar 落在第一行上，span 覆盖它们
            span = int(bars[r].get('span', 1))
            base = row_y(r + span - 1)
            cax = fig.add_axes([(left + n_cols * col_w + 0.12) / W,
                                base + span * row_h / H * 0.12,
                                bar_w / W, span * row_h / H * 0.76])
            draw_bar(fig, cax, bars[r]['vmax'], bars[r]['symmetric'],
                     cmap, fs_bar, args.unit)

    if groups:
        # 条件标签与量级写在组的中线上，右对齐到「视角名那一列的左边」
        x_label = (left - 0.06 - tag_w - 0.14) / W
        for g in groups:
            ymid = (H - top - (g['start'] + g['span'] / 2.) * row_h) / H
            fig.text(x_label, ymid + 0.05 / H, g['label'], ha='right',
                     va='bottom', fontsize=fs_row, linespacing=1.35)
            if g['note']:
                fig.text(x_label, ymid - 0.05 / H, g['note'], ha='right',
                         va='top', fontsize=fs_note, color='0.35')

    if cbar_mode == 'shared' and bars[0]['vmax']:
        cax = fig.add_axes([(left + n_cols * col_w + 0.12) / W,
                            (H - top - n_rows * row_h + 0.12) / H,
                            bar_w / W, (n_rows * row_h - 0.24) / H])
        draw_bar(fig, cax, bars[0]['vmax'], bars[0]['symmetric'],
                 cmap, fs_bar, args.unit)

    title = args.title or f'{mode} group mean source estimate'
    fig.text(0.5, (H - 0.10) / H, title, ha='center', va='top', fontsize=fs_title)
    view_txt = ' + '.join(VIEW_TAGS)
    fig.text(0.5, (H - 0.34) / H,
             subtitle or (f'{args.measure} | view {view_txt} | {n_rows} rows x '
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


def make_brain(stc, clim, t0, view):
    '''
    建一个脑图窗口用来截屏。

    标题留空：每格的时刻标在拼图时写，不靠 mne 的 title；colorbar 关掉，色标统一
    在拼图里画，这样整张长图的色标样式一致，也方便调。
    '''
    return stc.plot(subject='fsaverage', subjects_dir=SUBJECTS_DIR,
                    hemi=args.hemi, views=[view], initial_time=float(t0),
                    time_viewer=False, clim=clim, colormap=CMAP,
                    colorbar=False, size=tuple(args.panel_size),
                    background='white', foreground='black',
                    cortex=args.cortex,
                    smoothing_steps=args.smoothing_steps,
                    title='', verbose='ERROR',
                    brain_kwargs=dict(show=False))


def render_row(stc, clim, times, render_mode, view):
    '''
    渲染一行的所有帧。

    rebuild 每帧重建一次脑图（stage 14 走的就是这条路，稳）；reuse 建一次然后
    set_time 逐帧截屏，快得多，但重绘链要是没生效就会截出一排同样子的图，所以这里
    自己验一下相邻帧，一旦相同就回退 rebuild。
    '''
    if render_mode == 'rebuild':
        images = []
        for t in times:
            brain = make_brain(stc, clim, t, view)
            images.append(np.asarray(brain.screenshot(mode='rgb')))
            brain.close()
        return images

    images = []
    brain = make_brain(stc, clim, times[0], view)
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
        return render_row(stc, clim, times, 'rebuild', view)
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
    for v in vmaxes:
        for _view in VIEWS:          # 条件 × 视角，和主流程排法一致
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

    if args.colorbar == 'auto':
        args.colorbar = 'row' if args.clim_mode == 'row' else 'shared'
    sym = args.colorbar_symmetric or args.measure != 'abs'
    multi = len(VIEWS) > 1

    labels_out, notes_out, tags_out, bars_out = [], [], [], []
    groups = []
    for i, v in enumerate(vmaxes):
        vbar = v if args.clim_mode == 'row' else vmaxes[0]
        note = row_note(10, v, v / vmaxes[0], is_first=(i == 0))
        for j, view in enumerate(VIEWS):
            labels_out.append(labels[i])
            notes_out.append(note)
            tags_out.append(VIEW_NAMES[view] if multi else '')
            bars_out.append(dict(vmax=vbar, symmetric=sym, span=len(VIEWS),
                                 show=(j == 0)))
        groups.append(dict(start=i * len(VIEWS), span=len(VIEWS),
                           label=labels[i], note=note))

    flat = [im for row in panels for im in row]
    cropped = apply_crop(flat, [crop_box(im) for im in flat])
    n = len(times)
    n_rows = len(vmaxes) * len(VIEWS)
    panels = [cropped[r * n:(r + 1) * n] for r in range(n_rows)]

    assemble('SELF', panels, times, labels_out, notes_out, bars_out,
             plt.get_cmap(CMAP), OUTPUT_DIR / 'source-long-SELF-TEST.png',
             subtitle='self test, synthetic panels', tags=tags_out,
             groups=groups if multi else None)
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

    rows_spec = [parse_row(s) for s in split_tokens(args.rows)]
    row_labels = split_tokens(args.row_labels) if args.row_labels else None
    if row_labels and len(row_labels) != len(rows_spec):
        raise SystemExit(f'--row-labels 有 {len(row_labels)} 项，'
                         f'-r 有 {len(rows_spec)} 行，对不上')
    labels = (row_labels if row_labels
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

        # 一行 = 一个条件的一个视角。同一条件的几个视角连着排，共用一根色标
        # （vmax 是按条件定的，跨视角不变，所以两个视角的深浅可以直接比）。
        plan = [(i, view) for i in range(len(prepared)) for view in VIEWS]
        multi = len(VIEWS) > 1

        panels, labels_out, notes_out, tags_out, bars_out = [], [], [], [], []
        groups = []
        for k, (i, view) in enumerate(plan):
            p = prepared[i]
            if p['data'].shape != template.data.shape:
                raise SystemExit(
                    f'{mode} row {i + 1} 的数据是 {p["data"].shape}，'
                    f'模板是 {template.data.shape}，顶点或时刻对不上')
            stc = template.copy()
            stc.data = p['data']
            stc.subject = 'fsaverage'
            clim = clim_dict(p['vmax_used'], p['data_min'])

            note = row_note(len(p['subjects']), p['vmax_used'], p['ratio'],
                            is_first=(i == 0))
            labels_out.append(labels[i])
            notes_out.append(note)
            tags_out.append(VIEW_NAMES[view] if multi else '')
            bars_out.append(dict(vmax=p['vmax_used'], symmetric=p['symmetric'],
                                 span=len(VIEWS), show=(view == VIEWS[0])))
            if view == VIEWS[0]:
                groups.append(dict(start=k, span=len(VIEWS),
                                   label=labels[i], note=note))

            logger.info(f'{mode} row {k + 1} '
                        f'{labels[i].replace(chr(10), "/")} · {VIEW_NAMES[view]}'
                        f' clim = {clim}')

            if args.dry_run:
                # 浅灰占位格，只验版式：行标签、时刻、色标位置都能先看一眼，
                # 不必等到服务器上跑完渲染才发现排版不对
                hh, ww = args.panel_size[1], args.panel_size[0]
                blank = np.full((hh, ww, 3), 248, np.uint8)
                blank[[0, -1], :, :] = 190
                blank[:, [0, -1], :] = 190
                panels.append([blank.copy() for _ in times])
                continue

            imgs = render_row(stc, clim, times, args.render_mode, view)
            panels.append(imgs)
            if args.save_panels:
                folder = OUTPUT_DIR / f'source-long-{mode}-panels'
                folder.mkdir(parents=True, exist_ok=True)
                for im, t in zip(imgs, times):
                    plt.imsave(folder / f'row{k + 1}-{VIEW_NAMES[view]}-'
                                        f'{t:.3f}s.png', im)

            summary.append(dict(
                mode=mode, row=k + 1, epochs_fname=p['row']['epochs'],
                tag=p['row']['tag'], label=labels[i].replace('\n', ' '),
                view=VIEW_NAMES[view],
                n_subjects=len(p['subjects']),
                subjects=' '.join(p['subjects']),
                measure=args.measure, hemi=args.hemi,
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
        panels = [cropped[r * n:(r + 1) * n] for r in range(len(plan))]

        suffix = '-DRYRUN' if args.dry_run else ''
        assemble(mode, panels, times, labels_out, notes_out, bars_out,
                 plt.get_cmap(CMAP),
                 OUTPUT_DIR / f'source-long-{mode}{suffix}.png',
                 tags=tags_out, groups=groups if multi else None)
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
