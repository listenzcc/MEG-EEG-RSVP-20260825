"""
File: source-quick-slow.py
Date: 2026-11-03

Purpose:
    quick 与 slow 两组目标试次的源空间差值长图：一行一组视角，从左到右 0-1 s，
    受试间平均。stage 23 把这两组各画了一行（各看各的），这一支把它们**相减**，
    让差别自己显形——差值在哪一段、落在枕区还是中央区、俯视与后视是否同一个故事。

    同一份数据还能顺手给出两组各自的幅度作为对照行（--with-members），三者用同一个
    单位（相对该被试参考幅度的百分比），所以「差值 = 两条成员线之差」在逐被试层面
    严格成立。

Inputs:
    output/source-estimation/{MODE}-{SUBJ}/{epochs}.{tag}-lh.stc（tag 见 -r）
    默认读 stage 16 的 ave-quick-sharedcov / ave-slow-sharedcov：它们量在投影前
    那一份文件的协方差上，两组共用同一个逆算子，所以两组之差里不含正则化差异。
    stage 15 的 ave-quick / ave-slow（不带 sharedcov）是各自协方差，只有明确要
    复现那一路读数时才用。

Outputs:
    output/source-quick-slow/source-quick-slow-{MODE}.png        长图
    output/source-quick-slow/source-quick-slow-summary.csv       逐被试读数
    output/source-quick-slow/source-quick-slow-timeline.csv      逐时刻差值曲线
    output/source-quick-slow/panels-{MODE}/                      --save-panels 时的逐格

Usage:
    python python/source-quick-slow.py --dry-run -v dor cau
    python python/source-quick-slow.py -m MEG -v dor cau
    python python/source-quick-slow.py --with-members -v cau --clim-percent 95
    python python/source-quick-slow.py --measure diff --normalize none --data-scale 1e15
    python python/source-quick-slow.py --self-test -v dor cau

Notes:
    - **差值必然是双向色标**。这一支与 stage 23 不同：stage 23 只看幅度，单侧 magma；
      这里要说「谁更大」，零点必须有意义，所以默认 coolwarm、以 0 为中心对称。
    - 默认 --measure absdiff：逐被试算 |quick| − |slow| 再跨被试平均。源估计取的是
      皮层法向分量，同一块源在两组的符号可以相反，先取绝对值再相减不会把翻号读成
      减小。要复现 stage 15 的带符号配对差（那是置换检验用的量），用 --measure diff。
    - 默认 --normalize ref：差值除以该被试自己的参考幅度（两组 99 分位的均值，一个
      标量），结果读作「相对本人整体激活水平的百分比」。逐顶点做分母会在 |q|+|s| 近零
      的顶点上爆掉，所以分母取标量。
    - 逐被试先算差值再平均（配对）。这与「先平均两组再相减」在数学上等价，但配对差
      的跨被试离散度小得多，summary 里两种都给了。
    - 与 stage 15 的分工：15 出统计（标签×时间的符号翻转置换，两模态都没过 FDR）与
      单时刻渲染；这一支出 0-1 s 的连续图像。**图不替代统计**，正文引用显著性仍回 15。
    - 幅度口径别混：传感器水平 EEG quick/slow = 1.377（9/10，p = 0.0098，稳健）、
      MEG = 1.243（9/10，p = 0.0098；剔除试次不平衡者后 5/6、p = 0.156，不稳健）。
      这一支的百分比是「相对参考幅度」，与那两个比值不同口径，不能并列。
    - 渲染走 stc.plot，需要 3D 后端（pyvista）；无显示器的机器用 xvfb-run -a。
    - 图内文字用英文，避免服务器缺中文字体。
    - 两模态 × 一组差值 × 两视角 × 11 帧 = 44 次渲染，十几分钟量级；--with-members
      翻三倍。
"""


# %% ---- 2026-11-03 ------------------------
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
logger = logging.getLogger('source-quick-slow')


def split_tokens(values):
    '''把 nargs='+' 收上来的东西拆成单个 token。

    `-v dor cau` / `-v "dor cau"` / `-v dor,cau` 三种写法都认。zsh 不会对未加引号的
    变量分词，脚本里给变量加引号传多个值时，落地就是第二种。
    '''
    out = []
    for val in values:
        for tok in re.split(r'[,;/|\s]+', str(val).strip()):
            if tok:
                out.append(tok)
    return out


# %% ---- 2026-11-03 ------------------------
# The command line
parser = argparse.ArgumentParser(
    description='quick 与 slow 两组目标试次的源空间差值长图（0-1 s，受试间平均）')

# 数据
parser.add_argument('-r', '--tags', nargs='+',
                    default=['ave-quick-sharedcov', 'ave-slow-sharedcov'],
                    help='两个 tag，顺序是 quick slow。默认 stage 16 的 sharedcov 版'
                         '（两组共用同一个逆算子）')
parser.add_argument('--epochs', default='epochs-1-notch-epo.fif',
                    help='读哪个 epochs 文件的 stc，默认 target 条件')
parser.add_argument('-d', '--stc-dir', default='output/source-estimation',
                    help='stc 所在目录，默认 output/source-estimation')
parser.add_argument('-m', '--modes', nargs='+', default=['MEG', 'EEG'],
                    help='要画的模态，默认两个都画，各出一张图')
parser.add_argument('-s', '--subjects', nargs='*', default=None,
                    help='被试名，默认扫 stc 目录下所有被试')

# 差值与归一化
parser.add_argument('--measure', default='absdiff',
                    choices=['absdiff', 'diff'],
                    help='absdiff：逐被试 |quick| − |slow|（默认，不受源极性翻号影响）；'
                         'diff：逐被试 quick − slow 的带符号差（stage 15 统计用的那个量）')
parser.add_argument('--normalize', default='ref', choices=['ref', 'none'],
                    help='ref：除以该被试的参考幅度，读作百分比（默认）；'
                         'none：保留原始单位，配 --data-scale 用')
parser.add_argument('--ref-percent', type=float, default=99.,
                    help='参考幅度取这一分位数，默认 99')
parser.add_argument('--data-scale', type=float, default=1.0,
                    help='--normalize none 时乘这个数，只影响色标刻度与数字')
parser.add_argument('--unit', default=None,
                    help='色标标题里的单位，默认 normalize=ref 时是 %%，否则空')
parser.add_argument('--with-members', action='store_true',
                    help='除差值外再画 quick、slow 各自的幅度（三组 × 视角，行数三倍）')

# 时间轴
parser.add_argument('--start', type=float, default=0.0, help='时间网格起点（s）')
parser.add_argument('--stop', type=float, default=1.0, help='时间网格终点（s）')
parser.add_argument('--step', type=float, default=0.1, help='时间网格步长（s）')
parser.add_argument('--times', type=float, nargs='+', default=None,
                    help='直接给要画的时刻（s），给了就不用 start/stop/step')

# 视角
parser.add_argument('-v', '--view', '--views', dest='views', nargs='+',
                    default=['dor'],
                    help='视角，可以给多个：dor（俯视，默认）、cau（后视，看枕叶）、'
                         'lat、med、ven、ros、fro、par。给两个每个条件占两行。'
                         '写法 -v dor cau / -v "dor cau" / -v dor,cau 都认')
parser.add_argument('--hemi', default='both', choices=['both', 'lh', 'rh'],
                    help='both 两个半球并排一格，默认；lh / rh 只画一侧')
parser.add_argument('--cortex', default='classic',
                    help='皮层渲染方式，classic | low_contrast | high_contrast | bone')
parser.add_argument('--smoothing-steps', type=int, default=7, help='表面平滑步数')

# 色标
parser.add_argument('--clim', type=float, default=None,
                    help='固定色标上限（差值是对称的，范围就是 ±clim）')
parser.add_argument('--clim-percent', type=float, default=99.,
                    help='不给 --clim 时按这个分位数定上限，默认 99')
parser.add_argument('--clim-mode', default='row', choices=['row', 'shared'],
                    help='row：一组一根色标（默认，成员组与差值组量级不同）；'
                         'shared：全图一根')
parser.add_argument('--colorbar', default='auto', choices=['auto', 'row', 'shared', 'none'])
parser.add_argument('--colormap', default=None,
                    help='配色，默认 coolwarm（差值必须双向）。想要 stage 23 的单侧'
                         'magma 配 --with-members 的成员行也能对上，但差值行会读不出正负')

# 渲染
parser.add_argument('--panel-size', type=int, nargs=2, default=[700, 330],
                    help='单格的像素尺寸（宽 高），hemi=both 时宽要够放两个半球')
parser.add_argument('--render-mode', default='rebuild',
                    choices=['rebuild', 'reuse'],
                    help='rebuild：每格重建一次脑图（默认，稳）；'
                         'reuse：每行建一次、逐帧 set_time 截屏（快，自动回退）')
parser.add_argument('--no-crop', action='store_true',
                    help='不裁掉每格四周的白边（默认裁，长图更紧凑）')
parser.add_argument('--dpi', type=int, default=150)
parser.add_argument('--font-scale', type=float, default=1.0,
                    help='图里字太小就调大，默认 1.0')
parser.add_argument('--title', default=None, help='图标题，默认按模态与 measure 拼')
parser.add_argument('--no-ratio', action='store_true',
                    help='组标签不写相对第一组的倍数')

# 运行
parser.add_argument('-o', '--output', default='output/source-quick-slow',
                    help='输出目录')
parser.add_argument('--save-panels', action='store_true',
                    help='逐格 png 也留下来，方便单独放大')
parser.add_argument('--dry-run', action='store_true',
                    help='只读 stc、算差值与色标，不碰渲染器')
parser.add_argument('--self-test', action='store_true',
                    help='用合成图跑一遍排版与色标，不需要 stc 与 3D 后端')

args = parser.parse_args()

DATA_DIR = Path(args.stc_dir)
OUTPUT_DIR = Path(args.output)

TAGS = split_tokens(args.tags)
if len(TAGS) != 2:
    raise SystemExit(f'-r 要两个 tag（quick slow），读到 {len(TAGS)} 个：{TAGS}')
TAG_QUICK, TAG_SLOW = TAGS

MODES = [m.upper() for m in split_tokens(args.modes)]
UNIT = args.unit if args.unit is not None \
    else ('%' if args.normalize == 'ref' else '')

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


def normalize_view(name):
    '''把 --view 的写法统一成 mne 认识的三字母视角名。'''
    key = str(name).lower()
    if key not in VIEW_ALIASES:
        raise SystemExit(f'不认识的视角 {name}，可选：'
                         + '、'.join(sorted(VIEW_ALIASES)))
    return VIEW_ALIASES[key]


VIEWS = []
for _v in split_tokens(args.views):
    _v = normalize_view(_v)
    if _v not in VIEWS:
        VIEWS.append(_v)
if not VIEWS:
    raise SystemExit('--view 没给视角')
VIEW_TAGS = [VIEW_NAMES[v] for v in VIEWS]

# 差值必须双向，配色默认 coolwarm；给了 --colormap 就听用户的
CMAP = args.colormap or 'coolwarm'
SUBJECTS_DIR = None
_MEASURE_FIG = {}


# %% ---- 2026-11-03 ------------------------
# Times, subjects, stc
def time_grid(times_all):
    '''
    要画的时刻，吸附到 stc 自己的采样点上。

    图上标的时刻必须就是数据里的时刻，否则 0.300 实际画的是 0.295。
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


def load_members(mode):
    '''
    读两组、逐被试配对。

    配对是硬要求：quick 与 slow 是同一名被试的两半试次，缺一半这名被试就整组扔掉。
    否则差值里混进了「换了几个人」。--allow-partial 只影响是否报错，配对本身不放松。

    Returns:
        members: quick / slow 两个 dict(subj -> 2D array)，原始单位
        refs: subj -> 参考幅度（两组分位数的均值，标量）
        times_all: 时间轴
        template: 第一个读到的 stc，只借它的顶点编号与时间轴
        missing: 缺文件或只有一半的被试
    '''
    wanted = subject_names(mode)
    if not wanted:
        raise SystemExit(f'{DATA_DIR} 下没有 {mode}-* 目录，stage 8 / 16 还没跑？')

    quick, slow, refs, missing = {}, {}, {}, []
    times_all, template = None, None
    for subj in wanted:
        stems = [stc_stem(mode, subj, args.epochs, t) for t in (TAG_QUICK, TAG_SLOW)]
        if any(s is None for s in stems):
            missing.append(subj)
            continue
        q = mne.read_source_estimate(str(stems[0]))
        s = mne.read_source_estimate(str(stems[1]))
        if q.data.shape != s.data.shape:
            raise SystemExit(
                f'{mode}-{subj} 两组形状不一致：{q.data.shape} 对 {s.data.shape}。'
                '顶点或时刻轴不同，说明两份 stc 不是同一套源空间/同一段时间。')
        if template is None:
            template = q.copy()
            times_all = np.asarray(q.times, float)
        qd = q.data.astype(np.float64)
        sd = s.data.astype(np.float64)
        # 参考幅度是逐被试的标量：用逐顶点做分母会在 |q|+|s| 近零处爆掉
        ref = 0.5 * (np.nanpercentile(np.abs(qd), args.ref_percent)
                     + np.nanpercentile(np.abs(sd), args.ref_percent))
        if not np.isfinite(ref) or ref <= 0:
            logger.warning(f'{mode}-{subj} 的参考幅度是 {ref}，跳过这名被试')
            missing.append(subj)
            continue
        quick[subj], slow[subj], refs[subj] = qd, sd, float(ref)

    if missing:
        logger.warning(f'{mode} 缺 {missing} 的 {" 或 ".join(TAGS)}，'
                       '这些被试不参与配对（只做差值的这一半不算数）')
    if not quick:
        raise SystemExit(
            f'{mode} 没有可配对的被试。找的是\n'
            f'  {DATA_DIR / (mode + "-*")}/{args.epochs}.{TAG_QUICK}-lh.stc\n'
            f'  {DATA_DIR / (mode + "-*")}/{args.epochs}.{TAG_SLOW}-lh.stc\n'
            '先确认这两组由哪个 stage 出：\n'
            '  ave-quick / ave-slow              -> 15.rt-source-contrast.sh\n'
            '  ave-quick-sharedcov / -slow-...   -> 16.rt-source-projection.sh')
    return dict(quick=quick, slow=slow), refs, times_all, template, missing


def norm_block(block, ref):
    '''
    把一块源强度换成要画的量。

    ref 模式下是「相对该被试参考幅度的百分比」；none 模式保留原始单位，只乘
    --data-scale。两种模式都不改数据的符号，差值的正负留到最后由对称色标表达。
    '''
    if args.normalize == 'none':
        return block * args.data_scale
    return 100.0 * block / ref


def build_rows(members, refs):
    '''
    要画的几组数据，都是逐被试算完再平均。

    成员行恒画幅度 |s|（两组各自的均值），差值行按 --measure 走。分母是同一个标量，
    所以 absdiff 时「成员行之差 == 差值行」在逐被试层面严格成立，图上可以直接对。

    Returns:
        rows: [dict(label, note_name, data), ...]；差值行永远在最后
        keys: 参与配对的被试
        diff: 逐被试的差值（已除参考幅度，未乘 100 之外的缩放）
    '''
    keys = sorted(members['quick'])

    if args.measure == 'absdiff':
        diff = {k: np.abs(members['quick'][k]) - np.abs(members['slow'][k])
                for k in keys}
    else:
        diff = {k: members['quick'][k] - members['slow'][k] for k in keys}

    rows = []
    if args.with_members:
        for name in ('quick', 'slow'):
            rows.append(dict(
                label=f'{name}\n|s| mean',
                note_name=name,
                data=np.mean([norm_block(np.abs(members[name][k]), refs[k])
                              for k in keys], axis=0)))

    rows.append(dict(
        label='quick - slow\n' + ('abs diff' if args.measure == 'absdiff'
                                 else 'signed diff'),
        note_name='diff',
        data=np.mean([norm_block(diff[k], refs[k]) for k in keys], axis=0)))
    return rows, keys, diff


# %% ---- 2026-11-03 ------------------------
# Colour scale
def pick_vmax(frames):
    '''
    定色标上限。

    取要显示的那些帧里 |值| 的分位数，不是整条时间轴：0.6 s 之后的漂移会让真正要看的
    这十几帧全平。差值是双向的，上限就是 ±vmax。
    '''
    if args.clim is not None:
        return abs(float(args.clim))
    vals = np.concatenate([np.abs(f).ravel() for f in frames])
    if vals.size == 0:
        return 0.
    vmax = float(np.nanpercentile(vals, args.clim_percent))
    if not np.isfinite(vmax) or vmax <= 0:
        logger.warning(f'vmax 算出来是 {vmax}，色标交给 mne')
        return 0.
    return vmax


def clim_dict(vmax):
    '''
    色标一定以 0 为中心。

    差值要表达「谁更大」，零点必须有意义，单侧写法会把负的一侧压成一片同色。成员行
    （|s| 均值）其实全正，这里也照双向走：同一张图里两套色标如果一单一双，coolwarm
    的颜色含义就变了（成员行的 0 会跑到色带最左端），读者会把「蓝」读成两组不同的
    意思。代价是成员行只用到色带的正半轴，可接受。
    '''
    if not vmax or vmax <= 0:
        return 'auto'
    return dict(kind='value', lims=[-vmax, 0., vmax])


# %% ---- 2026-11-03 ------------------------
# Layout
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
    全白的格子（某帧没有超过阈值的点）没有包围盒，退到第一个可用的盒子。
    '''
    if args.no_crop:
        return list(images)
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return list(images)
    r0 = min(b[0] for b in boxes)
    r1 = max(b[1] for b in boxes)
    c0 = min(b[2] for b in boxes)
    c1 = max(b[3] for b in boxes)
    out = []
    for im in images:
        out.append(im[max(r0 - pad, 0):r1 + pad + 1,
                      max(c0 - pad, 0):c1 + pad + 1])
    return out


def text_width_in(text, fontsize):
    '''
    一段文字有多宽（英寸），用来定左边的留白。

    用 Agg 的渲染器实测 get_window_extent。TextPath 按字形度量估出来的宽度比实际窄，
    偏窄就意味着一行「N=10 · max 0.3」的左端会被图边界裁掉。多行取最宽的一行。
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


def draw_bar(fig, cax, vmax, cmap, fontsize, unit=''):
    '''
    画一根双向色标条。

    刻度标两端与零点，写成科学计数法——MNE 逆解是 1e-14 量级，默认的
    「1e-15 偏移 + 5/0/-5」读不出量级；归一化之后是百分比，同一段代码照样能标。
    '''
    norm = Normalize(vmin=-vmax, vmax=vmax)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 orientation='vertical')
    cax.set_yticks([-vmax, 0., vmax])
    cax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:.2g}'))
    cax.tick_params(labelsize=fontsize, length=2)
    if unit:
        cax.set_title(unit, fontsize=fontsize)


def row_note(n_subjects, vmax, ratio=None, is_first=False):
    '''
    行标签下面的那行小字：N、色标上限、相对第一组的倍数。

    每组各自定色标，深浅不能跨组读，量级只能写在这里。
    '''
    text = f'N={n_subjects} · max {vmax:.3g}{UNIT}'
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
        labels: 每行的大字标签
        notes: 每行的小字（N 与 max）
        bars: 每行一个 dict(vmax, span, show)，span 是这根色标覆盖几行
        cmap: 与 stc.plot 用同一个 colormap
        fname: 输出文件名
        subtitle: 第二行小字，默认由参数拼
        tags: 每行的视角名（多视角时才有）
        groups: 每组一个 dict(start, span, label, note)。标签与量级只写在组的中线上
            一份，视角名写在各自行上——六行各写一遍 label 会挤成一团
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
    # 左边留白按最长的那行标签实测宽度算
    widest_label = max([text_width_in(v, fs) for v, fs in left_texts] or [0.])
    # 多视角时左边排成两列：组标签一列、视角名一列，两列宽度都要先量
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
        # 右边留白按最长的刻度文字实测，写死会被右边界切掉最后一位
        vals = [b['vmax'] for b in bars if b.get('vmax')] or [0.]
        texts = [f'{v:.2g}' for v in vals] + [f'{v / 2:.2g}' for v in vals] \
            + ['0', UNIT]
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
            # 多视角：视角名单独一列，紧贴脑图；组标签在它左边一列
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
            # 同一组的几个视角共用一根色标：bar 落在第一行上，span 覆盖它们
            span = int(bars[r].get('span', 1))
            base = row_y(r + span - 1)
            cax = fig.add_axes([(left + n_cols * col_w + 0.12) / W,
                                base + span * row_h / H * 0.12,
                                bar_w / W, span * row_h / H * 0.76])
            draw_bar(fig, cax, bars[r]['vmax'], cmap, fs_bar, UNIT)

    if groups:
        # 组标签与量级写在组的中线上，右对齐到「视角名那一列的左边」
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
        draw_bar(fig, cax, bars[0]['vmax'], cmap, fs_bar, UNIT)

    title = args.title or f'{mode} quick - slow source contrast'
    fig.text(0.5, (H - 0.10) / H, title, ha='center', va='top', fontsize=fs_title)
    view_txt = ' + '.join(VIEW_TAGS)
    fig.text(0.5, (H - 0.34) / H,
             subtitle or (f'{args.measure} / {args.normalize} | view {view_txt} | '
                          f'{n_rows} rows x {n_cols} frames | '
                          f'colour scale: {args.clim_mode}'),
             ha='center', va='top', fontsize=fs_sub, color='0.35')

    fig.savefig(fname, dpi=args.dpi, facecolor='white')
    plt.close(fig)
    logger.info(f'Saved into {fname}, {W:.1f} x {H:.1f} in')


# %% ---- 2026-11-03 ------------------------
# Render
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

    标题留空：每格的时刻标在拼图时写；colorbar 关掉，色标统一在拼图里画。
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

    rebuild 每帧重建一次脑图（稳）；reuse 建一次然后 set_time 逐帧截屏，快得多，但
    重绘链要是没生效就会截出一排同样子的图，所以这里自己验一下相邻帧，一旦相同就
    回退 rebuild。
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


# %% ---- 2026-11-03 ------------------------
# Self test
def run_self_test():
    '''
    合成图跑一遍排版、色标、裁边。

    不需要 stc、不需要 fsaverage、不需要 3D 后端，任何机器上都能跑，用来确认
    「拼图这一段」本身是对的——真正渲染的部分要上服务器才知道。
    '''
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    times = [round(i * 0.1, 3) for i in range(11)]
    vmaxes = [40.0, 25.0, 12.0] if args.with_members else [40.0]
    labels = (['quick\nabs', 'slow\nabs'] if args.with_members else []) \
        + ['quick - slow\nabs diff']
    notes = [f'N=10 · max {v:.3g}{UNIT}' for v in vmaxes]

    panels = []
    rr, cc = np.ogrid[:110, :240]
    for v in vmaxes:
        for _view in VIEWS:          # 组 × 视角，和主流程排法一致
            row = []
            for c, _ in enumerate(times):
                img = np.full((110, 240, 3), 255, np.uint8)
                for k, (yy, xx) in enumerate([(40, 70), (58, 150), (78, 200)]):
                    hot = 180 + 40 * np.sin(c / 2. + k)
                    m = ((rr - yy) ** 2 / 900. + (cc - xx) ** 2 / 1600.) < 1
                    img[m] = [min(hot, 255), 80, 255 - hot]
                row.append(img)
            panels.append(row)

    if args.colorbar == 'auto':
        args.colorbar = 'row' if args.clim_mode == 'row' else 'shared'
    multi = len(VIEWS) > 1

    labels_out, notes_out, tags_out, bars_out = [], [], [], []
    groups = []
    for i, v in enumerate(vmaxes):
        note = row_note(10, v, v / vmaxes[0], is_first=(i == 0))
        for j, view in enumerate(VIEWS):
            labels_out.append(labels[i])
            notes_out.append(note)
            tags_out.append(VIEW_NAMES[view] if multi else '')
            bars_out.append(dict(vmax=v, span=len(VIEWS), show=(j == 0)))
        groups.append(dict(start=i * len(VIEWS), span=len(VIEWS),
                           label=labels[i], note=note))

    flat = [im for row in panels for im in row]
    cropped = apply_crop(flat, [crop_box(im) for im in flat])
    n = len(times)
    n_rows = len(vmaxes) * len(VIEWS)
    panels = [cropped[r * n:(r + 1) * n] for r in range(n_rows)]

    assemble('SELF', panels, times, labels_out, notes_out, bars_out,
             plt.get_cmap(CMAP), OUTPUT_DIR / 'source-quick-slow-SELF-TEST.png',
             subtitle=f'self test | {args.measure} / {args.normalize}',
             tags=tags_out, groups=groups if multi else None)
    logger.info(f'clim 一定是双向的：{clim_dict(vmaxes[0])}')
    logger.info(f'colormap: {CMAP}, colorbar: {args.colorbar}, '
                f'mode: {args.clim_mode}')


# %% ---- 2026-11-03 ------------------------
# Main
def main():
    '''
    主流程：读两组 stc -> 逐被试配对差值 -> 群体平均 -> 定色标 -> 渲染 -> 拼图。

    色标要看完所有组才能定（--clim-mode shared），所以分两段：先把每组的数据与要
    显示的帧拿全，再逐组渲染。
    '''
    global SUBJECTS_DIR

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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary, timeline, persub = [], [], []
    for mode in MODES:
        members, refs, times_all, template, _ = load_members(mode)
        rows, keys, diff = build_rows(members, refs)
        times = time_grid(times_all)
        idx = [int(np.argmin(np.abs(times_all - t))) for t in times]
        logger.info(f'{mode}: {len(times)} 帧 '
                    f'{times[0]:.3f}-{times[-1]:.3f} s，N = {len(keys)}, '
                    f'{TAG_QUICK} - {TAG_SLOW}')

        for r in rows:
            frames = r['data'][:, idx]
            r['vmax_own'] = pick_vmax([frames])
            r['data_min'] = float(np.nanmin(frames))
            r['data_max'] = float(np.nanmax(frames))

        if args.clim_mode == 'shared':
            shared = pick_vmax([r['data'][:, idx] for r in rows])
            logger.info(f'{mode} 全图共用一个 vmax = ±{shared:.4g}{UNIT}')
            for r in rows:
                r['vmax_used'] = shared
        else:
            for r in rows:
                r['vmax_used'] = r['vmax_own']

        first = rows[0]['vmax_used'] or np.nan
        for i, r in enumerate(rows):
            r['ratio'] = r['vmax_used'] / first
            logger.info(f'{mode} row {i + 1} '
                        f'{r["label"].replace(chr(10), "/")}: '
                        f'{len(keys)} 被试, vmax ±{r["vmax_used"]:.4g}{UNIT}, '
                        f'{r["ratio"]:.2f}x row1, 数据范围 '
                        f'[{r["data_min"]:.4g}, {r["data_max"]:.4g}]')

        # 差值逐时刻的成绩单：哪一刻差别最大、正差顶点占多少
        for t_i, t in enumerate(times):
            col = rows[-1]['data'][:, idx[t_i]]
            timeline.append(dict(
                mode=mode, time=f'{t:.3f}',
                mean_abs=f'{np.mean(np.abs(col)):.4g}',
                p95_abs=f'{np.nanpercentile(np.abs(col), 95):.4g}',
                frac_pos=f'{np.mean(col > 0):.4f}',
                frac_over_10pct=f'{np.mean(np.abs(col) > 10.):.4f}'))

        # 一行 = 一组的一个视角。同一组的几个视角连着排，共用一根色标。
        plan = [(i, view) for i in range(len(rows)) for view in VIEWS]
        multi = len(VIEWS) > 1

        panels, labels_out, notes_out, tags_out, bars_out = [], [], [], [], []
        groups = []
        for k, (i, view) in enumerate(plan):
            r = rows[i]
            if r['data'].shape != template.data.shape:
                raise SystemExit(
                    f'{mode} row {i + 1} 的数据是 {r["data"].shape}，'
                    f'模板是 {template.data.shape}，顶点或时刻对不上')
            stc = template.copy()
            stc.data = r['data']
            stc.subject = 'fsaverage'
            clim = clim_dict(r['vmax_used'])

            note = row_note(len(keys), r['vmax_used'], r['ratio'], is_first=(i == 0))
            labels_out.append(r['label'])
            notes_out.append(note)
            tags_out.append(VIEW_NAMES[view] if multi else '')
            bars_out.append(dict(vmax=r['vmax_used'], span=len(VIEWS),
                                 show=(view == VIEWS[0])))
            if view == VIEWS[0]:
                groups.append(dict(start=k, span=len(VIEWS),
                                   label=r['label'], note=note))

            logger.info(f'{mode} row {k + 1} '
                        f'{r["label"].replace(chr(10), "/")} · '
                        f'{VIEW_NAMES[view]} clim = {clim}')

            # 这一段全是数据侧的读数，与渲不渲染无关，所以 dry-run 也写，
            # 服务器上可以先看数再决定 --clim / --clim-percent
            summary.append(dict(
                mode=mode, row=k + 1, name=r['note_name'],
                label=r['label'].replace('\n', ' '),
                view=VIEW_NAMES[view], epochs_fname=args.epochs,
                tags=' '.join(TAGS), n_subjects=len(keys),
                subjects=' '.join(keys), measure=args.measure,
                normalize=args.normalize, hemi=args.hemi,
                times=' '.join(f'{t:.3f}' for t in times),
                vmax=f'{r["vmax_used"]:.6g}',
                vmax_vs_row1=f'{r["ratio"]:.4f}'
                if np.isfinite(r['ratio']) else '',
                data_min=f'{r["data_min"]:.6g}',
                data_max=f'{r["data_max"]:.6g}',
                clim_mode=args.clim_mode, colormap=CMAP))

            if args.dry_run:
                # 浅灰占位格，只验版式：行标签、时刻、色标位置都能先看一眼
                hh, ww = args.panel_size[1], args.panel_size[0]
                blank = np.full((hh, ww, 3), 248, np.uint8)
                blank[[0, -1], :, :] = 190
                blank[:, [0, -1], :] = 190
                panels.append([blank.copy() for _ in times])
                continue

            imgs = render_row(stc, clim, times, args.render_mode, view)
            panels.append(imgs)
            if args.save_panels:
                folder = OUTPUT_DIR / f'panels-{mode}'
                folder.mkdir(parents=True, exist_ok=True)
                for im, t in zip(imgs, times):
                    plt.imsave(folder / f'row{k + 1}-{VIEW_NAMES[view]}-'
                                        f'{t:.3f}s.png', im)

            summary.append(dict(
                mode=mode, row=k + 1, name=r['note_name'],
                label=r['label'].replace('\n', ' '),
                view=VIEW_NAMES[view], epochs_fname=args.epochs,
                tags=' '.join(TAGS), n_subjects=len(keys),
                subjects=' '.join(keys), measure=args.measure,
                normalize=args.normalize, hemi=args.hemi,
                times=' '.join(f'{t:.3f}' for t in times),
                vmax=f'{r["vmax_used"]:.6g}',
                vmax_vs_row1=f'{r["ratio"]:.4f}'
                if np.isfinite(r['ratio']) else '',
                data_min=f'{r["data_min"]:.6g}',
                data_max=f'{r["data_max"]:.6g}',
                clim_mode=args.clim_mode, colormap=CMAP))

        # 逐被试读数单独一张表：参考幅度、差值分位、正差顶点占比
        if args.with_members and args.measure == 'absdiff':
            # 成员行之差应严格等于差值行（同一个分母、逐被试线性），出偏就是算错了
            gap = float(np.max(np.abs(rows[-1]['data']
                                      - (rows[0]['data'] - rows[1]['data']))))
            logger.info(f'{mode} 自洽检查：成员行之差与差值行的最大出入 '
                        f'{gap:.3g}{UNIT}')
        for k in keys:
            d = norm_block(diff[k], refs[k])
            persub.append(dict(
                mode=mode, subject=k,
                ref=f'{refs[k]:.6g}',
                diff_p99=f'{np.nanpercentile(np.abs(d), 99):.4g}',
                diff_min=f'{np.nanmin(d):.4g}',
                diff_max=f'{np.nanmax(d):.4g}',
                frac_pos=f'{np.mean(d > 0):.3f}',
                frac_over_10=f'{np.mean(np.abs(d) > 10.):.3f}'))

        flat = [im for row in panels for im in row]
        cropped = apply_crop(flat, [crop_box(im) for im in flat])
        n = len(times)
        panels = [cropped[r * n:(r + 1) * n] for r in range(len(plan))]

        suffix = '-DRYRUN' if args.dry_run else ''
        assemble(mode, panels, times, labels_out, notes_out, bars_out,
                 plt.get_cmap(CMAP),
                 OUTPUT_DIR / f'source-quick-slow-{mode}{suffix}.png',
                 tags=tags_out, groups=groups if multi else None)
        if args.dry_run:
            logger.info(f'{mode} dry-run 到此为止：没有渲染，上面这张是空格的版式图')

    if not summary:
        logger.info('这几种模态都没有可配对的数据，什么也没写')
        return
    df = pd.DataFrame(summary)
    print(df.head(20).to_string(index=False))
    fname = OUTPUT_DIR / 'source-quick-slow-summary.csv'
    df.to_csv(fname, index=False)
    logger.info(f'Saved into {fname}')
    if timeline:
        tf = pd.DataFrame(timeline)
        tfname = OUTPUT_DIR / 'source-quick-slow-timeline.csv'
        tf.to_csv(tfname, index=False)
        logger.info(f'Saved into {tfname}')
    if persub:
        pf = pd.DataFrame(persub)
        pfname = OUTPUT_DIR / 'source-quick-slow-persubject.csv'
        pf.to_csv(pfname, index=False)
        logger.info(f'Saved into {pfname}')


# %% ---- 2026-11-03 ------------------------
# Play ground
if args.self_test:
    run_self_test()
else:
    main()


# %% ---- 2026-11-03 ------------------------
# Pending
# 1. 图上没有统计掩膜。哪一块差值是真的要回 15 的置换检验（两模态都没过 FDR），
#    这一支只负责把差值的形态与时间进程画出来。
# 2. 差值的双向色标必然浪费一半范围（多数时刻一侧为空心）。要看单侧幅度差可以把
#    --measure diff --normalize none 之后自己截正半轴，但零点就没意义了。
# 3. 俯视图看的是朝上的那一面，沟里的活动读不到；要看深部得换 --cortex 或改 volume 源。
