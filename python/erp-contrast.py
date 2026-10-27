"""
File: erp-contrast.py
Author: Chuncheng Zhang
Date: 2026-10-27

Purpose:
    target ERP 与 target - non-target 差异波，在投影前后的可直接对照图。

    stage 5 的图（output/ERP/{MODE}/all-{evt}-*.png）用的是 MNE plot_joint，
    拿来做比较有四个问题：
      1. 地形图帧时刻按 evt 各自选：target 固定取 5 帧，non-target 与 keypress
         走 'peaks' 自动选峰，投影前后、target/non-target 之间都不可比；
      2. 下排是蝴蝶图（全部通道叠画），看不见全局场功率，也读不出成分；
      3. 只有 5 帧，成分的时间演化看不出来；
      4. 投影前与投影后分成两张图，色标各自缩放，无法目测削减了多少。

    本脚本固定时间网格、两列共享色标、下排换成 GFP 与高负荷通道波形。

Inputs:
    output/ERP/{MODE}/all-1-{suffix}-ave.ave.fif        target 群体加权平均
    output/ERP/{MODE}/all-2-{suffix}-ave.ave.fif        non-target 群体加权平均
    suffix: notch（投影前）| notch-removal-artificial（投影后）
    这四个文件由 python/average-subjects-erp.py 写出，本脚本只读不重算。

Outputs:
    output/ERP/{MODE}/erp-contrast-target-{MODE}.png
    output/ERP/{MODE}/erp-contrast-diff-{MODE}.png

Usage:
    python python/erp-contrast.py --mode EEG --analysis both
    python python/erp-contrast.py --mode MEG --analysis diff --times-step 0.04

Notes:
    - 带通沿用 stage 5 的 0.1-5 Hz，基线沿用 (-0.2, 0) s，与已有图可比。
    - 两列共享色标是刻意的：投影后的幅度本来就该看着小，这才是要传达的信息。
    - 单位按通道类型换算（EEG -> uV，mag -> fT，grad -> fT/cm），图上标注。
    - 图内文字用英文，避免服务器缺中文字体。
"""


# %% ---- 2026-10-27 ------------------------
# Requirements and constants
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger('erp-contrast')

SCALE = dict(eeg=1e6, mag=1e15, grad=1e13, seeg=1e6, ecog=1e6)
UNIT = dict(eeg='uV', mag='fT', grad='fT/cm', seeg='uV', ecog='uV')

# 投影前 / 后：tag、文件后缀、图例文字
SPECS = [
    ('norp', 'notch', 'before projection'),
    ('rma', 'notch-removal-artificial', 'after projection'),
]
LS = {'norp': '-', 'rma': '--'}
COLOR = {'norp': '#C0392B', 'rma': '#1F4E9C'}
PALETTE = ['#C0392B', '#1F4E9C', '#1E8449', '#8E44AD', '#B9770E', '#117A65']

parser = argparse.ArgumentParser(description='target ERP / 差异波的投影前后对照图')
parser.add_argument('--mode', default='EEG', help='EEG | MEG')
parser.add_argument('--analysis', default='both',
                    choices=['target', 'diff', 'both'],
                    help='target 只看 target 平均；diff 看 target - non-target')
parser.add_argument('--erp-dir', default='output/ERP',
                    help='stage 5 写出 ave.fif 的根目录')
parser.add_argument('--epochs-dir', default='output/epochs',
                    help='ave.fif 缺失时从这里按试次数加权重算')
parser.add_argument('--times-start', type=float, default=0.10,
                    help='地形图网格起点（s）')
parser.add_argument('--times-stop', type=float, default=0.58,
                    help='地形图网格终点（s），含')
parser.add_argument('--times-step', type=float, default=0.08,
                    help='地形图网格步长（s）；0.08 从 0.10 到 0.58 共 7 帧')
parser.add_argument('--l-freq', type=float, default=0.1, help='高通（Hz）')
parser.add_argument('--h-freq', type=float, default=5.0, help='低通（Hz）')
parser.add_argument('--baseline', type=float, nargs=2, default=[-0.2, 0.0],
                    help='基线窗（s）')
parser.add_argument('--n-chan', type=int, default=4,
                    help='下排画几个高负荷传感器')
parser.add_argument('--hot-win', type=float, nargs=2, default=[0.24, 0.36],
                    help='按哪个时间窗挑高负荷传感器')
parser.add_argument('--xlim', type=float, nargs=2, default=[-0.2, 0.9],
                    help='曲线面板的时间范围（s）')
parser.add_argument('--dpi', type=int, default=150)
parser.add_argument('--vlim-percentile', type=float, default=99.0,
                    help='色标取 |值| 的哪个分位数，越小越不容易被单个极值压平')
parser.add_argument('--no-sensors', action='store_true',
                    help='地形图上不画传感器点（MEG 272 点太密时用）')
args = parser.parse_args()

MODE = args.mode.upper()
ERP_DIR = Path(args.erp_dir) / MODE
EPOCHS_DIR = Path(args.epochs_dir)


# %% ---- 2026-10-27 ------------------------
# Function and class
def average_from_epochs(evt, suffix):
    """
    退回方案：在没有 ave.fif 时，从逐被试 epochs 按试次数加权平均。
    注意 ave.fif 里的 nave 是第一个被试的试次数（stage 5 的存法），
    这里给出的是全部被试的试次总数，图标题上会写清楚。
    """
    pattern = f'{MODE}-S*/epochs-{evt}-{suffix}-epo.fif'
    files = sorted(EPOCHS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f'既没有 ave.fif，也找不到 {EPOCHS_DIR / pattern}。'
            f'先跑 5.average-subjects-erp.sh，或确认 --epochs-dir 指向正确。')
    evokeds, numbers = [], []
    for fname in files:
        epochs = mne.read_epochs(fname, preload=True, verbose='error')
        ev = epochs.average()
        evokeds.append(ev)
        numbers.append(ev.nave)
        logger.info(f'  {fname.parent.name}: {ev.nave} trials')
    total = float(np.sum(numbers))
    data = sum(ev.data * (n / total) for ev, n in zip(evokeds, numbers))
    out = evokeds[0].copy()
    out.data = data
    out.nave = int(total)
    logger.info(f'rebuilt {evt}/{suffix} from {len(files)} subjects, '
                f'{int(total)} trials total')
    return out


def load_pair(evt):
    """读入同一条事件在投影前后的群体加权平均。优先 ave.fif。"""
    out = {}
    for tag, suffix, _ in SPECS:
        fname = ERP_DIR / f'all-{evt}-{suffix}-ave.ave.fif'
        if fname.exists():
            ev = mne.read_evokeds(fname, verbose='error')[0]
            logger.info(f'read {fname.name}: nave={ev.nave}, '
                        f'{ev.times[0]:.2f}~{ev.times[-1]:.2f} s, '
                        f'{len(ev.ch_names)} channels')
        else:
            logger.warning(f'{fname.name} 不存在，从 epochs 重算')
            ev = average_from_epochs(evt, suffix)
        out[tag] = ev
    return out


def prep(ev):
    """带通 + 基线。返回副本，不改动读进来的对象。"""
    ev = ev.copy()
    ev.filter(l_freq=args.l_freq, h_freq=args.h_freq, verbose='error')
    mask = (ev.times >= args.baseline[0]) & (ev.times <= args.baseline[1])
    if mask.sum() == 0:
        logger.warning(f'基线窗 {args.baseline} 落在数据之外，跳过')
    else:
        ev.data = ev.data - ev.data[:, mask].mean(axis=1, keepdims=True)
    return ev


def channel_scale(ev):
    """逐通道量纲换算因子。混通道类型时报警，因为单位无法统一。"""
    factors = np.array([SCALE.get(t, 1.0) for t in ev.get_channel_types()])
    units = sorted({UNIT.get(t, '?') for t in ev.get_channel_types()})
    if len(units) > 1:
        logger.warning(f'通道类型混合 {sorted(set(ev.get_channel_types()))}，'
                       f'图上单位不统一，读数时注意')
        return factors, 'mixed'
    return factors, units[0]


def gfp(data):
    """全局场功率，跨通道 RMS。"""
    return np.sqrt((data ** 2).mean(axis=0))


def nearest(times, t):
    return int(np.argmin(np.abs(times - t)))


def draw(fname, analysis, frames, info, unit_label, frame_times, all_times,
         gfp_curves, chan_names, chan_series, nave_note):
    """
    画一张对照图。
    行 1：投影前地形图序列
    行 2：投影后地形图序列（与行 1 同色标、同时刻，可直接上下对看）
    行 3：GFP
    行 4：高负荷通道波形（颜色区分通道，实线投影前、虚线投影后）
    """
    n_frame = len(frame_times)
    per_frame = 1.55
    fig_w = 1.7 + per_frame * n_frame
    fig_h = per_frame * 2 + 3.7
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        4, 1, height_ratios=[per_frame, per_frame, 1.95, 1.75], hspace=0.34,
        left=0.055, right=0.90, top=0.895, bottom=0.075)

    # 共享色标：跨两行、跨全部帧一起取分位数
    stack = [frames[tag][:, [nearest(all_times, t) for t in frame_times]]
             for tag, _, _ in SPECS]
    stack = np.concatenate(stack, axis=1)
    vmax = float(np.percentile(np.abs(stack), args.vlim_percentile)) or 1.0
    vlim = (-vmax, vmax)
    logger.info(f'vlim = +/-{vmax:.4g} {unit_label} '
                f'(p{args.vlim_percentile:g} of |value|, both rows)')

    sm = ScalarMappable(norm=Normalize(*vlim), cmap='RdBu_r')
    sm.set_array([])
    cax = fig.add_axes([0.925, 0.70, 0.012, 0.19])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(f'{unit_label}', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    for row, (tag, _, label) in enumerate(SPECS):
        inner = outer[row].subgridspec(1, n_frame, wspace=0.04)
        d = frames[tag]
        for i, t in enumerate(frame_times):
            ax = fig.add_subplot(inner[0, i])
            mne.viz.plot_topomap(
                d[:, nearest(all_times, t)], info, axes=ax, show=False,
                vlim=vlim, cmap='RdBu_r', contours=0,
                sensors=False if args.no_sensors else 'k.', res=64)
            ax.set_title(f'{t:.2f} s' if i else f'{label}\n{t:.2f} s',
                         fontsize=9, pad=3)

    low = outer[2:].subgridspec(2, 1, hspace=0.55)
    t = all_times

    ax = fig.add_subplot(low[0])
    for c in gfp_curves:
        ax.plot(t, c['y'], color=c['color'], ls=c['ls'], lw=c['lw'],
                label=c['label'])
    ax.axhline(0, color='0.7', lw=0.5)
    ax.set_xlim(args.xlim)
    ax.set_ylabel(f'GFP ({unit_label})', fontsize=10)
    ax.set_title('global field power', fontsize=10, loc='left')
    ax.legend(fontsize=8, ncol=4, frameon=False, loc='upper right')
    ax.tick_params(labelsize=8)
    for t0 in frame_times:
        ax.axvline(t0, color='0.85', lw=0.4, zorder=0)

    ax = fig.add_subplot(low[1])
    handles = []
    for k, ch in enumerate(chan_names):
        color = PALETTE[k % len(PALETTE)]
        for tag, _, label in SPECS:
            ax.plot(t, chan_series[ch][tag], color=color, ls=LS[tag], lw=1.3)
        handles.append(Line2D([], [], color=color, ls='-', lw=1.3,
                              label=f'{ch}  before'))
        handles.append(Line2D([], [], color=color, ls='--', lw=1.3,
                              label=f'{ch}  after'))
    ax.axhline(0, color='0.7', lw=0.5)
    ax.set_xlim(args.xlim)
    ax.set_xlabel('time (s)', fontsize=10)
    ax.set_ylabel(f'amplitude ({unit_label})', fontsize=10)
    ax.set_title(f'top {len(chan_names)} sensors by |value| in '
                 f'{args.hot_win[0]:.2f}-{args.hot_win[1]:.2f} s',
                 fontsize=10, loc='left')
    ax.tick_params(labelsize=8)
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc='upper right',
              ncol=min(len(handles), 4))
    for t0 in frame_times:
        ax.axvline(t0, color='0.85', lw=0.4, zorder=0)

    title = {'target': 'target ERP',
             'diff': 'target - non-target difference'}[analysis]
    fig.suptitle(f'{MODE}   {title}   [{args.l_freq:g}-{args.h_freq:g} Hz, '
                 f'baseline {args.baseline[0]:g}~{args.baseline[1]:g} s, '
                 f'{nave_note}]', fontsize=12, y=0.985)
    fig.savefig(fname, dpi=args.dpi, facecolor='white')
    plt.close(fig)
    logger.info(f'wrote {fname}')


def run(analysis, target, nontarget, factors, unit_label, info, picks,
        frame_times, all_times):
    if analysis == 'target':
        frames = {tag: target[tag].data * factors[:, None] for tag in target}
        nave_note = f'target nave={target["norp"].nave}'
    else:
        frames = {tag: (target[tag].data - nontarget[tag].data) * factors[:, None]
                  for tag in target}
        nave_note = (f'target nave={target["norp"].nave} vs '
                     f'non-target nave={nontarget["norp"].nave}')

    gfp_curves = []
    for tag, _, label in SPECS:
        which = 'before projection' if tag == 'norp' else 'after projection'
        gfp_curves.append(dict(y=gfp(frames[tag]), color=COLOR[tag], ls=LS[tag],
                               lw=1.7, label=f'{analysis}  {which}'))
        if analysis == 'target':
            gfp_curves.append(dict(
                y=gfp(nontarget[tag].data * factors[:, None]),
                color='0.55', ls=LS[tag], lw=1.0,
                label=f'non-target  {which}'))

    mask = (all_times >= args.hot_win[0]) & (all_times <= args.hot_win[1])
    if mask.sum() == 0:
        mask = np.ones_like(all_times, dtype=bool)
    score = np.abs(frames['norp'][picks][:, mask]).max(axis=1)
    hot = picks[np.argsort(score)[::-1][:args.n_chan]]
    chan_names = [info['ch_names'][c] for c in hot]
    chan_series = {info['ch_names'][c]: {tag: frames[tag][c] for tag in frames}
                   for c in hot}
    logger.info(f'hot channels: {chan_names}')

    fname = ERP_DIR / f'erp-contrast-{analysis}-{MODE}.png'
    draw(fname, analysis, frames, info, unit_label, frame_times, all_times,
         gfp_curves, chan_names, chan_series, nave_note)

    # 打印读数，便于写稿时直接引用
    logger.info(f'--- {analysis} ---')
    for tag, _, label in SPECS:
        d = frames[tag]
        g = gfp(d)
        for t0 in frame_times:
            j = nearest(all_times, t0)
            logger.info(f'  {label:18s} t={t0:.2f}  GFP={g[j]:+9.3f}  '
                        f'max|ch|={np.abs(d[:, j]).max():+9.3f} {unit_label}')


# %% ---- 2026-10-27 ------------------------
# Play ground
def main():
    target = {tag: prep(ev) for tag, ev in load_pair('1').items()}
    nontarget = {tag: prep(ev) for tag, ev in load_pair('2').items()}

    factors, unit_label = channel_scale(target['norp'])
    all_times = target['norp'].times
    info = target['norp'].info
    picks = mne.pick_types(info, meg=True, eeg=True, exclude='bads')
    logger.info(f'{len(picks)} data channels, unit {unit_label}')

    frame_times = np.round(
        np.arange(args.times_start, args.times_stop + 1e-9, args.times_step), 4)

    for analysis in (['target', 'diff'] if args.analysis == 'both'
                     else [args.analysis]):
        run(analysis, target, nontarget, factors, unit_label, info, picks,
            frame_times, all_times)


if __name__ == '__main__':
    main()
