#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage 7（EEGNetV4）结果的投影前后对照汇总。

读 output/neuralnetwork-decode/{MODE}-S{XX}-EEGNetV4[-RA]/metrics.json，
按「模态 × 被试 × 条件」汇总逐被试指标，做配对比较（Wilcoxon 符号秩），
出一张配对散点图与两张汇总 csv。

关于条件的命名：
    目录名无 -RA 后缀  ->  remove_artificial = false  ->  投影前（before）
    目录名带 -RA 后缀  ->  remove_artificial = true   ->  投影后（after）

必须在统计里交代的两条设计事实（决定了差值能说到什么程度）：

1. 单次运行，无重复。eegnetv4-decode.py 全文没有设置随机种子，
   train_test_split 的 random_state 被注释掉，两次运行的测试集划分
   并不相同（y_true 序列不同，只是分层比例相同使计数相同）。
   因此单个被试的前后差里混着「划分差异 + 权重初始化 + shuffle + dropout」，
   不能把单次差直接读成投影的效应。只有跨 10 名被试系统同向的差异才有意义。

2. 每个目录各自按训练集做 z-score（normalization.npz）。投影后 std 大约
   降到投影前的一半，意味着整体幅度被各自归一掉了，网络看到的是形状。
   所以这组结果回答的是「去掉运动子空间之后，形状里的目标信息还剩多少」，
   幅度上的削减在这里读不出来。

另外注意数据裁剪：脚本里 epochs 被 crop 到 0.0–0.6 s，完整覆盖按键窗。
"""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)-7s %(message)s')
logger = logging.getLogger('summarize-eegnet')

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(
    description='汇总 stage 7 EEGNetV4 的投影前后分类表现')
parser.add_argument('--decode-dir', default='output/neuralnetwork-decode',
                    help='stage 7 的产物根目录')
parser.add_argument('--out-dir', default='output/eegnet-summary',
                    help='汇总输出目录')
parser.add_argument('--dpi', type=int, default=150)
args = parser.parse_args()

DECODE_DIR = ROOT / args.decode_dir
OUT_DIR = ROOT / args.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 关注的指标。recall 即 target 类的召回率（用户关心的目标命中率），
# specificity 是非目标类的召回率。balanced_accuracy 是二者的平均。
METRICS = ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1',
           'precision', 'recall']

# 图上用的短名
LABEL = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced acc',
    'roc_auc': 'ROC AUC',
    'f1': 'F1',
    'precision': 'precision',
    'recall': 'recall (target)',
}

COND = {'norp': 'before', 'rma': 'after'}


def check_auc(d, m):
    """从 test_predictions.npz 重算 AUC，确认 metrics.json 的口径。"""
    from sklearn.metrics import roc_auc_score
    z = np.load(d / 'test_predictions.npz')
    auc = roc_auc_score(z['y_true'], z['y_prob'])
    if abs(auc - m['roc_auc']) > 1e-5:
        logger.warning(f'{d.name}: roc_auc 重算不一致 '
                       f'({m["roc_auc"]:.5f} vs {auc:.5f})')
    return auc, z['y_true']


def load_one(d):
    """读一个被试一个条件的指标，顺带从混淆矩阵反算 specificity。"""
    m = json.load(open(d / 'metrics.json'))
    cfg = json.load(open(d / 'config.json'))
    cm = np.array(m['confusion_matrix'])
    # cm[0] 是非目标行 (TN, FP)，cm[1] 是目标行 (FN, TP)
    tn, fp = cm[0]
    fn, tp = cm[1]
    auc_check, ytrue = check_auc(d, m)
    row = {
        'mode': m['mode'],
        'subject': m['subject'],
        'condition': COND['rma' if cfg['remove_artificial'] else 'norp'],
        'remove_artificial': bool(cfg['remove_artificial']),
        'n_epochs': m['n_epochs'],
        'n_test': m['n_test'],
        'n_test_target': m['n_test_target'],
        'n_test_non_target': m['n_test_non_target'],
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'specificity': tn / (tn + fp) if (tn + fp) else np.nan,
        # 测试集划分的指纹：用来判断两次运行是否落在同一批试次上
        'split_fp': hashlib.md5(np.ascontiguousarray(
            ytrue).tobytes()).hexdigest()[:12],
    }
    for k in METRICS:
        row[k] = m[k]
    # 口径自检：balanced_accuracy 应当等于 (recall + specificity) / 2
    ba = (row['recall'] + row['specificity']) / 2
    if abs(ba - m['balanced_accuracy']) > 1e-6:
        logger.warning(f'{d.name}: balanced_accuracy 口径不一致 '
                       f'({m["balanced_accuracy"]:.4f} vs {ba:.4f})')
    return row


def main():
    dirs = sorted(p for p in DECODE_DIR.iterdir()
                  if p.is_dir() and (p / 'metrics.json').exists())
    if not dirs:
        raise SystemExit(f'{DECODE_DIR} 下没有找到 metrics.json')
    logger.info(f'读到 {len(dirs)} 个结果目录')

    rows = [load_one(d) for d in dirs]
    df = pd.DataFrame(rows).sort_values(
        ['mode', 'subject', 'condition']).reset_index(drop=True)

    # 配对检查：每个 (mode, subject) 必须两条都在
    grp = df.groupby(['mode', 'subject']).size()
    bad = grp[grp != 2]
    if len(bad):
        raise SystemExit(f'配对不完整：\n{bad}')
    logger.info(f'配对完整：{len(grp)} 个 (模态, 被试) 组合，各 2 条')

    # 划分一致性检查：两次运行的测试集是否为同一批试次。
    # 计数相同只说明分层比例相同，不代表样本相同，所以用 y_true 的指纹判定。
    wide_fp = df.pivot_table(index=['mode', 'subject'], columns='condition',
                             values='split_fp', aggfunc='first')
    same = (wide_fp['before'] == wide_fp['after'])
    n_same = int(same.sum())
    logger.info(f'测试集划分指纹相同：{n_same}/{len(same)}')
    if n_same < len(same):
        logger.warning('划分不同 -> 每个配对差里含有「划分差异 + 训练随机性」，'
                       '单次运行的幅度不能直接当作投影的效应量')

    per_subj = OUT_DIR / 'eegnet-per-subject.csv'
    df.to_csv(per_subj, index=False, encoding='utf-8-sig')
    logger.info(f'wrote {per_subj}')

    # ---- 配对比较 --------------------------------------------------
    wide = df.pivot_table(index=['mode', 'subject'],
                          columns='condition', values=METRICS + ['specificity'])
    recs = []
    for mode in ('EEG', 'MEG'):
        sub = wide.xs(mode, level='mode')
        for k in METRICS + ['specificity']:
            b = sub[(k, 'before')].to_numpy(float)
            a = sub[(k, 'after')].to_numpy(float)
            d = a - b
            try:
                w = stats.wilcoxon(a, b, zero_method='wilcox')
                p = w.pvalue
                stat = w.statistic
            except ValueError:
                p, stat = np.nan, np.nan
            recs.append({
                'mode': mode, 'metric': k,
                'before_median': float(np.median(b)),
                'after_median': float(np.median(a)),
                'before_mean': float(np.mean(b)),
                'after_mean': float(np.mean(a)),
                'delta_median': float(np.median(d)),
                'delta_mean': float(np.mean(d)),
                'delta_sd': float(np.std(d, ddof=1)),
                'dz_paired': (float(np.mean(d) / np.std(d, ddof=1))
                              if np.std(d, ddof=1) > 0 else np.nan),
                'n_subjects': len(d),
                'n_after_lower': int((d < 0).sum()),
                'n_after_higher': int((d > 0).sum()),
                'wilcoxon_stat': float(stat) if stat == stat else np.nan,
                'p': float(p) if p == p else np.nan,
            })
    cmp_df = pd.DataFrame(recs)
    cmp_path = OUT_DIR / 'eegnet-paired-compare.csv'
    cmp_df.to_csv(cmp_path, index=False, encoding='utf-8-sig')
    logger.info(f'wrote {cmp_path}')

    for mode in ('EEG', 'MEG'):
        logger.info(f'--- {mode} ---')
        for r in cmp_df[cmp_df['mode'] == mode].to_dict('records'):
            logger.info(f"  {r['metric']:<18s} before {r['before_median']:.4f}"
                        f"  after {r['after_median']:.4f}"
                        f"  delta {r['delta_median']:+.4f}"
                        f"  sd {r['delta_sd']:.4f}  dz {r['dz_paired']:+.2f}"
                        f"  {r['n_after_lower']}/{r['n_subjects']} 下降"
                        f"  p={r['p']:.4f}")

    # ---- 与滑窗解码的降幅对比 --------------------------------------
    # 两个量测的不是同一件事：
    #   EEGNet      整段 0–0.6 s 一起进网络，测「全部可用信息」；
    #   sliding     逐时间点单独解码，再跨时间取最大，测「最好的那一时刻」。
    sliding_csv = ROOT / 'output/sliding-decode/group-decode-summary.csv'
    if sliding_csv.exists():
        sl = pd.read_csv(sliding_csv)
        sl['condition'] = sl['condition'].map({'withoutRMA': 'before',
                                               'withRMA': 'after'})
        auc_p = sl.pivot_table(index='mode', columns='condition',
                               values='peak_auc')
        tim_p = sl.pivot_table(index='mode', columns='condition',
                               values='peak_time_s')
        recs2 = []
        for mode in auc_p.index:
            e = cmp_df[(cmp_df['mode'] == mode) &
                       (cmp_df['metric'] == 'roc_auc')].iloc[0]
            recs2.append({
                'mode': mode,
                'eegnet_whole_epoch_before': e['before_median'],
                'eegnet_whole_epoch_after': e['after_median'],
                'eegnet_delta': e['after_median'] - e['before_median'],
                'sliding_peak_before': float(auc_p.loc[mode, 'before']),
                'sliding_peak_after': float(auc_p.loc[mode, 'after']),
                'sliding_delta': float(auc_p.loc[mode, 'after'] -
                                       auc_p.loc[mode, 'before']),
                'sliding_peak_time_before_s': float(tim_p.loc[mode, 'before']),
                'sliding_peak_time_after_s': float(tim_p.loc[mode, 'after']),
            })
        cmp_sl = pd.DataFrame(recs2)
        sl_path = OUT_DIR / 'eegnet-vs-sliding.csv'
        cmp_sl.to_csv(sl_path, index=False, encoding='utf-8-sig')
        logger.info(f'wrote {sl_path}')
        logger.info('--- 降幅对比（AUC）---')
        for r in recs2:
            logger.info(f"  {r['mode']}: EEGNet 整段 "
                        f"{r['eegnet_whole_epoch_before']:.3f} -> "
                        f"{r['eegnet_whole_epoch_after']:.3f} "
                        f"({r['eegnet_delta']:+.3f})   |   "
                        f"sliding 峰 {r['sliding_peak_before']:.3f} -> "
                        f"{r['sliding_peak_after']:.3f} "
                        f"({r['sliding_delta']:+.3f})")
    else:
        logger.info(f'未找到 {sliding_csv}，跳过与滑窗解码的对比')

    # ---- 图：配对散点 ----------------------------------------------
    n = len(METRICS)
    fig, axes = plt.subplots(2, n, figsize=(2.05 * n, 5.6))
    for r, mode in enumerate(('EEG', 'MEG')):
        sub = wide.xs(mode, level='mode')
        for c, k in enumerate(METRICS):
            ax = axes[r][c]
            b = sub[(k, 'before')].to_numpy(float)
            a = sub[(k, 'after')].to_numpy(float)
            xs = sub.index.to_numpy()
            for i in range(len(b)):
                ax.plot([1, 2], [b[i], a[i]], color='0.75', lw=0.9, zorder=1)
            ax.scatter(np.ones(len(b)), b, s=26, color='#3b6ea5',
                       zorder=2, label='before')
            ax.scatter(np.ones(len(a)) * 2, a, s=26, color='#c1553b',
                       zorder=2, label='after')
            ax.plot([1, 2], [np.median(b), np.median(a)], color='k',
                    lw=1.8, zorder=3)
            rec = cmp_df[(cmp_df['mode'] == mode) &
                         (cmp_df['metric'] == k)].iloc[0]
            ax.set_title(f"{LABEL[k]}\n"
                         f"med {rec['before_median']:.3f} -> "
                         f"{rec['after_median']:.3f}  "
                         f"p={rec['p']:.3f}", fontsize=8.5)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['before', 'after'], fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_xlim(0.65, 2.35)
            ax.grid(axis='y', color='0.9', lw=0.5)
            ax.set_axisbelow(True)
            if c == 0:
                ax.set_ylabel(f'{mode}\nvalue', fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, frameon=False, loc='lower left')
    fig.suptitle('EEGNetV4: before vs after remove-artificial '
                 '(paired by subject, single run each)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    figpath = OUT_DIR / 'eegnet-before-after-paired.png'
    fig.savefig(figpath, dpi=args.dpi, facecolor='white')
    plt.close(fig)
    logger.info(f'wrote {figpath}')

    # ---- 图：target 命中率的分解 -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for r, mode in enumerate(('EEG', 'MEG')):
        ax = axes[r]
        sub = wide.xs(mode, level='mode')
        rec = sub[('recall', 'before')].to_numpy(float)
        rec_a = sub[('recall', 'after')].to_numpy(float)
        spec = sub[('specificity', 'before')].to_numpy(float)
        spec_a = sub[('specificity', 'after')].to_numpy(float)
        idx = np.arange(len(rec))
        w = 0.38
        ax.bar(idx - w / 2, rec, w, color='#3b6ea5', label='target recall: before')
        ax.bar(idx + w / 2, rec_a, w, color='#c1553b', label='target recall: after')
        ax.plot(idx - w / 2, spec, '_', color='#3b6ea5', ms=12, mew=1.6,
                label='non-target recall: before')
        ax.plot(idx + w / 2, spec_a, '_', color='#c1553b', ms=12, mew=1.6,
                label='non-target recall: after')
        ax.set_xticks(idx)
        ax.set_xticklabels([s.replace('S0', 'S') for s in sub.index],
                           fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel('rate', fontsize=9)
        ax.set_title(f'{mode}  (bar = target, tick = non-target)', fontsize=9.5)
        ax.grid(axis='y', color='0.9', lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        if r == 0:
            ax.legend(fontsize=7, frameon=False, ncol=2, loc='lower right')
    fig.suptitle('per-class recall: what remove-artificial costs',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figpath2 = OUT_DIR / 'eegnet-per-class-recall.png'
    fig.savefig(figpath2, dpi=args.dpi, facecolor='white')
    plt.close(fig)
    logger.info(f'wrote {figpath2}')


if __name__ == '__main__':
    main()
