#!/usr/bin/env python
'''
把 output/ 下现有的结果 csv 汇总成故事线用的统计表。

只读 csv，不读 epochs / stc，所以在没有原始数据的机器上也能跑。产出：

    output/summary/storyline-stats.md    人读的表,直接贴进写作文档
    output/summary/storyline-stats.csv   机器读的,一行一个统计量

每个统计量都带一列 readout,写明它是怎么算出来的,以及来自哪个文件。
跨文件的派生量(几何均值比、试次不平衡的 rho)只在这里算一次,避免文档里的
数字是手抄的。

用法:
    python python/summarize-storyline.py
'''

import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon, ttest_1samp

OUTPUT_DIR = Path('output')
SUMMARY_DIR = OUTPUT_DIR / 'summary'

MODE_ORDER = ('EEG', 'MEG')
SUBJECTS = tuple(f'S{i:02d}' for i in range(1, 11))

# 平衡子集的门限:两组的试次数比值不超过这个值才算平衡
BALANCE_RATIO = 2.0

logger = logging.getLogger('summarize-storyline')


def read_csv(path):
    '''Read one output csv into a list of dicts, empty list when missing.'''
    path = Path(path)
    if not path.exists():
        logger.warning(f'{path} does not exist, skipped')
        return []
    with open(path, encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def num(row, key, default=np.nan):
    '''Float value of one cell, nan for empty ones.'''
    value = (row.get(key) or '').strip()
    try:
        return float(value)
    except ValueError:
        return default


class Stats:
    '''Collects the derived statistics as one flat table.'''

    def __init__(self):
        self.rows = []

    def add(self, block, mode, measure, n, value, pvalue=np.nan,
            readout='', source=''):
        self.rows.append(dict(block=block, mode=mode, measure=measure, n=n,
                              value=value, pvalue=pvalue, readout=readout,
                              source=source))
        logger.info(f'{block:10s} {mode:4s} {measure:34s} '
                    f'value {value}  p {pvalue}')

    def dump(self):
        SUMMARY_DIR.mkdir(exist_ok=True, parents=True)
        fname = SUMMARY_DIR / 'storyline-stats.csv'
        with open(fname, 'w', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)
        logger.info(f'wrote {fname}')


def aligned_peak1_gfp(structure):
    '''First peak amplitude and latency for every subject and condition.'''
    table = {}
    for row in structure:
        if row['tag'] != 'rma':
            continue
        key = (row['mode'], row['subject'], row['condition'])
        table[key] = row
    return table


def summarize_decoding(stats):
    '''Sliding decode peak and the three peak window analyses.'''
    src = 'sliding-decode/group-decode-summary.csv'
    for row in read_csv(OUTPUT_DIR / src):
        label = f'{row["decoding"]}_{row["condition"]}'
        stats.add('decoding', row['mode'], f'peak_auc_{label}',
                  row['n_subjects'], num(row, 'peak_auc'),
                  readout='群体滑动解码曲线的最大 AUC',
                  source=src)
        stats.add('decoding', row['mode'], f'peak_time_{label}',
                  row['n_subjects'], num(row, 'peak_time_s'),
                  readout='群体滑动解码曲线的峰值时刻 (s)', source=src)

    src = 'peak-window-decode/group-peak-window-windows-rma.csv'
    for row in read_csv(OUTPUT_DIR / src):
        if not row['pvalue']:
            continue
        stats.add('decoding', row['mode'], f'window_auc_{row["window"]}',
                  row['n'], num(row, 'auc'), num(row, 'pvalue'),
                  readout='窗内时间平均后的 target vs non-target AUC,'
                          ' 对 0.5 做单样本 Wilcoxon',
                  source=src)

    src = 'peak-window-decode/group-peak-window-transfer-rma.csv'
    for row in read_csv(OUTPUT_DIR / src):
        if row['train_window'] == row['test_window']:
            continue
        stats.add('decoding', row['mode'],
                  f'transfer_{row["train_window"]}2{row["test_window"]}',
                  row['n'], num(row, 'auc'), num(row, 'pvalue'),
                  readout='跨窗迁移 AUC, 对 0.5 做单样本 Wilcoxon',
                  source=src)

    src = 'peak-window-decode/group-peak-window-rt-rma.csv'
    for row in read_csv(OUTPUT_DIR / src):
        stats.add('decoding', row['mode'], f'rt_decode_{row["measure"]}',
                  row['n'], num(row, 'auc'), num(row, 'pvalue'),
                  readout='试次内 quick vs slow 解码, 对 0.5 做单样本 Wilcoxon',
                  source=src)


def summarize_rt_split(stats):
    '''How the trial split was made and how comparable the two groups are.'''
    src = 'quick-slow/summary-rma.csv'
    rows = read_csv(OUTPUT_DIR / src)
    for mode in MODE_ORDER:
        sub = [r for r in rows if r['mode'] == mode]
        if not sub:
            continue
        for key, unit in (('n_valid', '有效试次数'), ('threshold', 's'),
                          ('rt_quick', 's'), ('rt_slow', 's'), ('gap', 's'),
                          ('n_quick', '试次'), ('n_slow', '试次')):
            values = np.array([num(r, key) for r in sub])
            stats.add('rt_split', mode, f'{key}_median', len(values),
                      float(np.median(values)),
                      readout=f'{unit}, 10 名被试的中位数', source=src)
        weak = [r['subject'] for r in sub if int(num(r, 'weak', 0))]
        stats.add('rt_split', mode, 'weak_subjects', len(weak), len(weak),
                  readout=f'两组 RT 分不开的被试 {weak or "无"}', source=src)
        methods = defaultdict(int)
        for row in sub:
            methods[row['method']] += 1
        stats.add('rt_split', mode, 'split_methods', len(sub),
                  len(methods),
                  readout='分组规则分布 ' + ', '.join(
                      f'{k} {v}/10' for k, v in sorted(methods.items())),
                  source=src)

        # 逐试次幅度与 RT 的 rho,跨被试检验它是否偏离 0
        rho = np.array([num(r, 'spearman_rho') for r in sub])
        stat, pvalue = ttest_1samp(rho, 0.)
        stats.add('rt_split', mode, 'trialwise_rho', len(rho),
                  float(rho.mean()), float(pvalue),
                  readout='逐试次第一峰幅度 vs RT 的 Spearman rho,'
                          f' 跨被试单样本 t 检验 (mean {rho.mean():+.4f})',
                  source=src)


def summarize_peak_structure(stats):
    '''Double peak rate, dip depth and the quick / slow contrast on peaks.'''
    src = 'peak-structure/group-peak-incidence-rma.csv'
    for row in read_csv(OUTPUT_DIR / src):
        stats.add('peak_structure', row['mode'],
                  f'double_peak_frac_{row["condition"]}', row['n'],
                  num(row, 'frac_double'),
                  readout='双峰被试比例, 阈值 n_peaks >= 2', source=src)
        stats.add('peak_structure', row['mode'],
                  f'dip_ratio_{row["condition"]}', row['n'],
                  num(row, 'dip_ratio'),
                  readout='逐被试谷深后取均值, dip_ratio = 两峰间谷 / 第一峰',
                  source=src)

    src = 'peak-structure/group-peak-latency-rma.csv'
    for row in read_csv(OUTPUT_DIR / src):
        stats.add('peak_structure', row['mode'],
                  f'latency_shift_{row["readout"]}', row['n'],
                  num(row, 'shift'), num(row, 'pvalue'),
                  readout='quick 减 slow 的潜伏期差 (s), Wilcoxon 配对检验',
                  source=src)

    src = 'peak-structure/group-peak-agreement-rma.csv'
    for row in read_csv(OUTPUT_DIR / src):
        stats.add('peak_structure', row['mode'],
                  f'argmax_vs_peak1_disagree_{row["condition"]}', row['n'],
                  num(row, 'n_disagree'),
                  readout='argmax 落到第二峰的被试数, 平均绝对差 '
                          f'{num(row, "mean_abs_diff"):.3f} s', source=src)

    src = 'peak-structure/peak-structure-rma.csv'
    table = aligned_peak1_gfp(read_csv(OUTPUT_DIR / src))

    # 幅度效应: 逐被试 quick / slow 的比值,用几何均值报(对数尺度上取平均),
    # 再对 1 做 Wilcoxon。试次数悬殊的被试单独剔一遍看稳健性。
    src_qs = 'quick-slow/summary-rma.csv'
    trial_count = {(r['mode'], r['subject']): r
                   for r in read_csv(OUTPUT_DIR / src_qs)}
    for mode in MODE_ORDER:
        ratios, balance = [], []
        for subject in SUBJECTS:
            quick = table.get((mode, subject, 'quick'))
            slow = table.get((mode, subject, 'slow'))
            counts = trial_count.get((mode, subject))
            if quick is None or slow is None or counts is None:
                logger.warning(f'{mode}-{subject}: peak structure or trial '
                               f'count missing, skipped')
                continue
            ratios.append(num(quick, 'peak1_gfp') / num(slow, 'peak1_gfp'))
            n_quick = num(counts, 'n_quick')
            n_slow = num(counts, 'n_slow')
            balance.append(max(n_quick, n_slow) / min(n_quick, n_slow))
        ratios = np.array(ratios)
        balance = np.array(balance)
        geometric = float(np.exp(np.mean(np.log(ratios))))
        stat, pvalue = wilcoxon(ratios, 1.)
        stats.add('fast_slow', mode, 'peak1_amp_ratio_geomean', len(ratios),
                  geometric, float(pvalue),
                  readout='对齐第一峰后 quick / slow 的幅度比 (逐被试取比值'
                          f'后做几何均值), 对 1 做 Wilcoxon, '
                          f'{int((ratios > 1).sum())}/{len(ratios)} 名被试 > 1',
                  source=src)

        keep = balance < BALANCE_RATIO
        stat, pvalue = wilcoxon(ratios[keep], 1.)
        stats.add('fast_slow', mode, 'peak1_amp_ratio_balanced', int(keep.sum()),
                  float(np.exp(np.mean(np.log(ratios[keep])))), float(pvalue),
                  readout=f'只留试次数比值 < {BALANCE_RATIO:g} 的被试后重算, '
                          f'试次比范围 {balance.min():.2f}-{balance.max():.2f}',
                  source=src + ' + ' + src_qs)

        rho, pvalue = spearmanr(np.log(balance), np.log(ratios))
        stats.add('fast_slow', mode, 'trial_balance_rho', len(ratios),
                  float(rho), float(pvalue),
                  readout='log(试次数比) 与 log(幅度比) 的 Spearman rho, '
                          '正值就是"试次越少、幅度看起来越大"这个伪影',
                  source=src + ' + ' + src_qs)


def summarize_source(stats):
    '''What the source leg has right now, and what is missing.'''
    root = OUTPUT_DIR / 'source-estimation'
    groups = sorted(p for p in root.glob('*-S*')) if root.exists() else []
    qc = sorted(root.glob('*-S*/*ssvep10-qc.png'))
    stc = sorted(root.glob('*-S*/*.stc'))
    stats.add('source', 'both', 'subject_folders', len(groups), len(groups),
              readout='output/source-estimation 下的被试目录数 (10 被试 x 2 模态)',
              source='output/source-estimation')
    stats.add('source', 'both', 'ssvep_qc_png', len(qc), len(qc),
              readout='10 Hz SSVEP 质检图张数', source='output/source-estimation')
    stats.add('source', 'both', 'stc_files', len(stc), len(stc),
              readout='源估计 stc 文件数, 0 表示群体图还做不出来',
              source='output/source-estimation')
    group = sorted((OUTPUT_DIR / 'group-source-map').glob('*')) \
        if (OUTPUT_DIR / 'group-source-map').exists() else []
    stats.add('source', 'both', 'group_source_files', len(group), len(group),
              readout='群体 z-map / 群体 mean 脑图产物数 (stage 13 / 14)',
              source='output/group-source-map')


def write_markdown(stats):
    '''The same table in markdown, grouped by block.'''
    fname = SUMMARY_DIR / 'storyline-stats.md'
    blocks = defaultdict(list)
    for row in stats.rows:
        blocks[row['block']].append(row)
    lines = ['# 故事线统计汇总', '',
             '由 `python/summarize-storyline.py` 生成,只读 `output/` 下的 csv,',
             '没有重新读 epochs / stc。每个数字的算法见 readout 列。', '']
    for block in ('decoding', 'rt_split', 'peak_structure', 'fast_slow',
                  'source'):
        rows = blocks.get(block)
        if not rows:
            continue
        lines += [f'## {block}', '',
                  '| 模态 | 指标 | n | 值 | p | 读数 | 出处 |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for row in rows:
            value = f'{row["value"]:.4g}' if isinstance(
                row['value'], float) else str(row['value'])
            pvalue = f'{row["pvalue"]:.4g}' if np.isfinite(
                row['pvalue']) else '—'
            lines.append(
                f'| {row["mode"]} | `{row["measure"]}` | {row["n"]} | '
                f'{value} | {pvalue} | {row["readout"]} | `{row["source"]}` |')
        lines.append('')
    with open(fname, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    logger.info(f'wrote {fname}')


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    stats = Stats()
    summarize_decoding(stats)
    summarize_rt_split(stats)
    summarize_peak_structure(stats)
    summarize_source(stats)
    stats.dump()
    write_markdown(stats)


if __name__ == '__main__':
    main()
