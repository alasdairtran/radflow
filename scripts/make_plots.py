import json
import os

import matplotlib.pyplot as plt
import numpy as np


def make_boxplots():
    smape = {}

    with open('expt/1_naive_previous_day/serialization/evaluate-metrics.json') as f:
        smape['naive'] = json.load(f)['smape']

    with open('expt/2_naive_seasonal/serialization/evaluate-metrics.json') as f:
        smape['SN'] = json.load(f)['smape']

    with open('expt/11_no_agg/serialization/evaluate-metrics.json') as f:
        smape['LSTM'] = json.load(f)['smape']

    with open('expt/12_peek/serialization/evaluate-metrics.json') as f:
        smape['static_mean'] = json.load(f)['smape']

    with open('expt/16_peek_daily/serialization/evaluate-metrics.json') as f:
        smape['dynamic_mean'] = json.load(f)['smape']

    with open('expt/18_peek_daily_attn/serialization/evaluate-metrics.json') as f:
        smape['dynamic_attn'] = json.load(f)['smape']

    with open('expt/17_peek_daily_sage/serialization/evaluate-metrics.json') as f:
        smape['dynamic_sage'] = json.load(f)['smape']

    smapes = [smape['naive'], smape['SN'], smape['LSTM'],
              smape['static_mean'], smape['dynamic_mean'],
              smape['dynamic_attn'],
              smape['dynamic_sage']
              ]

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)
    ax.boxplot(smapes, showfliers=False, meanline=True,
               showmeans=True, widths=0.7)
    ax.set_xticklabels(
        ['Naive', 'Seasonal', 'Baseline', 'Static Mean', 'Dynamic Mean',
         'Dynamic Attn',
         'Dynamic Sage',
         ])
    ax.set_ylabel('SMAPE')

    means = [np.mean(x) for x in smapes]
    pos = range(len(means))
    for tick, label in zip(pos, ax.get_xticklabels()):
        ax.text(pos[tick] + 0.85, means[tick] +
                0.07, '{0:.3f}'.format(means[tick]))

    fig.tight_layout()
    plt.show()
    fig.savefig('figures/smape.png')


def make_partial_info_plots():
    smape = {}

    with open('expt/missing/01_no_agg_20/serialization/evaluate-metrics.json') as f:
        smape['no_agg_20'] = json.load(f)['smape']

    with open('expt/missing/02_sage_20/serialization/evaluate-metrics.json') as f:
        smape['sage_20'] = json.load(f)['smape']

    with open('expt/missing/03_no_agg_40/serialization/evaluate-metrics.json') as f:
        smape['no_agg_40'] = json.load(f)['smape']

    with open('expt/missing/04_sage_40/serialization/evaluate-metrics.json') as f:
        smape['sage_40'] = json.load(f)['smape']

    with open('expt/missing/05_no_agg_00/serialization/evaluate-metrics.json') as f:
        smape['no_agg_00'] = json.load(f)['smape']

    with open('expt/missing/06_sage_00/serialization/evaluate-metrics.json') as f:
        smape['sage_00'] = json.load(f)['smape']

    with open('expt/missing/07_no_agg_60/serialization/evaluate-metrics.json') as f:
        smape['no_agg_60'] = json.load(f)['smape']

    with open('expt/missing/08_sage_60/serialization/evaluate-metrics.json') as f:
        smape['sage_60'] = json.load(f)['smape']

    no_agg_smapes = [smape['no_agg_00'], smape['no_agg_20'],
                     smape['no_agg_40'], smape['no_agg_60']]
    sage_smapes = [smape['sage_00'], smape['sage_20'],
                   smape['sage_40'], smape['sage_60']]

    no_agg_means = [np.median(x) for x in no_agg_smapes]
    sage_means = [np.median(x) for x in sage_smapes]
    # no_agg_means = [np.quantile(x, 0.25) for x in no_agg_smapes]
    # sage_means = [np.quantile(x, 0.25) for x in sage_smapes]

    xs = [0, 0.2, 0.4, 0.6]

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)
    ax.errorbar(xs, no_agg_means, marker='o')
    ax.errorbar(xs, sage_means, marker='x')

    ax.set_ylabel('Median SMAPE')
    ax.set_xlabel('% missing views')
    ax.legend(['No aggregation', 'GraphSage'])
    ax.set_title('Effect of node aggregation on missing data')

    fig.tight_layout()
    plt.show()
    fig.savefig('figures/partial_info.png')


def make_partial_edge_plots():
    smape = {}

    with open('expt/missing/08_sage_60/serialization/evaluate-metrics.json') as f:
        smape['1_hop_00'] = json.load(f)['smape']

    with open('expt/missing_edges/1_hop_20/serialization/evaluate-metrics.json') as f:
        smape['1_hop_80'] = json.load(f)['smape']

    with open('expt/missing_edges/1_hop_40/serialization/evaluate-metrics.json') as f:
        smape['1_hop_60'] = json.load(f)['smape']

    with open('expt/missing_edges/1_hop_60/serialization/evaluate-metrics.json') as f:
        smape['1_hop_40'] = json.load(f)['smape']

    # with open('expt/missing/07_no_agg_60/serialization/evaluate-metrics.json') as f:
    #     smape['2_hop_00'] = json.load(f)['smape']

    with open('expt/missing_edges/2_hop_20/serialization/evaluate-metrics.json') as f:
        smape['2_hop_80'] = json.load(f)['smape']

    with open('expt/missing_edges/2_hop_40/serialization/evaluate-metrics.json') as f:
        smape['2_hop_60'] = json.load(f)['smape']

    with open('expt/missing_edges/2_hop_60/serialization/evaluate-metrics.json') as f:
        smape['2_hop_40'] = json.load(f)['smape']

    one_hop_smapes = [smape['1_hop_00'], smape['1_hop_40'],
                      smape['1_hop_60'], smape['1_hop_80']]
    two_hop_smapes = [smape['2_hop_40'], smape['2_hop_60'],
                      smape['2_hop_80']]

    one_hop_means = [np.median(x) for x in one_hop_smapes]
    two_hop_smapes = [np.nan] + [np.median(x) for x in two_hop_smapes]
    # one_hop_means = [np.quantile(x, 0.25) for x in one_hop_smapes]
    # two_hop_smapes = [np.quantile(x, 0.25) for x in two_hop_smapes]

    xs = [0, 0.4, 0.6, 0.8]

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)
    ax.errorbar(xs, one_hop_means, marker='o')
    ax.errorbar(xs, two_hop_smapes, marker='x')

    ax.set_ylabel('Median SMAPE')
    ax.set_xlabel('% missing edges')
    ax.legend(['One Hop', 'Two Hops'])
    ax.set_title('Effect of node aggregation on missing edges')

    fig.tight_layout()
    plt.show()
    fig.savefig('figures/partial_edges.png')


def main():
    os.makedirs('figures', exist_ok=True)
    # make_boxplots()
    # make_partial_info_plots()
    make_partial_edge_plots()


if __name__ == '__main__':
    main()
