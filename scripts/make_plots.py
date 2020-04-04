import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import setp


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


def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
#     setp(bp['fliers'][0], color='blue')
#     setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
#     setp(bp['fliers'][2], color='red')
#     setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')

def make_partial_info_boxplots():
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

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)
    bp = ax.boxplot([smape['no_agg_00'], smape['sage_00']],
                positions = [1, 2],
               showfliers=False, meanline=True,
               showmeans=True, widths=0.6)
    setBoxColors(bp)

    bp = ax.boxplot([smape['no_agg_20'], smape['sage_20']],
                positions = [4, 5],
               showfliers=False, meanline=True,
               showmeans=True, widths=0.6)
    setBoxColors(bp)

    bp = ax.boxplot([smape['no_agg_40'], smape['sage_40']],
                positions = [7, 8],
               showfliers=False, meanline=True,
               showmeans=True, widths=0.6)
    setBoxColors(bp)

    bp = ax.boxplot([smape['no_agg_60'], smape['sage_60']],
                positions = [10, 11],
               showfliers=False, meanline=True,
               showmeans=True, widths=0.6)
    setBoxColors(bp)

    ax.set_xticklabels(['0', '0.2', '0.4', '0.6'])
    ax.set_xticks([1.5, 4.5, 7.5, 10.5])

    hB, = plt.plot([1,1],'b-')
    hR, = plt.plot([1,1],'r-')
    plt.legend((hB, hR),('No Aggregation', 'GraphSage'))
    hB.set_visible(False)
    hR.set_visible(False)


    ax.set_ylabel('SMAPE')
    ax.set_xlabel('% missing views')
    ax.set_title('Effect of node aggregation on missing data')

    fig.tight_layout()
    plt.show()
    fig.savefig('figures/partial_info_boxes.png')



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

    with open('expt/missing_edges/1_hop_80/serialization/evaluate-metrics.json') as f:
        smape['1_hop_20'] = json.load(f)['smape']

    with open('expt/missing_edges/2_hop_20/serialization/evaluate-metrics.json') as f:
        smape['2_hop_80'] = json.load(f)['smape']

    with open('expt/missing_edges/2_hop_40/serialization/evaluate-metrics.json') as f:
        smape['2_hop_60'] = json.load(f)['smape']

    with open('expt/missing_edges/2_hop_60/serialization/evaluate-metrics.json') as f:
        smape['2_hop_40'] = json.load(f)['smape']

    with open('expt/missing_edges/2_hop_80/serialization/evaluate-metrics.json') as f:
        smape['2_hop_20'] = json.load(f)['smape']

    with open('expt/missing_edges/2_hop_00/serialization/evaluate-metrics.json') as f:
        smape['2_hop_00'] = json.load(f)['smape']

    one_hop_smapes = [smape['1_hop_00'], smape['1_hop_20'], smape['1_hop_40'],
                      smape['1_hop_60'], smape['1_hop_80']]
    two_hop_smapes = [smape['2_hop_00'], smape['2_hop_20'], smape['2_hop_40'],
                      smape['2_hop_60'], smape['2_hop_80']]

    one_hop_means = [np.median(x) for x in one_hop_smapes]
    two_hop_smapes = [np.median(x) for x in two_hop_smapes]
    # one_hop_means = [np.quantile(x, 0.25) for x in one_hop_smapes]
    # two_hop_smapes = [np.quantile(x, 0.25) for x in two_hop_smapes]

    xs = [0, 0.2, 0.4, 0.6, 0.8]

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

def make_partial_edge_boxplots():
    smape = {}

    with open('expt/missing/08_sage_60/serialization/evaluate-metrics.json') as f:
        smape['1_hop_00'] = json.load(f)['smape']

    with open('expt/missing_edges/1_hop_20/serialization/evaluate-metrics.json') as f:
        smape['1_hop_80'] = json.load(f)['smape']

    with open('expt/missing_edges/1_hop_40/serialization/evaluate-metrics.json') as f:
        smape['1_hop_60'] = json.load(f)['smape']

    with open('expt/missing_edges/1_hop_60/serialization/evaluate-metrics.json') as f:
        smape['1_hop_40'] = json.load(f)['smape']

    with open('expt/missing_edges/1_hop_80/serialization/evaluate-metrics.json') as f:
        smape['1_hop_20'] = json.load(f)['smape']

    with open('expt/missing_edges/2_hop_20/serialization/evaluate-metrics.json') as f:
        smape['2_hop_80'] = json.load(f)['smape']

    with open('expt/missing_edges/2_hop_40/serialization/evaluate-metrics.json') as f:
        smape['2_hop_60'] = json.load(f)['smape']

    with open('expt/missing_edges/2_hop_60/serialization/evaluate-metrics.json') as f:
        smape['2_hop_40'] = json.load(f)['smape']

    with open('expt/missing_edges/2_hop_80/serialization/evaluate-metrics.json') as f:
        smape['2_hop_20'] = json.load(f)['smape']

    with open('expt/missing_edges/2_hop_00/serialization/evaluate-metrics.json') as f:
        smape['2_hop_00'] = json.load(f)['smape']

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)

    bp = ax.boxplot([smape['1_hop_00'], smape['2_hop_00']],
                positions = [1, 2],
               showfliers=False, meanline=True,
               showmeans=True, widths=0.6)
    setBoxColors(bp)

    bp = ax.boxplot([smape['1_hop_20'], smape['2_hop_20']],
                positions = [4, 5],
               showfliers=False, meanline=True,
               showmeans=True, widths=0.6)
    setBoxColors(bp)

    bp = ax.boxplot([smape['1_hop_40'], smape['2_hop_40']],
                positions = [7, 8],
               showfliers=False, meanline=True,
               showmeans=True, widths=0.6)
    setBoxColors(bp)

    bp = ax.boxplot([smape['1_hop_60'], smape['2_hop_60']],
                positions = [10, 11],
               showfliers=False, meanline=True,
               showmeans=True, widths=0.6)
    setBoxColors(bp)

    bp = ax.boxplot([smape['1_hop_80'], smape['2_hop_80']],
                positions = [13, 14],
               showfliers=False, meanline=True,
               showmeans=True, widths=0.6)
    setBoxColors(bp)

    ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8'])
    ax.set_xticks([1.5, 4.5, 7.5, 10.5, 13.5])

    hB, = plt.plot([1,1],'b-')
    hR, = plt.plot([1,1],'r-')
    plt.legend((hB, hR),('1 Hop', '2 Hops'))
    hB.set_visible(False)
    hR.set_visible(False)

    ax.set_ylabel('SMAPE')
    ax.set_xlabel('% missing edges')
    ax.set_title('Effect of node aggregation on missing edges')

    fig.tight_layout()
    plt.show()
    fig.savefig('figures/partial_edges_boxes.png')


def make_neighbour_boxplots():
    smape = {}

    with open('expt/neighbour_size/2_top_1/serialization/evaluate-metrics.json') as f:
        smape['Top 1'] = json.load(f)['smape']

    with open('expt/neighbour_size/3_top_5/serialization/evaluate-metrics.json') as f:
        smape['Top 5'] = json.load(f)['smape']

    with open('expt/neighbour_size/4_top_10/serialization/evaluate-metrics.json') as f:
        smape['Top 10'] = json.load(f)['smape']

    with open('expt/neighbour_size/1_top_20/serialization/evaluate-metrics.json') as f:
        smape['Top 20'] = json.load(f)['smape']

    smapes = [smape['Top 1'], smape['Top 5'], smape['Top 10'], smape['Top 20']]

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)
    ax.boxplot(smapes, showfliers=False, meanline=True,
               showmeans=True, widths=0.7)
    ax.set_xticklabels(
        ['Top 1', 'Top 5', 'Top 10', 'Top 20'])
    ax.set_ylabel('SMAPE')

    means = [np.mean(x) for x in smapes]
    pos = range(len(means))
    for tick, label in zip(pos, ax.get_xticklabels()):
        ax.text(pos[tick] + 0.85, means[tick] +
                0.07, '{0:.3f}'.format(means[tick]))

    fig.tight_layout()
    plt.show()
    fig.savefig('figures/neighbour_smape.png')


def main():
    os.makedirs('figures', exist_ok=True)
    # make_boxplots()
    # make_partial_info_plots()
    # make_partial_info_boxplots()
    # make_partial_edge_plots()
    # make_partial_edge_boxplots()
    make_neighbour_boxplots()


if __name__ == '__main__':
    main()
