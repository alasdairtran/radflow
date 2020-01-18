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

    with open('expt/9_agg_mean/serialization/evaluate-metrics.json') as f:
        smape['mean'] = json.load(f)['smape']

    with open('expt/8_agg_attention/serialization/evaluate-metrics.json') as f:
        smape['attn'] = json.load(f)['smape']

    with open('expt/10_gcn/serialization/evaluate-metrics.json') as f:
        smape['gcn'] = json.load(f)['smape']

    smapes = [smape['naive'], smape['SN'], smape['LSTM'],
              smape['mean'], smape['attn'], smape['gcn']]

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(1, 1, 1)
    ax.boxplot(smapes, showfliers=False, meanline=True,
               showmeans=True, widths=0.7)
    ax.set_xticklabels(
        ['Naive', 'Seasonal', 'No Agg', 'Mean', 'Attn', 'GCN'])
    ax.set_ylabel('SMAPE')

    means = [np.mean(x) for x in smapes]
    pos = range(len(means))
    for tick, label in zip(pos, ax.get_xticklabels()):
        ax.text(pos[tick] + 0.85, means[tick] +
                0.07, '{0:.3f}'.format(means[tick]))

    fig.tight_layout()
    plt.show()
    fig.savefig('smape.png')


def main():
    os.makedirs('figures', exist_ok=True)
    make_boxplots()


if __name__ == '__main__':
    main()
