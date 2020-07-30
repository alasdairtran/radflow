import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("ticks")
sns.set_context("paper")


def get_means(paths):
    smapes = []
    for metric_path in paths:
        with open(metric_path) as f:
            o = json.load(f)
            smapes.append(o['smapes'])

    smapes = np.array(smapes)
    metrics = np.mean(smapes.reshape(5, -1), axis=-1)

    return metrics


def plot_missing_views():
    data = 'vevo'
    no_hops_ff_00_metrics = [f'expt/missing/views/{data}/no_hops_ff_00/00/serialization/evaluate-metrics.json',
                             f'expt/missing/views/{data}/no_hops_ff_00/20/serialization/evaluate-metrics.json',
                             f'expt/missing/views/{data}/no_hops_ff_00/40/serialization/evaluate-metrics.json',
                             f'expt/missing/views/{data}/no_hops_ff_00/60/serialization/evaluate-metrics.json',
                             f'expt/missing/views/{data}/no_hops_ff_00/80/serialization/evaluate-metrics.json']

    one_hop_ff_00_metrics = [f'expt/missing/views/{data}/one_hop_ff_00/00/serialization/evaluate-metrics.json',
                             f'expt/missing/views/{data}/one_hop_ff_00/20/serialization/evaluate-metrics.json',
                             f'expt/missing/views/{data}/one_hop_ff_00/40/serialization/evaluate-metrics.json',
                             f'expt/missing/views/{data}/one_hop_ff_00/60/serialization/evaluate-metrics.json',
                             f'expt/missing/views/{data}/one_hop_ff_00/80/serialization/evaluate-metrics.json']

    two_hop_ff_00_metrics = [f'expt/missing/views/{data}/two_hops_ff_00/00/serialization/evaluate-metrics.json',
                             f'expt/missing/views/{data}/two_hops_ff_00/20/serialization/evaluate-metrics.json',
                             f'expt/missing/views/{data}/two_hops_ff_00/40/serialization/evaluate-metrics.json',
                             f'expt/missing/views/{data}/two_hops_ff_00/60/serialization/evaluate-metrics.json',
                             f'expt/missing/views/{data}/two_hops_ff_00/80/serialization/evaluate-metrics.json']

    y1 = get_means(no_hops_ff_00_metrics)
    y2 = get_means(one_hop_ff_00_metrics)
    y3 = get_means(two_hop_ff_00_metrics)

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 2, 1)

    xs = [0, 0.2, 0.4, 0.6, 0.8]
    ax.errorbar(xs, y1, marker='x', linestyle=':')
    ax.errorbar(xs, y2, marker='x', linestyle='--')
    ax.errorbar(xs, y3, marker='x', linestyle='-')

    ax.set_ylabel('SMAPE-7')
    ax.set_xlabel('Proportion of missing views')
    ax.legend(['No hops', 'One hop', 'Two hops'])

    ax = plt.subplot(1, 2, 2)
    ax.set_ylabel('SMAPE-28')
    ax.set_xlabel('Proportion of missing views')

    fig.tight_layout()
    fig.savefig('figures/vevo_missing_views.pdf')


def main():
    plot_missing_views()


if __name__ == '__main__':
    main()
