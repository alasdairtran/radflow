import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import STOPWORDS, WordCloud

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

    def get_paths(kind, data, n_hops):
        return [f'expt/missing/{kind}/{data}/{n_hops}_ff_00/00/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}_ff_00/20/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}_ff_00/40/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}_ff_00/60/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}_ff_00/80/serialization/evaluate-metrics.json']

    y1 = get_means(get_paths('views', 'vevo', 'no_hops'))
    y2 = get_means(get_paths('views', 'vevo', 'one_hop'))
    y3 = get_means(get_paths('views', 'vevo', 'two_hops'))

    fig = plt.figure(figsize=(6, 3))
    ax = plt.subplot(1, 2, 1)

    xs = [0, 20, 40, 60, 80]
    ax.errorbar(xs, y1, marker='x', linestyle=':')
    ax.errorbar(xs, y2, marker='x', linestyle='--')
    ax.errorbar(xs, y3, marker='x', linestyle='-')

    ax.set_ylabel('SMAPE-7')
    ax.set_xlabel('% of missing views')
    ax.legend(['No hops', 'One hop', 'Two hops'])

    ax = plt.subplot(1, 2, 2)

    y1 = get_means(get_paths('wiki', 'no_hops'))
    y2 = get_means(get_paths('wiki', 'one_hop'))
    y3 = get_means(get_paths('wiki', 'two_hops'))

    ax.errorbar(xs, y1, marker='x', linestyle=':')
    ax.errorbar(xs, y2, marker='x', linestyle='--')
    ax.errorbar(xs, y3, marker='x', linestyle='-')

    ax.set_ylabel('SMAPE-28')
    ax.set_xlabel('% of missing views')
    ax.legend(['No hops', 'One hop', 'Two hops'])

    fig.tight_layout()
    fig.savefig('figures/missing_views.pdf')


def plot_missing_edges():
    def get_paths(kind, data, n_hops):
        return [f'expt/missing/{kind}/{data}/{n_hops}_ff_00/00/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}_ff_00/20/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}_ff_00/40/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}_ff_00/60/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}_ff_00/80/serialization/evaluate-metrics.json']

    y1 = get_means(get_paths('edges', 'vevo', 'no_hops'))
    y2 = get_means(get_paths('edges', 'vevo', 'one_hop'))
    y3 = get_means(get_paths('edges', 'vevo', 'two_hops'))

    fig = plt.figure(figsize=(6, 3))
    ax = plt.subplot(1, 2, 1)

    xs = [0, 20, 40, 60, 80]
    ax.errorbar(xs, y1, marker=None, linestyle=':')
    ax.errorbar(xs, y2, marker='x', linestyle='--')
    ax.errorbar(xs, y3, marker='x', linestyle='-')

    ax.set_ylabel('SMAPE-7')
    ax.set_xlabel('% of missing edges')
    ax.legend(['No hops', 'One hop', 'Two hops'], bbox_to_anchor=(0.51, 0.9))

    ax = plt.subplot(1, 2, 2)

    y1 = get_means(get_paths('edges', 'wiki', 'no_hops'))
    y2 = get_means(get_paths('edges', 'wiki', 'one_hop'))
    y3 = get_means(get_paths('edges', 'wiki', 'two_hops'))

    ax.errorbar(xs, y1, marker=None, linestyle=':')
    ax.errorbar(xs, y2, marker='x', linestyle='--')
    ax.errorbar(xs, y3, marker='x', linestyle='-')

    ax.legend(['No hops', 'One hop', 'Two hops'], bbox_to_anchor=(0.51, 0.9))

    ax.set_ylabel('SMAPE-28')
    ax.set_xlabel('% of missing edges')

    fig.tight_layout()
    fig.savefig('figures/missing_edges.pdf')


def generate_word_cloud(seed):
    with open(f'data/wiki/node_titles/{seed}.pkl', 'rb') as f:
        titles = pickle.load(f)

    titles = [t.lower() for t in titles]
    wordcloud = WordCloud(
        background_color='white',
        stopwords=STOPWORDS,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1,
    ).generate(' '.join(titles))

    fig = plt.figure(1, figsize=(6, 6))
    plt.axis('off')

    plt.imshow(wordcloud)
    fig.tight_layout()
    fig.savefig(f'figures/word_cloud_{seed}.pdf')


def main():
    plot_missing_views()
    plot_missing_edges()

    generate_word_cloud('global_health')
    generate_word_cloud('programming_languages')
    generate_word_cloud('star_wars')
    generate_word_cloud('global_warming')


if __name__ == '__main__':
    main()
