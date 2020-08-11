import json
import pickle
from collections import defaultdict

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        return [f'expt/missing/{kind}/{data}/{n_hops}/00/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}/20/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}/40/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}/60/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}/80/serialization/evaluate-metrics.json']

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

    y1 = get_means(get_paths('views', 'wiki', 'no_hops'))
    y2 = get_means(get_paths('views', 'wiki', 'one_hop'))
    y3 = get_means(get_paths('views', 'wiki', 'two_hops'))

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
        return [f'expt/missing/{kind}/{data}/{n_hops}/00/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}/20/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}/40/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}/60/serialization/evaluate-metrics.json',
                f'expt/missing/{kind}/{data}/{n_hops}/80/serialization/evaluate-metrics.json']

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


def plot_wiki_smape_boxplots():
    path = 'expt/reports/wiki/no_hops/flow_lstm/serialization/evaluate-metrics.json'
    with open(path) as f:
        o3 = json.load(f)

    path = 'expt/reports/wiki/two_hops/flow_lstm/serialization/evaluate-metrics.json'
    with open(path) as f:
        o8 = json.load(f)

    smapes_dict_agg = defaultdict(list)
    smapes_dict_none = defaultdict(list)

    node_ids = {}
    with open('data/wiki/node_ids/global_health.pkl', 'rb') as f:
        node_ids['health'] = pickle.load(f)
    with open('data/wiki/node_ids/global_warming.pkl', 'rb') as f:
        node_ids['warm'] = pickle.load(f)
    with open('data/wiki/node_ids/programming_languages.pkl', 'rb') as f:
        node_ids['program'] = pickle.load(f)
    with open('data/wiki/node_ids/star_wars.pkl', 'rb') as f:
        node_ids['star'] = pickle.load(f)

    for k, ns, s in zip(o8['keys'], o8['neigh_keys'], o8['smapes']):
        for cat in ['health', 'warm', 'program', 'star']:
            if k in node_ids[cat]:
                smapes_dict_agg[cat] += s

    for k, s in zip(o3['keys'], o3['smapes']):
        for cat in ['health', 'warm', 'program', 'star']:
            if k in node_ids[cat]:
                smapes_dict_none[cat] += s

    fig = plt.figure(figsize=(6, 3))
    ax = plt.subplot(1, 1, 1)

    sns.set_palette("muted")
    current_palette = sns.color_palette()

    def setBoxColors(bp):
        plt.setp(bp['boxes'][0], color=current_palette[0])
        plt.setp(bp['caps'][0], color=current_palette[0])
        plt.setp(bp['caps'][1], color=current_palette[0])
        plt.setp(bp['whiskers'][0], color=current_palette[0])
        plt.setp(bp['whiskers'][1], color=current_palette[0])
        plt.setp(bp['means'][0], color=current_palette[0])
        plt.setp(bp['medians'][0], color=current_palette[0])

        plt.setp(bp['boxes'][1], color=current_palette[2])
        plt.setp(bp['caps'][2], color=current_palette[2])
        plt.setp(bp['caps'][3], color=current_palette[2])
        plt.setp(bp['whiskers'][2], color=current_palette[2])
        plt.setp(bp['whiskers'][3], color=current_palette[2])
        plt.setp(bp['means'][1], color=current_palette[2])
        plt.setp(bp['medians'][1], color=current_palette[2])

    smapes_pair = [smapes_dict_none['health'], smapes_dict_agg['health']]
    bp = ax.boxplot(smapes_pair, showfliers=False, meanline=True,
                    showmeans=True, widths=0.7, positions=[1, 2])
    setBoxColors(bp)

    smapes_pair = [smapes_dict_none['warm'], smapes_dict_agg['warm']]
    bp = ax.boxplot(smapes_pair, showfliers=False, meanline=True,
                    showmeans=True, widths=0.7, positions=[4, 5])
    setBoxColors(bp)

    smapes_pair = [smapes_dict_none['program'], smapes_dict_agg['program']]
    bp = ax.boxplot(smapes_pair, showfliers=False, meanline=True,
                    showmeans=True, widths=0.7, positions=[7, 8])
    setBoxColors(bp)

    smapes_pair = [smapes_dict_none['star'], smapes_dict_agg['star']]
    bp = ax.boxplot(smapes_pair, showfliers=False, meanline=True,
                    showmeans=True, widths=0.7, positions=[10, 11])
    setBoxColors(bp)

    ax.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax.set_xticklabels(
        ['Global Health', 'Global Warming', 'Programming', 'Star Wars'])
    ax.set_ylabel('SMAPE-28')

    fig.tight_layout()
    plt.show()
    fig.savefig('figures/smapes_pair.pdf')


def plot_layer_decompositions():
    path = 'expt/reports/wiki/no_hops/flow_lstm/serialization/evaluate-metrics.json'
    with open(path) as f:
        o3 = json.load(f)

    col1_dict = defaultdict(list)
    col2_dict = defaultdict(list)

    for k_parts in o3['f_parts']:
        for i, k_layer in enumerate(k_parts):
            col1_dict[i] += list(range(28))
            col2_dict[i] += k_layer
            # len(k_layer) == 28

    fig = plt.figure(figsize=(6, 3))

    layer_pred_out = 1
    order = [1, 2, 3, 4, 5, 6, 7, 8]
    for i, o in enumerate(order):
        ax = plt.subplot(2, 4, o)
        df = pd.DataFrame({'time': col1_dict[i], 'obs': col2_dict[i]})
        sns.lineplot(x="time", y="obs", data=df, ax=ax)
        ax.set_title(o)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    fig.tight_layout()
    fig.savefig('figures/decomposition.pdf')


def plot_series_averages():
    f = h5py.File('data/vevo/vevo.hdf5', 'r')
    vevo_views = f['views'][...]
    vevo_views = vevo_views.mean(0)

    f = h5py.File('data/wiki/wiki.hdf5', 'r')
    wiki_views = f['views'][...]
    wiki_views = wiki_views.mean(0)

    fig = plt.figure(figsize=(6, 3))
    ax = plt.subplot(2, 1, 1)
    ax.plot(vevo_views)
    ax.set_xticks([0, 30, 62])
    ax.set_xticklabels(['1 Sep 18', '1 Oct 18', '2 Nov 18'])
    ax.set_xlim([0, 62])
    ax.axvspan(56, 62, color='grey', alpha=0.3, lw=0)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    ax = plt.subplot(2, 1, 2)
    ax.plot(wiki_views[-182:])
    ax.axvspan(154, 181, color='grey', alpha=0.3, lw=0)
    ax.set_xlim([0, 181])
    ax.set_xticks([0, 60, 121, 181])
    ax.set_xticklabels(['1 Jan 20', '1 Mar 20', '1 May 20', '30 Jun 20'])
    fig.tight_layout()

    fig.savefig('figures/series_averages.pdf')


def main():
    plot_missing_views()
    plot_missing_edges()

    generate_word_cloud('global_health')
    generate_word_cloud('programming_languages')
    generate_word_cloud('star_wars')
    generate_word_cloud('global_warming')

    plot_wiki_smape_boxplots()
    plot_layer_decompositions()
    plot_series_averages()


if __name__ == '__main__':
    main()
