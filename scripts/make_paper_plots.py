import json
import os
import pickle
from collections import defaultdict

import h5py
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
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


def plot_missing_expt(data):
    def get_paths(kind, data, n_hops):
        return [f'expt/missing_information/{kind}/{data}/{n_hops}/00/serialization/evaluate-metrics.json',
                f'expt/missing_information/{kind}/{data}/{n_hops}/20/serialization/evaluate-metrics.json',
                f'expt/missing_information/{kind}/{data}/{n_hops}/40/serialization/evaluate-metrics.json',
                f'expt/missing_information/{kind}/{data}/{n_hops}/60/serialization/evaluate-metrics.json',
                f'expt/missing_information/{kind}/{data}/{n_hops}/80/serialization/evaluate-metrics.json']

    y1 = get_means(get_paths('views', data, 'no_hops'))
    y2 = get_means(get_paths('views', data, 'one_hop'))
    y3 = get_means(get_paths('views', data, 'two_hops'))

    fig = plt.figure(figsize=(6, 3))
    ax = plt.subplot(1, 2, 1)

    xs = [0, 20, 40, 60, 80]
    ax.errorbar(xs, y1, marker='x', linestyle=':')
    ax.errorbar(xs, y2, marker='x', linestyle='--')
    ax.errorbar(xs, y3, marker='x', linestyle='-')

    ax.set_ylabel('SMAPE-7')
    ax.set_xlabel('% of missing views')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.legend(['No hops', 'One hop', 'Two hops'], frameon=False)
    ax.set_title('Missing View Effect')

    ax = plt.subplot(1, 2, 2)

    y1 = get_means(get_paths('edges', data, 'no_hops'))
    y2 = get_means(get_paths('edges', data, 'one_hop'))
    y3 = get_means(get_paths('edges', data, 'two_hops'))

    xs = [0, 20, 40, 60, 80]
    ax.errorbar(xs, y1, marker=None, linestyle=':')
    ax.errorbar(xs, y2, marker='x', linestyle='--')
    ax.errorbar(xs, y3, marker='x', linestyle='-')

    ax.set_xlabel('% of missing edges')
    ax.legend(['No hops', 'One hop', 'Two hops'],
              bbox_to_anchor=(0.47, 0.75), frameon=False)
    ax.set_title('Missing Edge Effect')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    fig.tight_layout()
    fig.savefig(f'figures/missing_{data}.pdf')


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
    path = 'expt/pure_time_series/wiki_univariate/08_radflow_no_network/serialization/evaluate-metrics.json'
    with open(path) as f:
        o3 = json.load(f)

    path = 'expt/network_aggregation/wiki_univariate/imputation/one_hop/15_radflow/serialization/evaluate-metrics.json'
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
    path = 'expt/pure_time_series/wiki_univariate/08_radflow_no_network/serialization/evaluate-metrics.json'
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

    f = h5py.File('data/wiki/views_split.hdf5', 'r')
    wiki_views = f['views'][...]
    wiki_views = wiki_views.mean(0)

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(2, 1, 1)
    ax.plot(vevo_views)
    ax.set_xticks([0, 30, 62])
    ax.set_xticklabels(['1 Sep 18', '1 Oct 18', '2 Nov 18'])
    ax.set_xlim([0, 62])
    ax.axvspan(56, 62, color='grey', alpha=0.3, lw=0)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_title('VevoMusic')

    ax = plt.subplot(2, 1, 2)
    ax.plot(wiki_views[-182:, 0], label='desktop')
    ax.plot(wiki_views[-182:, 1], label='non-desktop')
    ax.legend(loc='upper left')
    ax.axvspan(154, 181, color='grey', alpha=0.3, lw=0)
    ax.set_xlim([0, 181])
    ax.set_xticks([0, 60, 121, 181])
    ax.set_xticklabels(['1 Jan 20', '1 Mar 20', '1 May 20', '30 Jun 20'])
    ax.set_title('WikiTraffic')
    fig.tight_layout()

    fig.savefig('figures/series_averages.pdf')


def plot_edge_dist():
    f = h5py.File('data/vevo/vevo.hdf5', 'r')
    f2 = h5py.File('data/wiki/wiki.hdf5', 'r')

    masks = f['masks']
    masks2 = f2['masks']

    path = 'data/vevo/edge_counter.pkl'
    if not os.path.exists(path):
        counter = defaultdict(int)
        for node_masks in tqdm(masks):
            node_masks = ~np.vstack(node_masks).transpose()
            for n_mask in node_masks:
                counter[n_mask.sum()] += 1
        with open(path, 'wb') as f:
            pickle.dump(counter, f)

    with open(path, 'rb') as f:
        counter = pickle.load(f)

    path = 'data/wiki/edge_counter.pkl'
    if not os.path.exists(path):
        counter2 = defaultdict(int)
        for node_masks in tqdm(masks2):
            node_masks = ~np.vstack(node_masks).transpose()
            for n_mask in node_masks:
                counter2[n_mask.sum()] += 1
        with open(path, 'wb') as f:
            pickle.dump(counter2, f)

    with open(path, 'rb') as f:
        counter2 = pickle.load(f)

    fig = plt.figure(figsize=(6, 2.5))

    ax = plt.subplot(1, 2, 1)
    xs = range(1, 64)
    ys = [counter[i] for i in range(1, 64)]

    n_total = np.sum(ys)
    n_short = np.sum([counter[i] for i in range(1, 5)])
    print('One day Vevo links:', n_short / n_total)

    ax.plot(xs, ys)
    ax.set_xlabel('Number of Days')
    ax.set_ylabel('Number of Edges')
    ax.set_yscale('log')
    ax.set_title('VevoMusic')

    ax = plt.subplot(1, 2, 2)
    xs = range(1, 1828)
    ys = [counter2[i] for i in range(1, 1828)]

    n_total = np.sum(ys)
    n_long = np.sum([counter2[i] for i in range(1827, 1828)])
    print('Complete Wiki links:', n_long / n_total)

    ax.plot(xs, ys)
    ax.set_xlabel('Number of Days')
    ylabels = ['{:,.0f}'.format(x) + 'K' for x in ax.get_yticks()/1000]
    ax.set_yticklabels(ylabels)
    ax.set_yscale('log')
    ax.set_title('WikiTraffic')

    fig.tight_layout()
    fig.savefig('figures/edge_distribution.pdf')


def plot_network_contribution():
    sns.set_palette("muted")
    current_palette = sns.color_palette()

    node_ids = {}
    cats = ['programming_languages', 'global_health',
            'global_warming', 'star_wars']
    for cat in cats:
        path = f'data/wiki/node_ids/{cat}.pkl'
        with open(path, 'rb') as f:
            node_ids[cat] = pickle.load(f)

    f_vevo = h5py.File('data/vevo/vevo.hdf5', 'r')
    path = 'expt/network_aggregation/vevo_dynamic/imputation/two_hops/15_radflow/serialization/evaluate-metrics.json'
    with open(path) as f:
        o8 = json.load(f)

    views_list = []
    ratios = []
    for i, k_parts in enumerate(o8['f_parts']):
        raw_views = f_vevo['views'][o8['keys'][i], -28:]
        views_list.append(raw_views.mean())

        k_parts = np.array(k_parts)
        ratios.append(k_parts[-1].sum() / k_parts.sum() * 100)

    fig = plt.figure(figsize=(6, 3))
    ax = plt.subplot(1, 2, 1)
    ax.scatter(np.array(views_list), ratios, s=1, alpha=0.05, edgecolors=None)
    ax.set_ylim(0, 6)
    # ax.set_xlim(10, 40000)
    ax.set_xscale('log')
    ax.set_xlabel('Average Daily Views')
    ax.set_ylabel('Network Contribution (%)')
    ax.set_title('VevoMusic')
    ax.locator_params(nbins=10, axis='y')
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    f_wiki = h5py.File('data/wiki/wiki.hdf5', 'r')
    path = 'expt/network_aggregation/wiki_univariate/imputation/one_hop/15_radflow/serialization/evaluate-metrics.json'
    with open(path) as f:
        o8 = json.load(f)

    views_list = []
    ratios = []
    for i, k_parts in enumerate(o8['f_parts']):
        raw_views = f_wiki['views'][o8['keys'][i], -28:]
        views_list.append(raw_views.mean())
        k_parts = np.array(k_parts)
        ratios.append(k_parts[-1].sum() / k_parts.sum() * 100)

    ax = plt.subplot(1, 2, 2)
    views = np.array(views_list)
    ratios = np.array(ratios)
    names = ['programming', 'global health', 'global warming', 'star wars']
    circles = []
    for j, cat in enumerate(cats):
        idx = [i for i, k in enumerate(o8['keys']) if k in node_ids[cat]]
        ax.scatter(views[idx], ratios[idx], color=current_palette[j],
                   s=1, alpha=0.2, edgecolors=None, label=names[j])

        circles.append(plt.Line2D([0], [0], marker='o', color=current_palette[j], label=names[j],
                                  markersize=5, linestyle=''))

    ax.legend(handles=circles, prop={
              'size': 8}, loc="lower left", scatterpoints=1, frameon=False)

    ax.set_ylim(0, 6)
    ax.set_xlim(10, 40000)
    ax.set_xscale('log')
    ax.set_xlabel('Average Daily Views')
    ax.set_title('WikiTraffic')

    fig.tight_layout()
    fig.savefig('figures/network_contribution.png', dpi=300)


def plot_attention_maps():
    path = 'expt/network_aggregation/wiki_univariate/imputation/one_hop/15_radflow/serialization/evaluate-metrics.json'
    with open(path) as f:
        o5 = json.load(f)

    path = 'data/wiki/title2graphid.pkl'
    with open(path, 'rb') as f:
        title2graphid = pickle.load(f)
    id2title = {i: title for title, i in title2graphid.items()}

    graph_f = h5py.File('data/wiki/wiki.hdf5', 'r')

    test_pos = 1358
    test_nodeid = o5['keys'][test_pos]

    avg_attn = np.array(o5['all_scores'][test_pos]).mean(0)[:-2]
    best_neighs = np.argsort(avg_attn)[::-1][:6]
    best_keys = np.array(o5['neigh_keys'][test_pos])[best_neighs]
    sort_idx = np.argsort(best_keys)
    reverse_idx = np.argsort(sort_idx)

    sorted_best_keys = best_keys[sort_idx]
    best_series = graph_f['views'][sorted_best_keys, -28:]
    best_series = best_series[reverse_idx]

    o_series = graph_f['views'][test_nodeid, -28:]

    fig = plt.figure(figsize=(6, 3.5))

    order = [1, 3, 5]
    idx = [0, 2, 4]
    axes = []
    for i, o in zip(idx, order):
        neigh_idx = best_neighs[i]
        ax = plt.subplot(3, 2, o)
        s = best_series[i]
        ax.plot(np.log(s), c='black')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(id2title[best_keys[i]])
        axes.append(ax)

        ax.set_xlim(0, 27)
        for start in range(27):
            attn = o5['all_scores'][test_pos][start][neigh_idx]
            plt.axvspan(start, start+1,
                        color=cm.Blues(attn/(0.15)), alpha=0.7, lw=0)

    # contribs = np.array(o5['f_parts'][1358])
    # net_contrib = contribs[-1]
    # ser_contrib = contribs[:-1].sum(0)

    ax = plt.subplot(3, 2, 4)
    ax.plot(np.log(o_series), c='black')
    axes.append(ax)

    ax.set_title(id2title[test_nodeid])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 27)

    ax0tr = axes[0].transData  # Axis 0 -> Display
    ax1tr = axes[-1].transData  # Axis 1 -> Display
    figtr = fig.transFigure.inverted()  # Display -> Figure
    ptB = figtr.transform(ax0tr.transform((27, 0.5)))
    ptE = figtr.transform(ax1tr.transform((6, 0.9)))
    arrow = mpl.patches.FancyArrowPatch(
        ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
        fc="grey", connectionstyle="arc3,rad=-0.2", arrowstyle='simple', alpha=0.3,
        mutation_scale=20.
    )
    fig.patches.append(arrow)

    ax0tr = axes[0].transData  # Axis 0 -> Display
    ax1tr = axes[-1].transData  # Axis 1 -> Display
    figtr = fig.transFigure.inverted()  # Display -> Figure
    ptB = figtr.transform(ax0tr.transform((27, -0.8)))
    ptE = figtr.transform(ax1tr.transform((-0.9, 0.4)))
    arrow = mpl.patches.FancyArrowPatch(
        ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
        fc="grey", connectionstyle="arc3,rad=0", arrowstyle='simple', alpha=0.3,
        mutation_scale=20.
    )
    fig.patches.append(arrow)

    ax0tr = axes[0].transData  # Axis 0 -> Display
    ax1tr = axes[-1].transData  # Axis 1 -> Display
    figtr = fig.transFigure.inverted()  # Display -> Figure
    ptB = figtr.transform(ax0tr.transform((27, -2.1)))
    ptE = figtr.transform(ax1tr.transform((6, -0.1)))
    arrow = mpl.patches.FancyArrowPatch(
        ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
        fc="grey", connectionstyle="arc3,rad=0.2", arrowstyle='simple', alpha=0.3,
        mutation_scale=20, edgecolor=None,
    )
    fig.patches.append(arrow)

    fig.tight_layout(w_pad=2)
    fig.savefig('figures/attention_map.pdf')


def plot_corr_density():
    path = 'expt/network_aggregation/vevo_dynamic/imputation/one_hop/15_radflow/serialization/evaluate-metrics.json'
    with open(path) as f:
        o1 = json.load(f)

    path = 'expt/network_aggregation/wiki_univariate/imputation/one_hop/15_radflow/serialization/evaluate-metrics.json'
    with open(path) as f:
        o2 = json.load(f)

    views_wiki = h5py.File('data/wiki/wiki.hdf5', 'r')['views'][...]
    views_vevo = h5py.File('data/vevo/vevo.hdf5', 'r')['views'][...]

    s_vevo = []
    v_vevo = []
    for key, scores, neighs in tqdm(zip(o1['keys'], o1['all_scores'], o1['neigh_keys'])):
        scores = np.array(scores).transpose()[:-1]
        for i, n in enumerate(neighs):
            mean_score = scores[i].mean()
            if mean_score == 0:
                continue
            v_vevo.append(np.corrcoef(
                views_vevo[key, -49:], views_vevo[n, -49:])[0, 1])
            s_vevo.append(mean_score)

    s_wiki = []
    v_wiki = []
    for key, scores, neighs in tqdm(zip(o2['keys'], o2['all_scores'], o2['neigh_keys'])):
        scores = np.array(scores).transpose()[:-1]
        for i, n in enumerate(neighs):
            mean_score = scores[i].mean()
            if mean_score == 0:
                continue
            v_wiki.append(np.corrcoef(
                views_wiki[key, -140:], views_wiki[n, -140:])[0, 1])
            s_wiki.append(mean_score)

    fig = plt.figure(figsize=(6, 3))

    ax = plt.subplot(1, 2, 1)

    sns.kdeplot(s_vevo, v_vevo, cmap="Blues",
                shade=True, shade_lowest=True, ax=ax)
    ax.set_xlabel('Attention Score')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('VevoMusic')
    ax.set_xlim(0, 0.15)
    ax.set_ylim(-0.25, 1)

    ax = plt.subplot(1, 2, 2)

    sns.kdeplot(s_wiki, v_wiki, cmap="Blues",
                shade=True, shade_lowest=True, ax=ax)
    ax.set_xlabel('Attention Score')
    ax.set_title('WikiTraffic')
    ax.set_xlim(0, 0.15)
    ax.set_ylim(-0.25, 1)

    fig.tight_layout()
    fig.savefig('figures/corr_attn.pdf')


def plot_counterfactuals():
    path = 'expt/counterfactuals/doubling/serialization/evaluate-metrics.json'
    with open(path) as f:
        o = json.load(f)

    path = 'data/wiki/title2graphid.pkl'
    with open(path, 'rb') as f:
        title2graphid = pickle.load(f)
    id2title = {i: title for title, i in title2graphid.items()}

    deltas = []
    attns_2 = []
    attns_1 = []
    a_d = []
    for p1, p2, s1, s2 in zip(o['preds'], o['preds_2'], o['all_scores'], o['all_scores_2']):
        for d in range(28):
            deltas.append(p2[d]/p1[d] - 1)
            attns_1.append(s1[d][0])
            attns_2.append(s2[d][0])
            a_d.append(s2[d][0] - s1[d][0])

    fig = plt.figure(figsize=(6, 3))

    ax = plt.subplot(1, 2, 1)
    ax.scatter(attns_1, attns_2, s=1, alpha=0.02, edgecolors=None)
    ax.set_xlabel('Score Before Doubling')
    ax.set_ylabel('Score After Doubling')
    ax.plot([0, 1], [0, 1], linewidth=1, linestyle='--')
    ax.set_ylim(0, 0.8)
    ax.set_xlim(0, 0.8)
    ax.set_yticks(np.round(np.linspace(0, 0.8, 5), 1))
    ax.set_title('Changes in Attention Scores')

    ax = plt.subplot(1, 2, 2)
    ax.scatter(np.array(deltas) * 100, attns_2,
               s=1, alpha=0.02, edgecolors=None)
    ax.set_xlabel('% Increase in Views of Ego Node')
    ax.set_ylabel('Score After Doubling')
    ax.set_ylim(0, 0.8)
    ax.set_yticks(np.round(np.linspace(0, 0.8, 5), 1))
    ax.set_xlim(-5, 50)
    ax.set_title("Effect of Attention on Ego's Traffic")

    fig.tight_layout()
    fig.savefig('figures/doubling_neighs.png', dpi=300)


def plot_performance_by_traffic():
    path = f'expt/pure_time_series/wiki_univariate/08_radflow_no_network/serialization/evaluate-metrics.json'
    with open(path) as f:
        o = json.load(f)
        keys5 = o['keys']
        preds5 = o['preds']
        smapes5 = o['smapes']

    preds_agg = defaultdict(list)
    preds_none = defaultdict(list)

    for k, p5 in zip(keys5, smapes5):
        avg_view = sum(wiki_views[k]) / len(wiki_views[k])
        if avg_view < 25:
            preds_none[25] += p5
        elif avg_view < 50:
            preds_none[50] += p5
        elif avg_view < 100:
            preds_none[100] += p5
        elif avg_view < 200:
            preds_none[200] += p5
        elif avg_view < 500:
            preds_none[500] += p5
        elif avg_view < 1000:
            preds_none[1000] += p5
        elif avg_view < 3000:
            preds_none[3000] += p5
        else:
            preds_none[10000] += p5

    fig = plt.figure(figsize=(9, 4))
    ax = plt.subplot(1, 1, 1)

    smapes = [preds_none[25],
              preds_none[50],
              preds_none[100],
              preds_none[200],
              preds_none[500],
              preds_none[1000],
              preds_none[3000],
              preds_none[10000],
              ]

    ax.boxplot(smapes, showfliers=False, meanline=True,
               showmeans=True, widths=0.4)

    # means = [np.mean(x) for x in smapes]
    # pos = range(len(means))
    # for tick, label in zip(pos, ax.get_xticklabels()):
    #     ax.text(pos[tick] + 0.85, means[tick] +
    #             0.07, '{0:.1f}'.format(means[tick]))

    ax.set_xticklabels(
        ['0-24', '25-49', '50-99', '100-199', '200-499',
         '500-999',
         '1000-2999', '>=3000'])
    ax.set_ylabel('SMAPE-28')
    ax.set_xlabel('Average Daily View Counts')
    # ax.set_title('WikiTraffic Performance Split By Traffic Volume')
    fig.tight_layout()
    fig.savefig('figures/traffic_split.pdf')
    plt.show()


def plot_taxi_series():
    los_f = h5py.File('data/taxi/los.h5', 'r')
    sz_f = h5py.File('data/taxi/sz.h5', 'r')

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(np.median(los_f['views'][...], axis=0))
    ax.set_xlim(0, 2015)
    ax.set_title('Los-Loop')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(np.median(sz_f['views'][...], axis=0)[:])
    ax.set_title('SZ-Taxi')
    ax.set_xlim(0, 2975)

    fig.tight_layout()
    fig.savefig('figures/taxi_series.pdf')


def plot_taxi_error_distribution():
    sz_f = h5py.File('data/taxi/sz.h5', 'r')
    with open('expt/taxi/shenzhen/15_radflow/serialization/evaluate-metrics.json') as f:
        o15 = json.load(f)
    with open('expt/taxi/shenzhen/01_copying_previous_step/serialization/evaluate-metrics.json') as f:
        o1 = json.load(f)

    y = sz_f['views'][:, -4:]
    p1 = np.array(o1['preds'])
    p15 = np.array(o15['preds'])

    e1 = p1 - y
    e15 = p15 - y

    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.subplot(1, 2, 1)
    ax.hist(e1.flatten(), bins=30)
    ax.set_title('Copying Previous Step')
    ax.set_xlim(-15, 15)
    ax.set_xlabel('Prediction Error')

    ax = plt.subplot(1, 2, 2)
    ax.hist(e15.flatten(), bins=30)
    ax.set_title('Radflow')
    ax.set_xlim(-15, 15)
    ax.set_xlabel('Prediction Error')

    fig.tight_layout()
    fig.savefig('figures/taxi_errors.pdf')


plot_taxi_error_distribution()


def main():
    plot_missing_expt('vevo')
    plot_missing_expt('wiki')

    generate_word_cloud('global_health')
    generate_word_cloud('programming_languages')
    generate_word_cloud('star_wars')
    generate_word_cloud('global_warming')

    plot_wiki_smape_boxplots()
    plot_layer_decompositions()
    plot_series_averages()
    plot_edge_dist()
    plot_network_contribution()
    plot_attention_maps()
    plot_corr_density()
    plot_counterfactuals()
    plot_performance_by_traffic()

    plot_taxi_series()
    plot_taxi_error_distribution()


if __name__ == '__main__':
    main()
