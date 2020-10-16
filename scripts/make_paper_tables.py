import json
from itertools import product

import numpy as np
from scipy.stats import ttest_rel

ds = {
    'vevo_static': r'\Vevo (static)',
    'vevo_dynamic': r'\Vevo (dynamic)',
    'wiki_univariate': r'\Wiki (uni.)',
    'wiki_bivariate': r'\Wiki (bi.)',
}

ks = {
    'forecast': 'Fore.',
    'imputation': 'Imp.',
}

hs = {
    'one_hop': '1H',
    'two_hops': '2H',
}

ms = {
    '11_lstm_mean': r'(11) LSTM-MeanPooling',
    '12_radflow_mean': r'(12) \modelname-MeanPooling',
    '13_radflow_graphsage': r'(13) \modelname-GraphSage',
    '14_radflow_gat': r'(14) \modelname-GAT',
    '15_radflow': r'(15) \modelname',
}


def round_to(x):
    if x < 10:
        return f'{x:.2f}'
    else:
        return f'{x:.1f}'


def print_row(d1, k1, h1, m1, s1, d2, k2, h2, m2, s2):
    pvalue = ttest_rel(s1, s2)[1]
    if pvalue < 0.0001:
        return
    print(f"{ms[m1]} & {ds[d1]} & {ks[k1]} & {hs[h1]} & {round_to(s1.mean())} & "
          f"{ms[m2]} & {ds[d2]} & {ks[k2]} & {hs[h2]} & {round_to(s2.mean())} & "
          f"{pvalue:.3f} \\\\")


def generate_pvalue_rows():
    datas = ['vevo_static', 'vevo_dynamic',
             'wiki_univariate', 'wiki_bivariate']
    kinds = ['forecast', 'imputation']
    hops = ['one_hop', 'two_hops']
    models = ['11_lstm_mean', '12_radflow_mean',
              '13_radflow_graphsage', '14_radflow_gat', '15_radflow']
    for data in datas:
        combs = list(product(kinds, hops, models))
        for i in range(len(combs)):
            for j in range(i, len(combs)):
                k1, h1, m1 = combs[i]
                k2, h2, m2 = combs[j]
                if k1 == k2 and h1 == h2 and m1 == m2:
                    continue

                path1 = f'./expt/network_aggregation/{data}/{k1}/{h1}/{m1}/serialization/evaluate-metrics.json'
                path2 = f'./expt/network_aggregation/{data}/{k2}/{h2}/{m2}/serialization/evaluate-metrics.json'
                with open(path1) as f:
                    smapes1 = json.load(f)['smapes']
                    smapes1 = np.array(smapes1).reshape(-1)
                with open(path2) as f:
                    smapes2 = json.load(f)['smapes']
                    smapes2 = np.array(smapes2).reshape(-1)

                print_row(data, k1, h1, m1, smapes1, data, k2, h2, m2, smapes2)


def main():
    generate_pvalue_rows()


if __name__ == '__main__':
    main()
