import json
from itertools import product

import numpy as np
from scipy.stats import ttest_rel

ds = {
    'los_angeles': r'\LA',
    'shenzhen': r'\SZ',
    'vevo': r'\Vevo',
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
    'no_hops': '0H',
    'one_hop': '1H',
    'two_hops': '2H',
}

ms = {
    '01_copying_previous_step': r'(1) Copying Previous Step',
    '01_copying_previous_day': r'(1) Copying Previous Day',
    '02_copying_previous_week': r'(2) Copying Previous Week',
    '05_seasonal_arima': r'(5) Seasonal ARIMA',
    '06_lstm': r'(6) LSTM',
    '07_nbeats': r'(7) N-BEATS',
    '08_radflow_no_network': r'(8) \modelname-NoNetwork',
    '09_tgcn': r'(9) T-GCN',
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


def print_row(d, k1, h1, m1, s1, k2, h2, m2, s2):
    pvalue = ttest_rel(s1, s2)[1]
    if pvalue < 0.0001:
        return
    print(f"{ds[d]} & "
          f"{ms[m1]} & {ks[k1]} & {hs[h1]} & {round_to(s1.mean())} & "
          f"{ms[m2]} & {ks[k2]} & {hs[h2]} & {round_to(s2.mean())} & "
          f"{pvalue:.3f} \\\\")


def generate_pvalue_rows():
    datas = ['vevo', 'vevo_static', 'vevo_dynamic',
             'wiki_univariate', 'wiki_bivariate']
    kinds = ['forecast', 'imputation']
    hops = ['no_hops', 'one_hop', 'two_hops']
    network_models = ['11_lstm_mean', '12_radflow_mean',
                      '13_radflow_graphsage', '14_radflow_gat', '15_radflow']
    series_models = ['01_copying_previous_day', '02_copying_previous_week',
                     '05_seasonal_arima', '06_lstm', '07_nbeats',
                     '08_radflow_no_network']
    models = network_models + series_models

    def is_valid_combo(d, m, h, k):
        if m in series_models:
            return h == 'no_hops' and k == 'forecast' and d not in ['vevo_static', 'vevo_dynamic']
        else:
            return d != 'vevo' and h != 'no_hops'

    for data in datas:
        combs = list(product(kinds, hops, models))
        for i in range(len(combs)):
            for j in range(i, len(combs)):
                k1, h1, m1 = combs[i]
                k2, h2, m2 = combs[j]
                if k1 == k2 and h1 == h2 and m1 == m2:
                    continue
                if not is_valid_combo(data, m1, h1, k1):
                    continue
                if not is_valid_combo(data, m2, h2, k2):
                    continue

                if m1 in series_models:
                    path1 = f'./expt/pure_time_series/{data}/{m1}/serialization/evaluate-metrics.json'
                else:
                    path1 = f'./expt/network_aggregation/{data}/{k1}/{h1}/{m1}/serialization/evaluate-metrics.json'
                if m2 in series_models:
                    path2 = f'./expt/pure_time_series/{data}/{m2}/serialization/evaluate-metrics.json'
                else:
                    path2 = f'./expt/network_aggregation/{data}/{k2}/{h2}/{m2}/serialization/evaluate-metrics.json'
                with open(path1) as f:
                    smapes1 = json.load(f)['smapes']
                    smapes1 = np.array(smapes1).reshape(-1)
                with open(path2) as f:
                    smapes2 = json.load(f)['smapes']
                    smapes2 = np.array(smapes2).reshape(-1)

                print_row(data, k1, h1, m1, smapes1, k2, h2, m2, smapes2)


def generate_ablation_pvalues():
    datas = ['vevo_static', 'vevo_dynamic']
    models = ['16_radflow_h', '17_radflow_p', '18_radflow_q', '19_radflow_hp',
              '20_radflow_hpq', '21_radflow_no_final_proj', '22_radflow_one_head']

    for data in datas:
        for model in models:
            path1 = f'./expt/network_aggregation/{data}/imputation/one_hop/15_radflow/serialization/evaluate-metrics.json'
            path2 = f'./expt/ablation_studies/{data}/{model}/serialization/evaluate-metrics.json'
            with open(path1) as f:
                smapes1 = json.load(f)['smapes']
                smapes1 = np.array(smapes1).reshape(-1)
            with open(path2) as f:
                smapes2 = json.load(f)['smapes']
                smapes2 = np.array(smapes2).reshape(-1)

            pvalue = ttest_rel(smapes1, smapes2)[1]
            print(data, model, smapes1.mean(), smapes2.mean(), pvalue)


def print_taxi_row(d, m1, s1, m2, s2):
    pvalue = ttest_rel(s1, s2)[1]
    if pvalue < 0.0001:
        return
    print(f"{ds[d]} & "
          f"{ms[m1]} & {round_to(s1.mean())} & "
          f"{ms[m2]} & {round_to(s2.mean())} & "
          f"{pvalue:.3f} \\\\")


def generate_taxi_pvalue_rows():
    datas = ['los_angeles', 'shenzhen']
    models = ['01_copying_previous_step', '08_radflow_no_network',
              '09_tgcn', '15_radflow']

    for data in datas:
        for i in range(len(models)):
            for j in range(i, len(models)):
                m1 = models[i]
                m2 = models[j]
                if m1 == m2:
                    continue

                path1 = f'./expt/taxi/{data}/{m1}/serialization/evaluate-metrics.json'
                path2 = f'./expt/taxi/{data}/{m2}/serialization/evaluate-metrics.json'

                with open(path1) as f:
                    smapes1 = json.load(f)['smapes']
                    smapes1 = np.array(smapes1).reshape(-1)
                with open(path2) as f:
                    smapes2 = json.load(f)['smapes']
                    smapes2 = np.array(smapes2).reshape(-1)

                print_taxi_row(data, m1, smapes1, m2, smapes2)


def generate_main_paper_pvalues():
    pairs = [
        ['./expt/pure_time_series/vevo/08_radflow_no_network/serialization/evaluate-metrics.json',
         './expt/network_aggregation/vevo_static/forecast/one_hop/15_radflow/serialization/evaluate-metrics.json'],

        ['./expt/pure_time_series/wiki_univariate/08_radflow_no_network/serialization/evaluate-metrics.json',
         './expt/network_aggregation/wiki_univariate/forecast/one_hop/13_radflow_graphsage/serialization/evaluate-metrics.json'],

        ['./expt/network_aggregation/vevo_static/imputation/one_hop/15_radflow/serialization/evaluate-metrics.json',
         './expt/network_aggregation/vevo_static/imputation/two_hops/15_radflow/serialization/evaluate-metrics.json'],

        ['./expt/network_aggregation/vevo_dynamic/imputation/one_hop/15_radflow/serialization/evaluate-metrics.json',
         './expt/network_aggregation/vevo_dynamic/imputation/two_hops/15_radflow/serialization/evaluate-metrics.json'],
    ]

    for path1, path2 in pairs:
        with open(path1) as f:
            smapes1 = json.load(f)['smapes']
            smapes1 = np.array(smapes1).reshape(-1)

        with open(path2) as f:
            smapes2 = json.load(f)['smapes']
            smapes2 = np.array(smapes2).reshape(-1)

        pvalue = ttest_rel(smapes1, smapes2)[1]
        print(smapes1.mean(), smapes2.mean(), pvalue)


def main():
    generate_pvalue_rows()
    generate_taxi_pvalue_rows()
    generate_ablation_pvalues()


if __name__ == '__main__':
    main()
