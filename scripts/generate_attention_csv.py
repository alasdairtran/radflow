import json

import numpy as np
from tqdm import tqdm


def convert_expt_output_to_csv(in_path, out_path):
    with open(in_path) as fi:
        o = json.load(fi)

    keys = o['keys']
    neighs = o['neigh_keys']
    scores = o['all_scores']

    with open(out_path, 'w') as fo:
        fo.write('graphid,neighbour,attention\n')
        for k, ns, ss in zip(keys, neighs, scores):
            ss = np.array(ss).transpose().tolist()
            for n, s in zip(ns, ss):
                if n == -1:
                    continue
                score_str = ';'.join([str(x) for x in s])
                out_line = f"{k},{n},{score_str}\n"
                fo.write(out_line)


def convert_to_csv(in_path, out_path):
    with open(in_path) as fi:
        with open(out_path, 'w') as fo:
            fo.write('graphid,neighbour,attention\n')

            for line in tqdm(fi):
                o = json.loads(line)
                neighs = o['n']
                scores = np.array(o['s']).transpose()[:-2]
                for n, s in zip(neighs, scores):
                    if n == -1:
                        continue
                    score_str = ';'.join([str(x) for x in s.tolist()])
                    out_line = f"{o['k']},{n},{score_str}\n"
                    fo.write(out_line)


def main():
    in_path = 'expt/network_aggregation/wiki_univariate/reports/test/serialization/scores.jsonl'
    out_path = 'expt/network_aggregation/wiki_univariate/reports/test/serialization/wiki_attention_test.csv'
    convert_to_csv(in_path, out_path)

    in_path = 'expt/network_aggregation/wiki_univariate/reports/train/serialization/scores.jsonl'
    out_path = 'expt/network_aggregation/wiki_univariate/reports/train/serialization/wiki_attention_train.csv'
    convert_to_csv(in_path, out_path)

    # To extract attention scores from outputs of vevo experiments. First run
    # CUDA_VISIBLE_DEVICES=0 radflow evaluate \
    #   expt/network_aggregation/vevo_dynamic/imputation/one_hop/15_radflow/config.yaml \
    #   -m expt/network_aggregation/vevo_dynamic/imputation/one_hop/15_radflow/serialization/best.th \
    #   -s train --overrides '{"dataset_reader": {"test_path": "data/vevo/vevo_all_nodes.pkl"}}'
    in_path = 'expt/network_aggregation/vevo_dynamic/imputation/one_hop/15_radflow/serialization/evaluate-metricstrain.json'
    out_path = 'vevo_attention_scores.csv'
    convert_expt_output_to_csv(in_path, out_path)


if __name__ == '__main__':
    main()
