import h5py
import numpy as np
import pandas as pd


def convert_data(data):
    speeds = pd.read_csv(f'data/taxi/{data}_speed.csv').to_numpy().transpose()
    n_roads, n_steps = speeds.shape
    # speeds.shape = [n_roads, n_steps]

    A = pd.read_csv(f'data/taxi/{data}_adj.csv', header=None).to_numpy()
    # A.shape = [n_roads, n_roads]

    data_path = f'data/taxi/{data}.h5'
    data_f = h5py.File(data_path, 'a')

    data_f.create_dataset('views', dtype=np.float64, data=speeds)

    int32_dt = h5py.vlen_dtype(np.dtype('int32'))
    edges = data_f.create_dataset('edges', (n_roads, n_steps), int32_dt)

    bool_dt = h5py.vlen_dtype(np.dtype('bool'))
    masks = data_f.create_dataset('masks', (n_roads, n_steps), bool_dt)

    max_edges = 0
    for i in range(n_roads):
        neighs = A[i].nonzero()[0][None, :]
        # neighs.shape == [1, max_neighs]

        neighs = neighs.repeat(n_steps, axis=0)
        # neighs.shape == [n_steps, max_neighs]

        edges[i] = neighs
        max_edges = max(neighs.shape[1], max_edges)

        masks[i] = np.zeros((n_steps, neighs.shape[1]))


def main():
    convert_data('los')  # max 26 edges
    convert_data('sz')  # max 6 edges


if __name__ == '__main__':
    main()
