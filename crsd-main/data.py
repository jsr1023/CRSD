from pathlib import Path
import numpy as np
import torch
import dgl
from dgl.data import FraudAmazonDataset, FraudYelpDataset
from dgl.data.utils import load_graphs
from scipy.sparse import csr_matrix
from models import Graph

DATASETS = ['yelp', 'amazon', 'elliptic', 'tfinance']


def labels_of(value):
    return (value.argmax(-1) if value.ndim > 1 else value).long().view(-1)


def load_raw(name, root):
    root = Path(root)
    if name in ('yelp', 'amazon'):
        dataset = FraudYelpDataset if name == 'yelp' else FraudAmazonDataset
        hetero = dataset(raw_dir=str(root), verbose=False)[0]
        node_data = hetero.nodes[hetero.ntypes[0]].data
        x = torch.nan_to_num(node_data['feature'].float())
        y = labels_of(node_data['label'])
        relations = [dgl.remove_self_loop(dgl.edge_type_subgraph(hetero, [etype])) for etype in hetero.canonical_etypes]
        homo = dgl.to_homogeneous(hetero, ndata=['feature', 'label'])
    else:
        path = root / name
        if path.is_dir():
            path = path / name
        homo = load_graphs(str(path))[0][0]
        if not homo.is_homogeneous:
            homo = dgl.to_homogeneous(homo, ndata=['feature', 'label'])
        x = torch.nan_to_num(homo.ndata['feature'].float())
        y = labels_of(homo.ndata['label'])
        relations = [dgl.remove_self_loop(homo)]
    merged = dgl.add_self_loop(dgl.remove_self_loop(homo))
    return x, y.numpy(), relations, merged


def csr(graph):
    indptr, indices, _ = graph.adj_tensors('csc')
    n = graph.num_nodes()
    matrix = csr_matrix((np.ones(indices.numel(), dtype=np.float32), indices.numpy(), indptr.numpy()), shape=(n, n))
    matrix.sum_duplicates()
    matrix.sort_indices()
    adjacency = torch.sparse_csr_tensor(torch.from_numpy(matrix.indptr.astype(np.int64)), torch.from_numpy(matrix.indices.astype(np.int64)),
                                        torch.from_numpy(matrix.data.copy()), size=matrix.shape)
    return adjacency, (indptr[1:] - indptr[:-1]).float()


def prepare(name, root, device):
    cache = Path(root) / 'processed' / f'{name}.pt'
    if cache.exists():
        return torch.load(cache, map_location='cpu', weights_only=False)
    x, y, relations, merged = load_raw(name, root)
    xd = x.to(device)
    means = []
    for relation in relations:
        means.append(Graph.from_dgl(relation, device).mean(xd).cpu().numpy())
        torch.cuda.empty_cache()
    raw = x.numpy()
    views = np.concatenate([raw[:, None, :], np.stack(means, 1), np.stack([raw - m for m in means], 1)], 1)
    adjacency, degree = csr(merged)
    data = dict(x=x, y=y, views=views, adjacency=adjacency, degree=degree)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache)
    return data


def split(y, seed):
    rng = np.random.RandomState(seed)
    positive = rng.choice(np.flatnonzero(y == 1), 40, replace=False)
    negative = rng.choice(np.flatnonzero(y == 0), 160, replace=False)
    train = np.sort(np.r_[positive[:20], negative[:80]])
    validation = np.sort(np.r_[positive[20:], negative[80:]])
    known = np.flatnonzero((y == 0) | (y == 1))
    test = np.setdiff1d(known, np.r_[train, validation])
    return train, validation, test
