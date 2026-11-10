import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
import dgl
from torch.nn import functional as F
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from data import DATASETS, prepare, split
from models import Graph, SpectralBackbone, CalibrationBranch, rank_features

CONFIG = {
    'yelp': dict(hidden=64, min_leaf=5, weights=None, calibration=None),
    'amazon': dict(hidden=32, min_leaf=30, weights=(.5, .5, 0.), calibration=None),
    'elliptic': dict(hidden=64, min_leaf=30, weights=(.05, .05, .9), calibration='gated'),
    'tfinance': dict(hidden=64, min_leaf=30, weights=(.65, .3, .05), calibration='uniform'),
}
TREE = dict(max_iter=200, max_leaf_nodes=7, learning_rate=.05, l2_regularization=5, max_bins=255, early_stopping=False)
EPOCHS = 200
RATIO = 4.
THETA = .5


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def selection_score(y, p):
    values = np.unique(np.r_[.01, np.quantile(p, np.linspace(0, 1, 501)), .99])
    order = np.argsort(p, kind='stable')
    sp, sy = p[order], y[order]
    pref = np.r_[0, np.cumsum(sy == 1)]
    cut = np.searchsorted(sp, values, side='left')
    fn = pref[cut]
    tp = pref[-1] - fn
    fp = len(y) - cut - tp
    tn = cut - fn
    mf = tp / np.maximum(2 * tp + fp + fn, 1) + tn / np.maximum(2 * tn + fp + fn, 1)
    t = values[np.argmax(mf)]
    return f1_score(y, p >= t, average='macro', zero_division=0) + roc_auc_score(y, p)


def tree_evidence(views, y, train, seed, min_leaf):
    r = (views.shape[1] - 1) // 2
    evidence = []
    for j, positions in enumerate([[0] + list(range(1, r + 1)), [0] + list(range(r + 1, 2 * r + 1))]):
        features = views[:, positions, :].reshape(len(views), -1)
        tree = HistGradientBoostingClassifier(**TREE, min_samples_leaf=min_leaf, random_state=seed + j * 100003)
        tree.fit(features[train], y[train], sample_weight=np.where(y[train] == 1, RATIO, 1.))
        p = tree.predict_proba(features)[:, 1].clip(1e-5, 1 - 1e-5)
        evidence.append(np.log(p / (1 - p)).astype(np.float32).astype(float))
    return evidence


def spectral_margin(data, y, train, validation, seed, hidden, device):
    seed_all(seed)
    graph = Graph(data['adjacency'].to(device), data['degree'].long().to(device))
    x = data['x'].to(device)
    target = torch.as_tensor(np.where(y >= 0, y, 0), device=device)
    tr = torch.as_tensor(train, device=device)
    model = SpectralBackbone(x.shape[1], hidden, graph).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=0.)
    weight = x.new_tensor([1., RATIO])
    best, best_margin = -float('inf'), None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, s = model(x)
        yt, st = target[tr], s[tr]
        ce = F.cross_entropy(logits[tr], yt, weight=weight)
        bce = F.binary_cross_entropy_with_logits(st, yt.float(), pos_weight=x.new_tensor(RATIO))
        rank = F.softplus(.5 - st[yt == 1].mean() + st[yt == 0].mean())
        (ce + .25 * bce + .1 * rank).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                z, _ = model(x)
            p = z.softmax(1)[:, 1].cpu().numpy()
            score = selection_score(y[validation], p[validation])
            if score > best + 1e-10:
                best, best_margin = score, (z[:, 1] - z[:, 0]).cpu().numpy()
    return best_margin.astype(float)


def calibration_margin(data, y, train, validation, seed, gated, device):
    seed_all(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    raw = data['x'].numpy()
    views = rank_features(data['views'].reshape(len(raw), -1), train, device)
    x = torch.from_numpy(rank_features(raw, train, device)).double().to(device)
    adjacency = data['adjacency'].double().to(device)
    degree = data['degree'].double().to(device).clamp_min(1)[:, None]
    p1 = torch.sparse.mm(adjacency, x) / degree
    p2 = torch.sparse.mm(adjacency, p1) / degree
    bands = torch.stack([x, p1, p2, x - p1, p1 - p2], 1).float().cpu().numpy()
    del adjacency, x, p1, p2
    anchor_index = np.sort(np.random.RandomState(seed + 5801).choice(len(raw), min(256, len(raw)), replace=False))
    seed_all(seed)
    anchors = torch.from_numpy(bands[anchor_index, 0].copy()).to(device)
    tb, tv = torch.from_numpy(bands[train]).to(device), torch.from_numpy(views[train]).to(device)
    vb, vv = torch.from_numpy(bands[validation]).to(device), torch.from_numpy(views[validation]).to(device)
    yt = torch.from_numpy(y[train]).float().to(device)
    model = CalibrationBranch(bands.shape[2], views.shape[1], len(train), gated).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    pos_weight = yt.new_tensor(float((y[train] == 0).sum() / (y[train] == 1).sum()) ** .5)
    best, state = -float('inf'), None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(tb, tv, anchors), yt, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0:
            model.eval()
            replica = copy.deepcopy(model).double().eval()
            with torch.no_grad():
                margin = replica(vb.double(), vv.double(), anchors.double()).cpu().numpy()
            score = selection_score(y[validation], expit(margin))
            if score > best + 1e-10:
                best, state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model = CalibrationBranch(bands.shape[2], views.shape[1], len(train), gated).double().to(device)
    model.load_state_dict(state)
    model.eval()
    anchors = anchors.double()
    margins = []
    with torch.no_grad():
        for i in range(0, len(bands), 4096):
            b = torch.from_numpy(bands[i:i + 4096]).double().to(device)
            v = torch.from_numpy(views[i:i + 4096]).double().to(device)
            margins.append(model(b, v, anchors).cpu().numpy())
    torch.use_deterministic_algorithms(False)
    return np.concatenate(margins).astype(float)


def align(reference, value, train):
    slope = np.clip(np.std(reference[train]) / max(np.std(value[train]), 1e-8), .1, 10)
    return slope * value + (np.mean(reference[train]) - slope * np.mean(value[train]))


def fuse(name, spectral, context, deviation, train):
    if name == 'yelp':
        t = .75 * context + .25 * deviation
        slope = np.clip(t[train].std() / max(spectral[train].std(), 1e-8), .1, 10)
        shift = t[train].mean() - slope * spectral[train].mean()
        bound = max(2 * t[train].std(), .1)
        gap = np.abs(context - deviation)
        gate = gap / (gap + max((context[train] - deviation[train]).std(), .1))
        return t + .6 * gate * np.clip(slope * spectral + shift - t, -bound, bound)
    ws, wc, wd = CONFIG[name]['weights']
    return (ws * spectral + wc * context + wd * deviation) / (ws + wc + wd)


def anomaly_homophily(data, labels):
    adjacency = data['adjacency']
    crow, col = adjacency.crow_indices().numpy(), adjacency.col_indices().numpy()
    same = total = 0
    for v in np.flatnonzero(labels == 1):
        neighbors = col[crow[v]:crow[v + 1]]
        neighbors = neighbors[(neighbors != v) & (labels[neighbors] >= 0)]
        same += int((labels[neighbors] == 1).sum())
        total += len(neighbors)
    return same / total if total else 0.


def smooth(z, data, theta):
    zs = (z - z.mean()) / z.std()
    if theta == 0:
        return zs
    degree = data['degree'].double().numpy()
    neighbors = torch.sparse.mm(data['adjacency'].double(), torch.from_numpy(zs)[:, None]).squeeze(1).numpy()
    return (1 - theta) * zs + theta * np.where(degree > 0, neighbors / np.maximum(degree, 1), zs)


def macro_f1_curve(y, s, candidates):
    order = np.argsort(s)
    ss, ys = s[order], y[order]
    positives, negatives = ys.sum(), len(y) - ys.sum()
    cut = np.searchsorted(ss, candidates, side='left')
    tn = np.r_[0, np.cumsum(ys == 0)][cut]
    fn = np.r_[0, np.cumsum(ys == 1)][cut]
    tp, fp = positives - fn, negatives - tn
    return .5 * (2 * tp / np.maximum(2 * tp + fp + fn, 1) + 2 * tn / np.maximum(2 * tn + fp + fn, 1))


def plateau_median(y, s):
    u = np.unique(s)
    candidates = np.r_[u[0] - 1, (u[1:] + u[:-1]) / 2, u[-1] + 1]
    values = macro_f1_curve(y, s, candidates)
    return float(np.median(candidates[values >= values.max() - 1e-12]))


def select_threshold(y, s, resamples=200):
    rng = np.random.RandomState(0)
    thresholds = []
    for _ in range(resamples):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() != y[i].max():
            thresholds.append(plateau_median(y[i], s[i]))
    return float(np.median(thresholds))


def run(name, seed, data, device):
    cfg = CONFIG[name]
    y = data['y']
    train, validation, test = split(y, seed)
    labels = np.full(len(y), -1, dtype=np.int64)
    labels[train], labels[validation] = y[train], y[validation]
    context, deviation = tree_evidence(data['views'], labels, train, seed, cfg['min_leaf'])
    spectral = spectral_margin(data, labels, train, validation, seed, cfg['hidden'], device)
    if cfg['calibration']:
        calibration = calibration_margin(data, labels, train, validation, seed, cfg['calibration'] == 'gated', device)
        spectral = .75 * spectral + .25 * align(spectral, calibration, train)
    theta = THETA if anomaly_homophily(data, labels) > .5 else 0.
    s = smooth(fuse(name, spectral, context, deviation, train), data, theta)
    tau = select_threshold(labels[validation], s[validation])
    return 100 * f1_score(y[test], s[test] >= tau, average='macro'), 100 * roc_auc_score(y[test], s[test])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 52, 62])
    parser.add_argument('--data-root', default=str(Path(__file__).resolve().parent / 'dataset'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = {}
    for name in args.datasets:
        data = prepare(name, args.data_root, args.device)
        rows = []
        for seed in args.seeds:
            mf1, auc = run(name, seed, data, args.device)
            rows.append((mf1, auc))
            print(f'{name} seed {seed}: MF1 {mf1:.2f} AUC {auc:.2f}', flush=True)
        rows = np.array(rows)
        sd = rows.std(0, ddof=1) if len(rows) > 1 else np.zeros(2)
        results[name] = dict(MF1=rows[:, 0].mean(), MF1_std=sd[0], AUC=rows[:, 1].mean(), AUC_std=sd[1], per_seed=rows.tolist())
        print(f'{name}: MF1 {rows[:, 0].mean():.2f}±{sd[0]:.2f} AUC {rows[:, 1].mean():.2f}±{sd[1]:.2f}', flush=True)
        del data
        torch.cuda.empty_cache()
    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
