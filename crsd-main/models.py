import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Graph:
    def __init__(self, adjacency, degree):
        self.adjacency = adjacency
        self.degree = degree

    @classmethod
    def from_dgl(cls, graph, device):
        n = graph.num_nodes()
        if torch.device(device).type == 'cuda' and graph.num_edges() > 10_000_000:
            indptr, source, _ = graph.adj_tensors('csc')
            indptr = indptr.to(device=device, dtype=torch.int32)
            source = source.to(device=device, dtype=torch.int32)
            values = torch.ones(source.numel(), dtype=torch.float32, device=device)
            return cls(torch.sparse_csr_tensor(indptr, source, values, (n, n), device=device), (indptr[1:] - indptr[:-1]).long())
        source, target = graph.edges()
        source = source.to(device=device, dtype=torch.long)
        target = target.to(device=device, dtype=torch.long)
        values = torch.ones(target.shape[0], dtype=torch.float32, device=device)
        adjacency = torch.sparse_coo_tensor(torch.stack([target, source]), values, (n, n), device=device)
        if target.numel() <= 10_000_000:
            adjacency = adjacency.coalesce()
        return cls(adjacency, torch.bincount(target, minlength=n))

    def sum(self, values):
        return torch.nan_to_num(torch.sparse.mm(self.adjacency, values))

    def mean(self, values):
        mean = torch.nan_to_num(torch.sparse.mm(self.adjacency, values) / self.degree.clamp_min(1).to(values.dtype).unsqueeze(1))
        return torch.where(self.degree.gt(0).unsqueeze(1), mean, values)


def mlp(din, h, dout, dropout=.1):
    return nn.Sequential(nn.Linear(din, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, dout))


class FilterBank(nn.Module):
    def __init__(self, h, graph):
        super().__init__()
        self.graph = graph
        self.adapters = nn.ModuleList([mlp(h, h, h) for _ in range(5)])
        for adapter in self.adapters:
            nn.init.normal_(adapter[-1].weight, std=.01)
            nn.init.zeros_(adapter[-1].bias)
        self.gate = mlp(h + 3, h, 5)
        self.norm = nn.LayerNorm(h)

    def forward(self, base, weights):
        d = self.graph.degree.float().clamp_min(1).rsqrt().unsqueeze(1)
        l1 = base - self.graph.sum(base * d) * d
        l2 = l1 - self.graph.sum(l1 * d) * d
        bands = [3 * base - 3 * l1 + .75 * l2, 3 * l1 - 1.5 * l2, .75 * l2, l1, l2]
        corrections = torch.stack([adapter(band - base) for adapter, band in zip(self.adapters, bands)], 1)
        return self.norm(base + (weights.unsqueeze(-1) * corrections).sum(1)), corrections


class SpectralBackbone(nn.Module):
    def __init__(self, din, h, graph):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(din, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(.1))
        self.bank = FilterBank(h, graph)
        self.fusion = mlp(h, h, h)
        self.classifier = nn.Linear(h, 2)
        self.band_head = mlp(h + 1, h, 1)
        self.spectral_head = mlp(h + 2, h, 1)

    def forward(self, x):
        base = self.encoder(x)
        weights = base.new_full((len(base), 5), .2)
        z, corrections = self.bank(base, weights)
        logits = self.classifier(self.fusion(z))
        energy = corrections.norm(dim=2)
        concentration = 1 + (weights * weights.clamp_min(1e-8).log()).sum(1) / math.log(5)
        s = self.spectral_head(torch.cat([z, (weights * energy).sum(1, keepdim=True), concentration.unsqueeze(1)], 1)).squeeze(1)
        s = s + (weights * self.band_head(torch.cat([corrections, energy.unsqueeze(-1)], 2)).squeeze(-1)).sum(1)
        return torch.stack([logits[:, 0], logits[:, 1] + .5 * s], 1), s


def encoder(d, h):
    return nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU())


class CalibrationBranch(nn.Module):
    def __init__(self, dim, viewdim, ntrain, gated, h=64):
        super().__init__()
        self.gated = gated
        self.shrink = ntrain / (ntrain + 400.)
        self.self_encoder = encoder(dim, h)
        self.local_encoders = nn.ModuleList([encoder(dim, h) for _ in range(4)])
        self.view_encoder = encoder(viewdim, h)
        self.prototype_assignment = nn.Linear(h, 16)
        self.q = nn.Linear(h, h)
        self.k = nn.Linear(h, h)
        self.global_encoder = encoder(2 * h, h)
        self.gate = nn.Sequential(nn.Linear(h + 3, h), nn.GELU(), nn.Linear(h, 6))
        self.mix = nn.Parameter(torch.tensor(-1.5))
        self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(.1), nn.Linear(h, 1))

    def prototypes(self, anchors):
        z = self.self_encoder(anchors)
        assignment = self.prototype_assignment(z).softmax(1)
        return (assignment.T @ z) / assignment.sum(0).clamp_min(1e-8)[:, None]

    def forward(self, bands, views, anchors):
        z = self.self_encoder(bands[:, 0])
        paths = [enc(bands[:, j + 1]) for j, enc in enumerate(self.local_encoders)]
        paths.append(self.view_encoder(views))
        if self.gated:
            proto = self.prototypes(anchors)
            attention = (F.normalize(self.q(z), dim=1) @ F.normalize(self.k(proto), dim=1).T * 4).softmax(1)
            context = attention @ proto
            paths.append(self.global_encoder(torch.cat([context, z - context], 1)))
            stats = torch.stack([bands[:, 3].square().mean(1).sqrt(), bands[:, 4].square().mean(1).sqrt(), (bands[:, 0] - bands[:, 2]).abs().mean(1)], 1)
            raw = self.gate(torch.cat([z, stats], 1))[:, :len(paths)].softmax(1)
            weights = self.shrink * raw + (1 - self.shrink) / len(paths)
        else:
            weights = z.new_full((len(z), len(paths)), 1 / len(paths))
        z = z + self.mix.sigmoid() * (weights[:, :, None] * torch.stack(paths, 1)).sum(1)
        return self.head(z).squeeze(-1)


def rank_features(values, train, device, chunk=8192):
    q = np.quantile(values[train], np.linspace(0, 1, 33), axis=0).astype('float32')
    qt = torch.from_numpy(q.T.copy()).to(device).contiguous()
    pieces = []
    for i in range(0, len(values), chunk):
        x = torch.from_numpy(values[i:i + chunk].T.copy()).to(device).contiguous()
        k = torch.searchsorted(qt, x, right=True).clamp(1, 32)
        lo = torch.gather(qt, 1, k - 1)
        hi = torch.gather(qt, 1, k)
        frac = ((x - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)
        pieces.append(((k - 1 + frac) / 32 * 2 - 1).T.contiguous().cpu().numpy())
    return np.concatenate(pieces)
