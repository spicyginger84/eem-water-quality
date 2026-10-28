"""Full-EEM 2-D CNN baseline (Sci. Rep. 2021-style: no peak/PARAFAC extraction).

log1p EEM image -> 3 conv blocks (16/32/64, BN, ReLU, max-pool) -> global average pool ->
[field parameters] -> 64 -> one output per target (standardised log labels). Adam
(lr 1e-3, weight decay 1e-4), batch 32, up to `epochs`; the epoch is chosen on a 15%
inner holdout (stations if groups are given, else rows) and that checkpoint is kept.
Multi-output: one network predicts every target.
"""

from __future__ import annotations

import numpy as np

SEED = 11


class CNNRegressor:
    multi_output = True

    def __init__(self, epochs=200, batch=32, threads=8):
        self.epochs, self.batch, self.threads = epochs, batch, threads

    def _net(self, n_field, n_out):
        from torch import nn

        def block(a, b, pool=True):
            return [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU()] + ([nn.MaxPool2d(2)] if pool else [])

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(*block(1, 16), *block(16, 32), *block(32, 64, False), nn.AdaptiveAvgPool2d(1))
                self.head = nn.Sequential(nn.Linear(64 + n_field, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_out))

            def forward(self, x, f):
                import torch

                return self.head(torch.cat([self.conv(x).flatten(1), f], 1))

        return Net()

    def fit(self, images, field, LY, groups=None):
        import torch
        from torch import nn

        torch.set_num_threads(self.threads)
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        n = len(images)
        if groups is not None:
            st = np.unique(groups)
            held = rng.choice(st, max(1, round(0.15 * len(st))), replace=False)
            val = np.isin(groups, held)
        else:
            val = rng.random(n) < 0.15
        fit = np.flatnonzero(~val)
        img = np.log1p(images)
        self.mu, self.sd = img[fit].mean(), img[fit].std() + 1e-9
        self.fm, self.fs = (field[fit].mean(0), field[fit].std(0) + 1e-9) if field is not None else (None, None)
        self.ym, self.ys = LY[fit].mean(0), LY[fit].std(0) + 1e-9
        self.lo, self.hi = LY.min(0) - np.log(2), LY.max(0) + np.log(2)
        X, F = self._tensors(images, field)
        Y = torch.tensor((LY - self.ym) / self.ys, dtype=torch.float32)
        self.net = self._net(F.shape[1], LY.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        fit_t, val_t = torch.tensor(fit), torch.tensor(np.flatnonzero(val))
        best, state = np.inf, None
        for _ in range(self.epochs):
            self.net.train()
            for b in fit_t[torch.randperm(len(fit_t))].split(self.batch):
                if len(b) < 2:
                    continue
                opt.zero_grad()
                nn.functional.mse_loss(self.net(X[b], F[b]), Y[b]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                loss = nn.functional.mse_loss(self.net(X[val_t], F[val_t]), Y[val_t]).item()
            if loss < best:
                best, state = loss, {k: v.clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(state)
        return self

    def _tensors(self, images, field):
        import torch

        X = torch.tensor((np.log1p(images) - self.mu) / self.sd, dtype=torch.float32)[:, None]
        if field is None:
            F = torch.zeros((len(images), 0))
        else:
            F = torch.tensor((field - self.fm) / self.fs, dtype=torch.float32)
        return X, F

    def predict(self, images, field):
        import torch

        X, F = self._tensors(images, field)
        self.net.eval()
        with torch.no_grad():
            out = self.net(X, F).numpy() * self.ys + self.ym
        return np.clip(out, self.lo, self.hi)
