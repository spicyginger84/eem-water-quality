"""Additive Gaussian EEM peaks with fold-local masks and optional soft priors.

Region names identify hypotheses, not confirmed chemical components. Intensities
are signed observations; shapes and inferred amplitudes are nonnegative.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from threadpoolctl import threadpool_limits

from .peak_decomposition import infer_scores, match_surfaces, valid_grid

# Deliberately allow small wavelength deviations from the reference peak table.
REGIONS = (
    ("B", 270., 280., 305., 320.),
    ("T", 275., 285., 320., 350.),
    ("M", 290., 315., 370., 420.),
    ("A", 230., 265., 380., 480.),
    ("C", 320., 360., 420., 480.),
    ("UV", 220., 240., 315., 365.),
)


def peak_initialization(train, ex, em, mask, regions=REGIONS, seed=42):
    """Positive, mask-aware region maxima; broad shoulders remain hypotheses."""
    rng = np.random.default_rng(seed)
    positive = np.maximum(train, 0)
    scale = np.maximum(np.linalg.norm(positive[:, mask], axis=1), 1e-12)
    maps = positive / scale[:, None, None]
    # Starts see different combinations of training samples, not just seed jitter.
    chosen = rng.choice(len(train), len(train), replace=True)
    mean = maps[chosen].mean(axis=0)
    sigma = (2 / np.median(np.diff(em)), 3 / np.median(np.diff(ex)))
    smooth = gaussian_filter(mean * mask, sigma) / np.maximum(
        gaussian_filter(mask.astype(float), sigma), 1e-12)
    smooth[~mask] = 0
    xx, yy = np.meshgrid(ex, em)
    threshold = max(float(smooth.max()) * 1e-6, 1e-15)
    centers, supported = [], []
    local = (smooth == maximum_filter(smooth, size=(9, 3))) & mask & (smooth > threshold)
    for _, lo, hi, bottom, top in regions:
        window = mask & (xx >= lo) & (xx <= hi) & (yy >= bottom) & (yy <= top)
        valid = window & (smooth > threshold)
        if valid.any():
            candidates = np.where(valid, smooth, -np.inf)
            iy, ix = np.unravel_index(candidates.argmax(), smooth.shape)
            center = np.array([ex[ix], em[iy]])
            supported.append(bool(local[iy, ix]))
        else:
            # Explicit unsupported prior seed, never reported as an observed peak.
            center = np.array([(lo + hi) / 2, (bottom + top) / 2])
            supported.append(False)
        center += rng.normal(size=2) * [2.5, 3]
        centers.append(np.clip(center, [ex.min(), em.min()], [ex.max(), em.max()]))
    return np.asarray(centers), supported


@dataclass
class ImprovedV0Fit:
    surfaces: np.ndarray
    full_surfaces: np.ndarray
    mask: np.ndarray
    centers: np.ndarray
    widths: np.ndarray
    rho: np.ndarray
    names: list
    runs: list
    best_start: int
    history: list
    seed_centers: np.ndarray
    seed_supported: list
    start_surfaces: np.ndarray
    start_centers: np.ndarray
    norm_floor: float

    def transform(self, eem):
        eem = np.asarray(eem, float)
        if eem.ndim != 3 or eem.shape[1:] != self.mask.shape or not np.isfinite(eem).all():
            raise ValueError("Expected finite EEMs with the fitted wavelength grid")
        return infer_scores(eem[:, self.mask], self.surfaces)

    def reconstruct(self, eem):
        return np.einsum("nk,kij->nij", self.transform(eem), self.full_surfaces)

    def seed_stability(self):
        result = []
        for start, shapes in enumerate(self.start_surfaces):
            rows, cols, similarity = match_surfaces(self.surfaces[:-1], shapes[:-1])
            shifts = self.start_centers[start][cols] - self.centers[rows]
            result.append({"start": start, "cosine_min": float(similarity.min()),
                           "cosine_mean": float(similarity.mean()),
                           "center_shift_nm_max": float(np.linalg.norm(shifts, axis=1).max())})
        return result


@threadpool_limits.wrap(limits=2)
def fit_improved_v0(
    train, ex, em, *, regions=REGIONS, mask=None, prior_strength=0.002,
    rho_max=0., rho_penalty=0.002, n_starts=2, max_steps=350,
    seed=42, learning_rate=0.025, center_learning_rate=0.008,
    warmup_steps=50, check_every=25, patience=5, tolerance=1e-5,
):
    """Balanced least squares + soft region/rho penalties; exact NNLS checkpoints.

    Surface scale is fixed on the TRAIN observed domain. Background is a constant
    additive surface, not a probability that a pixel belongs to a noise class.
    Seeds without a local peak are explicitly recorded; NNLS may switch them off.
    """
    import torch
    from torch import nn

    train, ex, em = np.asarray(train, float), np.asarray(ex, float), np.asarray(em, float)
    if train.ndim != 3 or train.shape[1:] != (len(em), len(ex)):
        raise ValueError("EEM dimensions must be (sample, emission, excitation)")
    if len(ex) < 2 or len(em) < 2 or np.any(np.diff(ex) <= 0) or np.any(np.diff(em) <= 0):
        raise ValueError("Wavelength axes must increase strictly")
    if not 0 <= rho_max < 1 or min(n_starts, max_steps, check_every, patience) < 1:
        raise ValueError("Invalid training configuration")
    if prior_strength < 0 or rho_penalty < 0 or len(regions) < 1:
        raise ValueError("Invalid regularization/regions")
    mask = valid_grid(train, mask)
    if not np.any(train[:, mask] > 0):
        raise ValueError("No positive signal in training EEMs")
    norms = np.linalg.norm(train[:, mask], axis=1)
    floor = max(float(np.quantile(norms, 0.05)), 1e-12)
    scales = np.maximum(norms, floor)
    target_np = train[:, mask] / scales[:, None]
    dtype = torch.float32
    target = torch.tensor(target_np, dtype=dtype)
    xx, yy = np.meshgrid(ex / 100, em / 100)
    xy = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=dtype)
    keep = torch.tensor(mask.ravel())
    rank = len(regions)
    bounds = torch.tensor([r[1:] for r in regions], dtype=dtype)
    low = bounds[:, [0, 2]] / 100
    high = bounds[:, [1, 3]] / 100
    region_scale = torch.tensor([0.15, 0.25], dtype=dtype)
    min_width = torch.tensor([5., 5.], dtype=dtype) / 100
    max_width = torch.tensor([80., 100.], dtype=dtype) / 100

    class ShapeModel(nn.Module):
        def __init__(self, centers, run_seed):
            super().__init__()
            rng = np.random.default_rng(run_seed)
            self.center = nn.Parameter(torch.tensor(centers/100, dtype=dtype))
            widths = np.array([[12, 14] if r[0] in {"B", "T", "UV"} else [25, 35]
                               for r in regions], float)
            widths *= rng.uniform(0.8, 1.2, widths.shape)
            fraction = (widths / 100 - min_width.numpy()) / (max_width-min_width).numpy()
            self.width_raw = nn.Parameter(torch.tensor(np.log(fraction/(1-fraction)), dtype=dtype))
            self.rho_raw = nn.Parameter(torch.zeros(rank, dtype=dtype), requires_grad=rho_max > 0)

        def widths(self):
            return min_width + (max_width-min_width)*torch.sigmoid(self.width_raw)

        def forward(self):
            delta = (xy[None] - self.center[:, None]) / self.widths()[:, None]
            dx, dy = delta[:, :, 0], delta[:, :, 1]
            rho = rho_max * torch.tanh(self.rho_raw)[:, None]
            logz = -(dx**2 + dy**2 - 2*rho*dx*dy) / (2*(1-rho**2))
            logz -= logz[:, keep].max(dim=1).values[:, None]
            shapes = torch.exp(logz.clamp(max=50))
            shapes = shapes / torch.linalg.vector_norm(shapes[:, keep], dim=1)[:, None]
            return torch.cat([shapes, torch.ones_like(shapes[:1]) / np.sqrt(mask.sum())])

        def penalty(self):
            distance = (torch.relu(low-self.center) + torch.relu(self.center-high)) / region_scale
            rho = rho_max * torch.tanh(self.rho_raw)
            return prior_strength * distance.square().mean() + rho_penalty * rho.square().mean()

    runs, solutions = [], []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for start in range(n_starts):
            centers, supported = peak_initialization(train, ex, em, mask, regions, seed+start)
            model = ShapeModel(centers, seed+start)
            initial = model().detach().numpy()[:, mask.ravel()]
            scores = nn.Parameter(torch.tensor(infer_scores(target_np, initial), dtype=dtype))
            optimizer = torch.optim.Adam([
                {"params": [model.center], "lr": center_learning_rate},
                {"params": [model.width_raw, model.rho_raw, scores], "lr": learning_rate},
            ])
            best, history, stale, previous = None, [], 0, None
            converged = False
            for step in range(1, max_steps+1):
                optimizer.zero_grad()
                full = model()
                error = ((scores @ full[:, keep] - target)**2).sum(1).mean()
                loss = error + model.penalty()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite objective")
                loss.backward()
                if step <= warmup_steps:
                    model.center.grad.zero_()
                    if model.rho_raw.grad is not None:
                        model.rho_raw.grad.zero_()
                optimizer.step()
                with torch.no_grad():
                    scores.clamp_(min=0)
                    model.center[:, 0].clamp_(float(ex.min()/100), float(ex.max()/100))
                    model.center[:, 1].clamp_(float(em.min()/100), float(em.max()/100))
                if step % check_every != 0 and step != max_steps:
                    continue
                # Evaluate the SAME post-update state that is saved; re-solve amplitudes.
                with torch.no_grad():
                    full_np = model().numpy().astype(np.float64)
                    exact = infer_scores(target_np, full_np[:, mask.ravel()])
                    data_loss = float(np.mean(np.sum((exact @ full_np[:, mask.ravel()]-target_np)**2, 1)))
                    penalty = float(model.penalty())
                    value = data_loss + penalty
                    scores.copy_(torch.tensor(exact, dtype=dtype))
                optimizer.state.pop(scores, None)
                history.append({"step": step, "objective": value, "data_loss": data_loss,
                                "penalty": penalty})
                if best is None or value < best[0]:
                    best = (value, full_np.copy(), model.center.detach().numpy().copy()*100,
                            model.widths().detach().numpy().copy()*100,
                            rho_max*np.tanh(model.rho_raw.detach().numpy().copy()), data_loss)
                delta = np.inf if previous is None else abs(previous-value)
                cutoff = max(1e-9, tolerance*abs(previous or value))
                stale = stale+1 if step > warmup_steps and delta < cutoff else 0
                previous = value
                if stale >= patience:
                    converged = True
                    break
            runs.append({"start": start, "objective": best[0], "data_loss": best[5],
                         "steps": step, "converged": converged, "seed": seed+start})
            full_np = best[1]
            full_np /= full_np[:, mask.ravel()].sum(1, keepdims=True)
            solutions.append((full_np, best[2], best[3], best[4], history, centers, supported))
    finally:
        torch.set_num_threads(previous_threads)
    winner = min(range(n_starts), key=lambda i: runs[i]["objective"])
    full, centers, widths, rho, history, seeds, supported = solutions[winner]
    return ImprovedV0Fit(full[:, mask.ravel()], full.reshape(-1, len(em), len(ex)), mask,
                         centers, widths, rho, [r[0] for r in regions] + ["background"],
                         runs, winner, history, seeds, supported,
                         np.stack([s[0][:, mask.ravel()] for s in solutions]),
                         np.stack([s[1] for s in solutions]), floor)
