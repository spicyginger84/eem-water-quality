"""Experimental shared EEM surfaces; used by the V1 vs PARAFAC notebook.

PyTorch is an optional notebook dependency. Both families minimize the same
masked, sample-weighted squared error. This is a nonnegative CP/PARAFAC model
with an Adam solver, not the TensorLy HALS solver in the earlier notebook.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment, nnls
from threadpoolctl import threadpool_limits


def infer_scores(values, surfaces):
    """Exact small-rank NNLS per sample, with fixed surfaces (rank, pixels)."""
    values = np.asarray(values, dtype=np.float64)
    surfaces = np.asarray(surfaces, dtype=np.float64)
    # QR reduces thousands of spectral rows to rank rows, without normal equations.
    q, r = np.linalg.qr(surfaces.T, mode="reduced")
    with threadpool_limits(limits=2):
        projected = values @ q
        return np.vstack([nnls(r, row, maxiter=1000)[0] for row in projected])


def valid_grid(train, supplied_mask=None):
    """Infer active cells from TRAIN only, or use a supplied physical valid mask."""
    train = np.asarray(train)
    if train.ndim != 3 or not np.isfinite(train).all():
        raise ValueError("Expected finite (samples, emission, excitation) EEMs")
    if supplied_mask is None:
        mask = np.any(train != 0, axis=0)
    else:
        mask = np.asarray(supplied_mask, dtype=bool).copy()
        if mask.shape != train.shape[1:]:
            raise ValueError("Mask shape must equal (emission, excitation)")
    if mask.sum() < 10:
        raise ValueError("Too few valid wavelength pairs")
    return mask


def match_surfaces(left, right):
    """Optimal matching by cosine on a COMMON observed wavelength grid."""
    a = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-15)
    b = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-15)
    similarity = np.clip(a @ b.T, -1, 1)
    rows, cols = linear_sum_assignment(-similarity)
    return rows, cols, similarity[rows, cols]


@dataclass
class PeakFit:
    family: str
    rank: int
    mask: np.ndarray
    # All rows sum to one ON THE TRAIN VALID GRID; final row can be background.
    surfaces: np.ndarray
    full_surfaces: np.ndarray
    background: bool
    runs: list
    best_start: int
    history: list
    parameters: dict
    norm_floor: float

    def transform(self, eem):
        """Infer only sample amplitudes; never change learned shapes or mask."""
        eem = np.asarray(eem)
        if eem.shape[1:] != self.mask.shape or not np.isfinite(eem).all():
            raise ValueError("New EEM grid/values do not match the fitted model")
        return infer_scores(eem[:, self.mask], self.surfaces)

    def reconstruct_valid(self, eem):
        return self.transform(eem) @ self.surfaces


def _initial_peaks(train, ex, em, mask, rank, seed):
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(train[:, mask], axis=1)
    mean = np.mean(np.maximum(train, 0) / np.maximum(norms[:, None, None], 1e-12), axis=0)
    mean[~mask] = 0
    candidates = np.argwhere((mean == maximum_filter(mean, size=(17, 5))) & mask)
    candidates = candidates[np.argsort(mean[tuple(candidates.T)])[::-1]]
    chosen = []
    for iy, ix in candidates:
        center = np.array([ex[ix], em[iy]])
        if all(np.linalg.norm((center - old) / [25, 30]) > 1 for old in chosen):
            chosen.append(center)
        if len(chosen) == rank:
            break
    grid = np.argwhere(mask)
    while len(chosen) < rank:
        iy, ix = grid[rng.integers(len(grid))]
        chosen.append(np.array([ex[ix], em[iy]]))
    centers = np.asarray(chosen) + rng.normal(size=(rank, 2)) * [5, 7]
    return np.clip(centers, [ex.min(), em.min()], [ex.max(), em.max()])


@threadpool_limits.wrap(limits=2)
def fit_peak_model(
    train, ex, em, *, family="parafac", rank=4, mask=None,
    weighting="balanced", background=False, n_starts=2, max_steps=500,
    learning_rate=0.035, seed=42, tolerance=1e-5, patience=5,
):
    """Fit shared nonnegative CP, diagonal Gaussian or correlated Gaussian peaks.

    Signed corrected observations are retained. Balanced loss divides each EEM
    by its train-derived norm (floor: 5th percentile); only one common scalar
    is used for raw loss. Normalize each shape to unit area after fitting.
    No target labels are accepted. Initialization and all scales use train only.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    if family not in {"parafac", "gaussian_diag", "gaussian_full"}:
        raise ValueError(f"Unknown family: {family}")
    if weighting not in {"balanced", "raw"}:
        raise ValueError("weighting must be balanced or raw")
    if rank < 1 or n_starts < 1 or max_steps < 1:
        raise ValueError("rank, n_starts and max_steps must be positive")
    train = np.asarray(train, dtype=np.float64)
    ex, em = np.asarray(ex), np.asarray(em)
    mask = valid_grid(train, mask)
    if train.shape[1:] != (len(em), len(ex)):
        raise ValueError("Axis lengths do not match EEM")
    observed = train[:, mask]
    norms = np.linalg.norm(observed, axis=1)
    floor = max(float(np.quantile(norms, 0.05)), 1e-12)
    scales = np.maximum(norms, floor) if weighting == "balanced" else np.full(
        len(train), max(float(np.median(norms)), 1e-12)
    )
    target = torch.tensor(observed / scales[:, None], dtype=torch.float32)
    ex_t = torch.tensor(ex / 100, dtype=torch.float32)
    em_t = torch.tensor(em / 100, dtype=torch.float32)
    flat_mask = torch.tensor(mask.ravel())
    xx, yy = torch.meshgrid(ex_t, em_t, indexing="xy")

    def inverse_softplus(x):
        return np.log(np.expm1(np.maximum(x, 1e-7)))

    class Surfaces(nn.Module):
        def __init__(self, centers, run_seed):
            super().__init__()
            rng = np.random.default_rng(run_seed)
            self.center = nn.Parameter(torch.tensor(centers / 100, dtype=torch.float32),
                                       requires_grad=family != "parafac")
            # Bounds in nm: sigma Ex 5..100, Em 3..100. No repulsion by default.
            self.width_logit = nn.Parameter(torch.tensor(
                np.tile([-1.5, -1.1], (rank, 1)) + rng.normal(0, 0.2, (rank, 2)),
                dtype=torch.float32), requires_grad=family != "parafac")
            self.rho_raw = nn.Parameter(torch.zeros(rank),
                                        requires_grad=family == "gaussian_full")
            if family == "parafac":
                b = np.exp(-0.5 * ((em[:, None] - centers[:, 1]) / 30) ** 2) + 0.02
                c = np.exp(-0.5 * ((ex[:, None] - centers[:, 0]) / 22) ** 2) + 0.02
                b *= rng.uniform(0.8, 1.2, b.shape)
                c *= rng.uniform(0.8, 1.2, c.shape)
                self.b = nn.Parameter(torch.tensor(inverse_softplus(b), dtype=torch.float32))
                self.c = nn.Parameter(torch.tensor(inverse_softplus(c), dtype=torch.float32))

        def forward(self):
            if family == "parafac":
                full = torch.einsum("ik,jk->kij", F.softplus(self.b), F.softplus(self.c))
            else:
                widths = torch.tensor([0.05, 0.03]) + torch.tensor([0.95, 0.97]) * torch.sigmoid(
                    self.width_logit)
                dx = (xx[None] - self.center[:, 0, None, None]) / widths[:, 0, None, None]
                dy = (yy[None] - self.center[:, 1, None, None]) / widths[:, 1, None, None]
                rho = (0.9 * torch.tanh(self.rho_raw))[:, None, None]
                log_shape = -(dx**2 + dy**2 - 2*rho*dx*dy) / (2*(1-rho**2))
                # Stable on truncated/masked domain, not a continuous-density EM update.
                log_shape = log_shape - log_shape.reshape(rank, -1)[:, flat_mask].max(
                    dim=1).values[:, None, None]
                full = torch.exp(log_shape.clamp(max=50))
            full = full.reshape(rank, -1)
            full = full / torch.linalg.vector_norm(full[:, flat_mask], dim=1).clamp_min(
                1e-12)[:, None]
            if background:
                full = torch.cat([full, torch.ones_like(full[:1]) / np.sqrt(mask.sum())])
            return full

    best = None
    runs = []
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for start in range(n_starts):
            centers = _initial_peaks(train, ex, em, mask, rank, seed + start)
            module = Surfaces(centers, seed + start)
            initial = module().detach().numpy()[:, mask.ravel()]
            initial_scores = infer_scores(target.numpy(), initial)
            scores = nn.Parameter(torch.tensor(initial_scores, dtype=torch.float32))
            optimizer = torch.optim.Adam([*module.parameters(), scores], lr=learning_rate)
            history, flat_checks, previous = [], 0, None
            best_loss, best_full, best_state = np.inf, None, None
            converged = False
            for step in range(1, max_steps + 1):
                optimizer.zero_grad()
                full = module()
                prediction = scores @ full[:, flat_mask]
                loss = ((prediction - target)**2).sum(dim=1).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite objective: {family}, start {start}")
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    scores.clamp_(min=0)
                    module.center[:, 0].clamp_(float(ex.min()/100), float(ex.max()/100))
                    module.center[:, 1].clamp_(float(em.min()/100), float(em.max()/100))
                if step % 25 == 0 or step == max_steps:
                    # Coordinate minimization makes train/test scores use the same NNLS rule.
                    current_full = module().detach().numpy().astype(float)
                    current = current_full[:, mask.ravel()]
                    exact_scores = infer_scores(target.numpy(), current)
                    value = float(np.mean(np.sum((target.numpy() - exact_scores @ current)**2,
                                                  axis=1)))
                    with torch.no_grad():
                        scores.copy_(torch.tensor(exact_scores, dtype=torch.float32))
                    optimizer.state.pop(scores, None)
                    history.append({"step": step, "loss": value})
                    if value < best_loss:
                        best_loss, best_full = value, current_full.copy()
                        best_state = {k: v.detach().numpy().copy()
                                      for k, v in module.state_dict().items()}
                    relative_change = np.inf if previous is None else abs(previous-value) / max(
                        abs(previous), 1e-12)
                    flat_checks = flat_checks + 1 if relative_change < tolerance else 0
                    previous = value
                    if flat_checks >= patience:
                        converged = True
                        break
            runs.append({"start": start, "loss": best_loss, "steps": step,
                         "converged": converged})
            if best is None or best_loss < best[0]:
                best = best_loss, best_full, start, history, best_state
    finally:
        torch.set_num_threads(old_threads)
    _, full, best_start, history, params = best
    totals = full[:, mask.ravel()].sum(axis=1)
    full /= totals[:, None]
    # Stable labels for plots ONLY; cross-fit matching must still use surface similarity.
    peaks = np.argmax(full[:rank], axis=1)
    order = np.argsort(peaks)
    if background:
        order = np.r_[order, rank]
    full = full[order]
    if family.startswith("gaussian"):
        params = {"centers_nm": params["center"][order[:rank]] * 100,
                  "widths_nm": (np.array([5, 3]) + np.array([95, 97]) /
                                (1 + np.exp(-params["width_logit"][order[:rank]]))),
                  "rho": 0.9 * np.tanh(params["rho_raw"][order[:rank]])}
    else:
        params = {"emission_loadings": np.logaddexp(0, params["b"])[:, order[:rank]],
                  "excitation_loadings": np.logaddexp(0, params["c"])[:, order[:rank]]}
    params["surface_order"] = order
    return PeakFit(family, rank, mask, full[:, mask.ravel()], full.reshape(-1, len(em), len(ex)),
                   background, runs, best_start, history, params, floor)


def reconstruction_metrics(eem, fitted):
    """Held-out observed-cell error; amplitudes inferred from the held-out EEM."""
    y = np.asarray(eem)[:, fitted.mask]
    pred = fitted.reconstruct_valid(eem)
    sse = np.sum((y - pred)**2, axis=1)
    energy = np.sum(y*y, axis=1)
    return {"relative_rmse_median": float(np.median(np.sqrt(sse / np.maximum(energy, 1e-24)))),
            "relative_mse_mean": float(np.mean(sse / np.maximum(energy, 1e-24))),
            "fit_energy": float(1 - sse.sum() / max(energy.sum(), 1e-24))}
