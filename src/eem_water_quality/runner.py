"""Benchmark runner: every model x outer scheme x fold x target, zero-shot and few-shot.

Jobs run on a thread pool (BLAS limited to one thread per job, so the machine is never
oversubscribed); fixed feature blocks are built once and PARAFAC once per outer fold.
The CNN runs alone afterwards with all threads. Outputs in the run directory:
  predictions.parquet   long table: scheme, model, regime, target, sample_id, ..., y_pred
  metrics.csv           per scheme / model / regime / scored subset / target
  summary.csv           station schemes aggregated: mean and min over schemes
  paired.csv            station-MAE vs the reference model, station bootstrap
  report.md, config.json, timings.csv
Zero-shot is always scored on every sample ("all"); few-shot designs are scored on their
own query rows, next to the zero-shot prediction on the same rows.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TARGETS, load_dataset
from .features import FeatureBuilder
from .metrics import paired_station_bootstrap, regression_metrics
from .models import resolve
from .protocols import calibrate, is_station_scheme, outer_splits


@dataclass
class Config:
    data: str = "data/processed"
    em_step: int = 2
    targets: tuple = TARGETS
    schemes: list = field(default_factory=lambda: ["random", "station"])
    models: list = field(default_factory=lambda: ["parafac_xgb+field", "svr"])
    reference: str = "parafac_xgb+field"
    fewshot_designs: list = field(default_factory=lambda: ["early", "rolling", "random"])
    fewshot_k: int = 3
    fewshot_prior: float = 3.0
    workers: int = 8
    parafac_rank: int = 4
    parafac_iters: int = 200
    cnn_epochs: int = 200
    bootstrap: int = 2000

    @classmethod
    def load(cls, path, **overrides):
        values = json.loads(Path(path).read_text()) if path else {}
        values |= {k: v for k, v in overrides.items() if v is not None}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values)


def _pool(fn, jobs, workers):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1), ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda a: fn(*a), jobs))


class Benchmark:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.D = load_dataset(cfg.data, cfg.em_step, tuple(cfg.targets))
        self.fb = FeatureBuilder(self.D, cfg.parafac_rank, cfg.parafac_iters)
        self.LY = np.log(self.D.y)
        self.timings = []

    def _inputs(self, model, key, train):
        """Feature matrix (or cube / image), column groups and field flag for a model."""
        inputs, factory, plus_field = resolve(model)
        D = self.D
        if isinstance(inputs, list):
            Z, _, groups = self.fb.table(inputs, key, train)
            return Z, groups, factory
        if inputs == "unfolded":
            Z = self.fb.unfolded()
            if plus_field:
                F = (D.field - D.field[train].mean(0)) / (D.field[train].std(0) + 1e-9)
                Z = np.column_stack([Z, F * Z[train].std(0).mean()])
            return Z, None, factory
        if inputs == "cube":
            return D.x.astype(np.float64), None, factory
        return D.x, None, None  # image (CNN)

    def run_scheme(self, scheme):
        D, cfg = self.D, self.cfg
        splits = outer_splits(D, scheme)
        station_groups = is_station_scheme(scheme)
        if any("parafac" in m for m in cfg.models):
            t0 = time.time()
            _pool(lambda i, tr: self.fb.fit_block("parafac", (scheme, i), tr),
                  [(i, tr) for i, (tr, _) in enumerate(splits)], cfg.workers)
            self.timings.append({"scheme": scheme, "step": "parafac", "seconds": time.time() - t0})
        preds = {m: np.full(D.y.shape, np.nan) for m in cfg.models}
        tabular = [m for m in cfg.models if not m.startswith("cnn")]

        def job(model, i, t):
            tr, te = splits[i]
            Z, groups, factory = self._inputs(model, (scheme, i), tr)
            est = factory(groups).fit(Z[tr], self.LY[tr, t], D.station[tr] if station_groups else None)
            return model, te, t, est.predict(Z[te])

        t0 = time.time()
        jobs = [(m, i, t) for m in tabular for i in range(len(splits)) for t in range(len(cfg.targets))]
        for model, te, t, p in _pool(job, jobs, cfg.workers):
            preds[model][te, t] = p
        self.timings.append({"scheme": scheme, "step": "tabular models", "seconds": time.time() - t0})
        for model in [m for m in cfg.models if m.startswith("cnn")]:
            from .cnn import CNNRegressor

            t0 = time.time()
            plus_field = resolve(model)[2]
            for tr, te in splits:
                F = D.field if plus_field else None
                net = CNNRegressor(cfg.cnn_epochs, threads=cfg.workers).fit(
                    D.x[tr], None if F is None else F[tr], self.LY[tr], D.station[tr] if station_groups else None)
                preds[model][te] = net.predict(D.x[te], None if F is None else F[te])
            self.timings.append({"scheme": scheme, "step": model, "seconds": time.time() - t0})
        fold = np.empty(len(D), object)
        for i, (_, te) in enumerate(splits):
            fold[te] = i
        return preds, fold

    def run(self, out):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2))
        D, cfg = self.D, self.cfg
        long, rows, paired = [], [], []
        for scheme in cfg.schemes:
            log_preds, fold = self.run_scheme(scheme)
            regimes = {"zero-shot": (log_preds, np.ones(len(D), bool))}
            if is_station_scheme(scheme):
                for design in cfg.fewshot_designs:
                    cal = {m: calibrate(D, p, design, cfg.fewshot_k, cfg.fewshot_prior) for m, p in log_preds.items()}
                    query = next(iter(cal.values()))[1]
                    regimes[f"few-shot {design}"] = ({m: c[0] for m, c in cal.items()}, query)
            for regime, (P, rows_mask) in regimes.items():
                subsets = {"all": rows_mask} if regime == "zero-shot" else {f"query {regime[9:]}": rows_mask}
                for model, lp in P.items():
                    y_pred = np.exp(lp)
                    long.append(pd.DataFrame({"scheme": scheme, "model": model, "regime": regime, "fold": np.repeat(fold, len(cfg.targets)),
                                                  "target": np.tile(cfg.targets, len(D)), "sample_id": np.repeat(D.sample_id, len(cfg.targets)),
                                                  "station": np.repeat(D.station, len(cfg.targets)), "scored": np.repeat(rows_mask, len(cfg.targets)),
                                                  "y_true": D.y.ravel(), "y_pred": y_pred.ravel()}))
                    for subset, m in subsets.items():
                        for t, name in enumerate(cfg.targets):
                            r = regression_metrics(D.y[m, t], y_pred[m, t], D.station[m], D.brackish[m])
                            rows.append({"scheme": scheme, "model": model, "regime": regime, "subset": subset, "target": name} | r)
                            if regime != "zero-shot":
                                z = np.exp(log_preds[model])
                                r0 = regression_metrics(D.y[m, t], z[m, t], D.station[m], D.brackish[m])
                                rows.append({"scheme": scheme, "model": model, "regime": "zero-shot", "subset": subset, "target": name} | r0)
                if is_station_scheme(scheme) and cfg.reference in P:
                    ref = np.exp(P[cfg.reference])
                    for model, lp in P.items():
                        if model == cfg.reference:
                            continue
                        for t, name in enumerate(cfg.targets):
                            m = rows_mask
                            d, lo, hi = paired_station_bootstrap(D.y[m, t], ref[m, t], np.exp(lp)[m, t], D.station[m], cfg.bootstrap)
                            paired.append({"scheme": scheme, "regime": regime, "model": model, "reference": cfg.reference, "target": name,
                                               "delta_station_mae": d, "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0 or hi < 0)})
            print(f"[{scheme}] done", flush=True)
        pd.concat(long, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
        metrics = pd.DataFrame(rows).drop_duplicates(["scheme", "model", "regime", "subset", "target"])
        metrics.to_csv(out / "metrics.csv", index=False)
        station = metrics[metrics.scheme.map(lambda s: s.startswith("station"))]
        summary = (station.groupby(["model", "regime", "subset", "target"], sort=False)
                   .agg(r2_mean=("r2", "mean"), r2_min=("r2", "min"), station_mae_mean=("station_mae", "mean"),
                        schemes=("scheme", "nunique")).reset_index())
        summary.to_csv(out / "summary.csv", index=False)
        pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
        pd.DataFrame(self.timings).to_csv(out / "timings.csv", index=False)
        (out / "report.md").write_text(report(metrics, summary, pd.DataFrame(paired), cfg))
        return metrics, summary


def _table(frame, index, value, targets):
    t = frame.pivot_table(index=index, columns="target", values=value, sort=False)[list(targets)]
    head = "| " + " | ".join([*index, *targets]) + " |\n|" + "---|" * (len(index) + len(targets)) + "\n"
    body = "".join("| " + " | ".join([*(k if isinstance(k, tuple) else (k,)), *(f"{v:.3f}" for v in r)]) + " |\n"
                   for k, r in zip(t.index, t.to_numpy()))
    return head + body


def report(metrics, summary, paired, cfg):
    tg = cfg.targets
    md = ["# EEM benchmark report\n",
          ("Zero-shot is scored on every sample; few-shot rows are scored on their own query set "
          f"(K={cfg.fewshot_k}, shrinkage prior {cfg.fewshot_prior}). R² and station-MAE in mg/L.\n")]
    zero = metrics[(metrics.regime == "zero-shot") & (metrics.subset == "all")]
    for scheme in cfg.schemes:
        md += [f"\n## Zero-shot, scheme `{scheme}` (all samples)\n\n### R²\n", _table(zero[zero.scheme == scheme], ["model"], "r2", tg),
               "\n### Station-MAE\n", _table(zero[zero.scheme == scheme], ["model"], "station_mae", tg)]
    if len(summary):
        md += ["\n## Station schemes, mean over schemes\n"]
        for (regime, subset), g in summary.groupby(["regime", "subset"], sort=False):
            md += [f"\n### {regime}, scored on {subset} — mean R² (min in summary.csv)\n", _table(g, ["model"], "r2_mean", tg)]
    if len(paired):
        p = paired[paired.scheme == "station"].copy()
        p["cell"] = p.apply(lambda r: f"{r.delta_station_mae:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]{' *' if r.significant else ''}", axis=1)
        md += [f"\n## Station-MAE vs `{cfg.reference}`, scheme `station` (< 0 = better, * = 95% CI excludes 0)\n"]
        for regime, g in p.groupby("regime", sort=False):
            t = g.pivot_table(index="model", columns="target", values="cell", aggfunc="first", sort=False)[list(tg)]
            md += [f"\n### {regime}\n", "| model | " + " | ".join(tg) + " |\n|---|" + "---|" * len(tg) + "\n",
                   "".join(f"| {m} | " + " | ".join(r) + " |\n" for m, r in zip(t.index, t.to_numpy()))]
    return "".join(md)
