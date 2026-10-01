"""Reproducible ten-station experiment used by the improved V0 notebook.

All improved-model temporal evaluation fits shapes on training EEMs only.
The historical V0 artifacts are descriptive in-sample references ONLY.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eem_water_quality.features.improved_v0 import REGIONS, fit_improved_v0
from eem_water_quality.features.peak_decomposition import match_surfaces

DEFAULT_CONFIG = {
    "seed": 42, "full_steps": 400, "cv_steps": 300,
    "full_starts": 3, "cv_starts": 2, "ridge_alpha": 10.,
    "variants": {
        "diag_no_prior": {"prior_strength": 0., "rho_max": 0., "rho_penalty": 0.},
        "diag_soft_prior": {"prior_strength": 0.002, "rho_max": 0., "rho_penalty": 0.},
        "rho_soft_prior": {"prior_strength": 0.002, "rho_max": 0.3, "rho_penalty": 0.002},
    },
}


def slug(value):
    return re.sub(r"[^\w.-]+", "_", str(value)).strip("_")


def temporal_splits(frame):
    """First six months train; test 7–8, 9–10, 11–12 for a twelve-month series."""
    dates = frame.Year.to_numpy(int)*12 + frame.Month.to_numpy(int)
    unique = np.unique(dates)
    if len(unique) < 8:
        raise ValueError("Need at least eight distinct months for three temporal blocks")
    for fold, months in enumerate(np.array_split(unique[len(unique)//2:], 3)):
        train = np.where(dates < months.min())[0]
        test = np.where(np.isin(dates, months))[0]
        assert dates[train].max() < dates[test].min()
        yield fold, train, test


def cached_fit(train, ex, em, cache_dir, config):
    import scipy
    import torch

    h = hashlib.sha256()
    for array in [train, ex, em]:
        h.update(str((array.shape, array.dtype)).encode())
        h.update(np.ascontiguousarray(array).tobytes())
    dependencies = [ROOT / "src/eem_water_quality/features/improved_v0.py",
                    ROOT / "src/eem_water_quality/features/peak_decomposition.py"]
    for path in dependencies:
        h.update(path.read_bytes())
    versions = {"numpy": np.__version__, "torch": torch.__version__, "scipy": scipy.__version__}
    h.update(json.dumps({"config": config, "versions": versions}, sort_keys=True).encode())
    key = h.hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.joblib"
    if path.exists():
        return joblib.load(path), key, True
    fitted = fit_improved_v0(train, ex, em, **config)
    joblib.dump(fitted, path, compress=3)
    return fitted, key, False


def reconstruction_rows(values, prediction, mask, ex, em, indices, station, variant, split):
    error = values[:, mask]-prediction[:, mask]
    energy = np.sum(values[:, mask]**2, axis=1)
    rel = np.sqrt(np.sum(error**2, axis=1) / np.maximum(energy, 1e-24))
    rows = []
    for i, sample_index in enumerate(indices):
        obs = np.where(mask, values[i], -np.inf)
        pred = np.where(mask, prediction[i], -np.inf)
        oy, ox = np.unravel_index(obs.argmax(), mask.shape)
        py, px = np.unravel_index(pred.argmax(), mask.shape)
        unseen = ~mask & (values[i] != 0)
        rows.append({"station": station, "variant": variant, "split": split,
            "sample_index": int(sample_index), "relative_rmse": rel[i],
            "rmse": np.sqrt(np.mean(error[i]**2)),
            "observed_peak_Ex": ex[ox], "observed_peak_Em": em[oy],
            "reconstructed_peak_Ex": ex[px], "reconstructed_peak_Em": em[py],
            "peak_distance_nm": np.hypot(ex[ox]-ex[px], em[oy]-em[py]),
            "unseen_nonzero_cells": int(unseen.sum()),
            "unseen_energy_fraction": float(np.sum(values[i][unseen]**2)
                                             / max(np.sum(values[i]**2), 1e-24))})
    return rows


def regional_rows(values, prediction, mask, ex, em, station, variant, split):
    xx, yy = np.meshgrid(ex, em)
    rows = []
    for name, lo, hi, bottom, top in REGIONS:
        region = mask & (xx >= lo) & (xx <= hi) & (yy >= bottom) & (yy <= top)
        if not region.any():
            continue
        actual, estimated = values[:, region], prediction[:, region]
        distances = np.hypot(xx[region][actual.argmax(1)]-xx[region][estimated.argmax(1)],
                             yy[region][actual.argmax(1)]-yy[region][estimated.argmax(1)])
        relative = np.linalg.norm(actual-estimated, axis=1) / np.maximum(
            np.linalg.norm(actual, axis=1), 1e-12)
        rows.append({"station": station, "variant": variant, "split": split, "region": name,
                     "median_relative_rmse": np.median(relative),
                     "median_window_max_distance_nm": np.median(distances)})
    return rows


def region_features(values, mask, ex, em):
    xx, yy = np.meshgrid(ex, em)
    columns = []
    for _, lo, hi, bottom, top in REGIONS:
        region = mask & (xx >= lo) & (xx <= hi) & (yy >= bottom) & (yy <= top)
        columns.append(np.maximum(values[:, region].max(1), 0) if region.any() else np.zeros(len(values)))
    return np.stack(columns, axis=1)


def parameter_rows(fit, data, station, variant):
    scores = fit.transform(data)
    shares = scores / np.maximum(scores.sum(1, keepdims=True), 1e-12)
    rows = []
    for k, (name, lo, hi, bottom, top) in enumerate(REGIONS):
        cx, cy = fit.centers[k]
        sx, sy = fit.widths[k]
        rho = fit.rho[k]
        eig = np.linalg.eigvalsh([[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]])
        rows.append({"station": station, "variant": variant, "component": name,
            "center_Ex_nm": cx, "center_Em_nm": cy, "sigma_Ex_nm": sx, "sigma_Em_nm": sy,
            "rho": rho, "axis_ratio_nm": np.sqrt(eig[-1]/eig[0]),
            "inside_reference_window": bool(lo <= cx <= hi and bottom <= cy <= top),
            "initial_local_max_supported": fit.seed_supported[k],
            "median_share": float(np.median(shares[:, k])),
            "fraction_share_over_1pct": float(np.mean(shares[:, k] > 0.01)),
            "prior_distance_nm": float(np.hypot(max(lo-cx, 0, cx-hi), max(bottom-cy, 0, cy-top)))})
    return rows


def predictive_rows(train_x, test_x, frame, train, test, variant, station, fold, alpha):
    rows = []
    for target in ["BOD", "COD", "TOC"]:
        column = next(c for c in frame if c.splitlines()[0].strip() == target)
        y = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        good_train = np.isfinite(y[train])
        if good_train.sum() < 3:
            continue
        baseline = np.mean(y[train][good_train])
        if train_x is None:
            pred = np.full(len(test), baseline)
            train_pred = np.full(good_train.sum(), baseline)
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(np.log1p(np.maximum(train_x[good_train], 0)), y[train][good_train])
            pred = model.predict(np.log1p(np.maximum(test_x, 0)))
            train_pred = model.predict(np.log1p(np.maximum(train_x[good_train], 0)))
        train_r2 = r2_score(y[train][good_train], train_pred)
        for i, pos in enumerate(test):
            if np.isfinite(y[pos]):
                rows.append({"station": station, "variant": variant, "fold": fold,
                    "target": target, "sample_index": int(frame.index[pos]),
                    "observed": y[pos], "predicted": pred[i], "train_mean": baseline,
                    "train_R2": train_r2})
    return rows


def prediction_summary(predictions):
    rows = []
    for (station, variant, target), frame in predictions.groupby(["station", "variant", "target"]):
        actual, predicted = frame.observed.to_numpy(), frame.predicted.to_numpy()
        sse = np.sum((actual-predicted)**2)
        base_sse = np.sum((actual-frame.train_mean.to_numpy())**2)
        rows.append({"station": station, "variant": variant, "target": target, "n_test": len(frame),
                     "R2_oof": r2_score(actual, predicted),
                     "skill_vs_train_mean": 1-sse/base_sse if base_sse > 0 else np.nan,
                     "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
                     "MAE": mean_absolute_error(actual, predicted),
                     "train_R2_mean": frame.groupby("fold").train_R2.first().mean()})
    return pd.DataFrame(rows)


def write_station_plot(data, index, old_prediction, fit, ex, em, station, out):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    scores = fit.transform(data[None])[0]
    contributions = scores[:, None, None]*fit.full_surfaces
    prediction = contributions.sum(0)
    titles = ["Observed", "Historical V0 (in-sample)", "Improved diagonal (in-sample)",
              "Observed - improved"] + [f"{n}: contribution" for n in fit.names]
    if old_prediction is None:
        titles[1] = "Historical group V0 unavailable"
        old_prediction = np.full_like(data, np.nan)
    maps = [data, old_prediction, prediction, data-prediction, *contributions]
    rows = (len(maps)+2)//3
    figure = make_subplots(rows=rows, cols=3, subplot_titles=titles)
    vmax = max(float(data[fit.mask].max()), float(prediction[fit.mask].max()))
    residual_max = float(np.max(np.abs((data-prediction)[fit.mask])))
    for i, values in enumerate(maps):
        r, c = i//3+1, i%3+1
        figure.add_trace(go.Heatmap(x=ex, y=em, z=np.where(fit.mask, values, np.nan),
            colorscale="RdBu_r" if i == 3 else "Viridis",
            zmin=-residual_max if i == 3 else 0,
            zmax=residual_max if i == 3 else vmax, showscale=False,
            hovertemplate="Ex=%{x} nm<br>Em=%{y} nm<br>I=%{z:.4f}<extra></extra>"), r, c)
        figure.update_xaxes(title_text="Ex (nm)", row=r, col=c)
        figure.update_yaxes(title_text="Em (nm)", row=r, col=c)
        if 4 <= i < 4+len(REGIONS):
            k = i-4
            figure.add_trace(go.Scatter(x=[fit.centers[k, 0]], y=[fit.centers[k, 1]],
                mode="markers", marker={"color": "red", "symbol": "x", "size": 9},
                showlegend=False), r, c)
    figure.update_layout(title=f"{station} — EEM {index}; region names are hypotheses",
                         height=290*rows, width=1200, template="plotly_white")
    figure.write_html(out, include_plotlyjs=True)


def run_experiment(output=None, config=None, *, group_col="Point"):
    config = json.loads(json.dumps(config or DEFAULT_CONFIG))
    out = Path(output or ROOT / "runs/diagnostics/v0_improved_10stations")
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    x = np.asarray(np.load(ROOT / "data/processed/eem.npy"), float)
    frame = pd.read_csv(ROOT / "data/processed/samples.csv")
    with np.load(ROOT / "data/processed/wavelengths.npz") as wavelengths:
        ex, em = wavelengths["excitation"], wavelengths["emission"]
    source = ROOT / "runs/diagnostics/shared_gaussian_v0/by_station_top10_unmasked"
    if group_col == "Group":
        source = ROOT / "runs/diagnostics/shared_gaussian_v0/by_group_unmasked"
        selected = frame.groupby("Group").agg(n_samples=("Point", "size"),
                                               n_months=("Month", "nunique")).reset_index()
        selected["eligible"] = (selected.n_samples >= 8) & (selected.n_months >= 8)
        selected.to_csv(out / "group_eligibility.csv", index=False)
        selected = selected[selected.eligible].sort_values("Group").reset_index(drop=True)
        selected.to_csv(out / "selected_groups.csv", index=False)
    else:
        selected = pd.read_csv(source / "selected_top_stations.csv")
        selected.to_csv(out / "selected_stations.csv", index=False)
    manifest = {**config, "regions": REGIONS, "benchmark_column": group_col,
        "mask": "training_nonzero_empirical_not_verified_physical_mask",
        "prediction": "fixed Ridge(log1p amplitudes), chronological outer folds, no target selection",
        "validation": "same-month EEM-to-quality calibration on future months, not EEM-free forecast",
        "historical_V0": "full-data artifact, descriptive in-sample only"}
    (out / "config.json").write_text(json.dumps(manifest, indent=2))
    recon, regional, parameters, seeds, folds, predictions, matches, run_rows, cache_rows = (
        [] for _ in range(9))
    started = time.monotonic()
    for station_i, station in enumerate(selected[group_col]):
        print(f"[{station_i+1}/{len(selected)}] {station}", flush=True)
        indices = frame.index[frame[group_col].eq(station)].to_numpy()
        local = frame.loc[indices]
        data = x[indices]
        old_path = source / f"{slug(station)}_components.npz"
        old_pred = None
        common_mask = np.any(data != 0, axis=0)
        if old_path.exists():
            with np.load(old_path) as old:
                if np.array_equal(old["sample_index"], indices):
                    old_pred = np.einsum("nk,kij->nij", old["sample_weights"]*old["totals"][:, None],
                                         old["surfaces"])
            if old_pred is not None:
                recon.extend(reconstruction_rows(data, old_pred, common_mask, ex, em, indices,
                                                 station, "historical_V0", "full"))
                regional.extend(regional_rows(data, old_pred, common_mask, ex, em,
                                              station, "historical_V0", "full"))
        for variant_i, (variant, settings) in enumerate(config["variants"].items()):
            base_config = {**settings, "seed": config["seed"]+station_i*100}
            full_config = {**base_config, "max_steps": config["full_steps"],
                           "n_starts": config["full_starts"]}
            fit, key, hit = cached_fit(data, ex, em, out / "cache", full_config)
            cache_rows.append({"station": station, "variant": variant, "split": "full", "key": key})
            print(f"  {variant}: full {'cache' if hit else 'fit'}; loss="
                  f"{fit.runs[fit.best_start]['data_loss']:.5f}", flush=True)
            pred = fit.reconstruct(data)
            recon.extend(reconstruction_rows(data, pred, common_mask, ex, em, indices, station, variant, "full"))
            regional.extend(regional_rows(data, pred, common_mask, ex, em, station, variant, "full"))
            parameters.extend(parameter_rows(fit, data, station, variant))
            seeds.extend({"station": station, "variant": variant, "best_start": fit.best_start, **r}
                         for r in fit.seed_stability())
            run_rows.extend({"station": station, "variant": variant, "split": "full", **r} for r in fit.runs)
            np.savez_compressed(out / "models" / f"{slug(station)}_{variant}.npz",
                surfaces=fit.full_surfaces, amplitudes=fit.transform(data), mask=fit.mask,
                centers=fit.centers, widths=fit.widths, rho=fit.rho,
                sample_index=indices, excitation=ex, emission=em)
            if variant == "diag_soft_prior":
                write_station_plot(data[0], indices[0], old_pred[0] if old_pred is not None else None, fit, ex, em, station,
                                   out / "figures" / f"{slug(station)}.html")
            for fold, train, test in temporal_splits(local):
                if variant_i == 0:
                    folds.extend({"station": station, "fold": fold, "role": role,
                                  "sample_index": int(indices[pos]), "year": int(local.iloc[pos].Year),
                                  "month": int(local.iloc[pos].Month)}
                                 for role, positions in [("train", train), ("test", test)] for pos in positions)
                cfg = {**base_config, "seed": base_config["seed"]+fold+1,
                       "n_starts": config["cv_starts"], "max_steps": config["cv_steps"]}
                held, key, hit = cached_fit(data[train], ex, em, out / "cache", cfg)
                cache_rows.append({"station": station, "variant": variant, "split": f"fold{fold}", "key": key})
                run_rows.extend({"station": station, "variant": variant, "split": f"fold{fold}", **r}
                                for r in held.runs)
                held_pred = held.reconstruct(data[test])
                recon.extend(reconstruction_rows(data[test], held_pred, held.mask, ex, em,
                    indices[test], station, variant, f"fold{fold}"))
                regional.extend(regional_rows(data[test], held_pred, held.mask, ex, em,
                                               station, variant, f"fold{fold}"))
                # Descriptive alignment with full fit; never influences prediction/model choice.
                common = held.mask & fit.mask
                left = fit.full_surfaces[:-1, common]
                right = held.full_surfaces[:-1, common]
                rows, cols, similarity = match_surfaces(left, right)
                matches.extend({"station": station, "variant": variant, "fold": fold,
                                "component": fit.names[a], "matched_component": held.names[b],
                                "cosine": sim, "center_shift_nm": float(np.linalg.norm(
                                    fit.centers[a]-held.centers[b]))}
                               for a, b, sim in zip(rows, cols, similarity))
                train_scores, test_scores = held.transform(data[train]), held.transform(data[test])
                predictions.extend(predictive_rows(train_scores, test_scores, local, train, test,
                    variant, station, fold, config["ridge_alpha"]))
                if variant_i == 0:
                    predictions.extend(predictive_rows(None, None, local, train, test,
                        "train_mean", station, fold, config["ridge_alpha"]))
                    totals = np.maximum(data[:, held.mask], 0).sum(1, keepdims=True)
                    predictions.extend(predictive_rows(totals[train], totals[test], local, train, test,
                        "total_intensity", station, fold, config["ridge_alpha"]))
                    regions_x = region_features(data, held.mask, ex, em)
                    predictions.extend(predictive_rows(regions_x[train], regions_x[test], local, train, test,
                        "region_maxima", station, fold, config["ridge_alpha"]))
                print(f"    future fold {fold+1}: {len(train)} train/{len(test)} test"
                      f" {'cache' if hit else 'fit'}", flush=True)
        # Checkpoint accumulated tables after each station; cached fitting is resumable.
        for filename, records in [("reconstruction", recon), ("regional_reconstruction", regional),
            ("component_parameters", parameters), ("seed_stability", seeds), ("folds", folds),
            ("predictions", predictions), ("temporal_stability", matches), ("fit_runs", run_rows),
            ("model_manifest", cache_rows)]:
            pd.DataFrame(records).to_csv(out / f"{filename}.csv", index=False)
        print(f"  elapsed: {time.monotonic()-started:.1f}s", flush=True)
    prediction_summary(pd.DataFrame(predictions)).to_csv(out / "prediction_summary.csv", index=False)
    result = summarize(out)
    if group_col == "Group":
        for path in out.glob("*.csv"):
            table = pd.read_csv(path).rename(columns={"station": "Group"})
            if "sample_index" in table:
                table["Point"] = table.sample_index.map(frame.Point)
            table.to_csv(path, index=False)
        result["reconstruction_summary"] = result["reconstruction_summary"].rename(columns={"station": "Group"})
    return result


def summarize(out):
    out = Path(out)
    group_mode = json.loads((out / "config.json").read_text()).get("benchmark_column") == "Group"
    recon = pd.read_csv(out / "reconstruction.csv").rename(columns={"Group": "station"})
    recon["evaluation"] = np.where(recon.split.eq("full"), "in_sample", "future_months")
    by_station = recon.groupby(["station", "variant", "evaluation"]).agg(
        median_relative_rmse=("relative_rmse", "median"),
        median_peak_distance_nm=("peak_distance_nm", "median"), n=("sample_index", "count")).reset_index()
    by_station.rename(columns={"station": "Group"} if group_mode else {}).to_csv(out / "reconstruction_summary.csv", index=False)
    overview = by_station.groupby(["variant", "evaluation"]).agg(
        median_station_relative_rmse=("median_relative_rmse", "median"),
        median_station_peak_distance_nm=("median_peak_distance_nm", "median")).reset_index()
    if group_mode:
        overview = overview.rename(columns=lambda name: name.replace("median_station_", "median_group_"))
    overview.to_csv(out / "overview.csv", index=False)
    pred = pd.read_csv(out / "prediction_summary.csv")
    predictive = pred.groupby(["variant", "target"])[["R2_oof", "skill_vs_train_mean", "MAE"]].median()
    predictive.to_csv(out / "prediction_overview.csv")
    runs = pd.read_csv(out / "fit_runs.csv")
    convergence = runs.groupby("variant").agg(converged_fraction=("converged", "mean"),
                                               n_starts=("start", "count"))
    convergence.to_csv(out / "convergence.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, evaluation in zip(axes, ["in_sample", "future_months"]):
        table = by_station[by_station.evaluation.eq(evaluation)].pivot(
            index="station", columns="variant", values="median_relative_rmse")
        image = ax.imshow(table, aspect="auto", vmin=0, vmax=max(0.3, float(table.max().max())), cmap="magma_r")
        ax.set_xticks(range(len(table.columns)), table.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(table.index)), table.index)
        for i in range(len(table)):
            for j in range(len(table.columns)):
                ax.text(j, i, f"{table.iloc[i,j]:.3f}", ha="center", va="center", fontsize=8)
        ax.set_title(evaluation + ": median relative RMSE")
        fig.colorbar(image, ax=ax)
    fig.savefig(out / "figures/reconstruction_comparison.png", dpi=170)
    plt.close(fig)
    report = "# V0 cải tiến: experiment 10 trạm\n\n"
    report += "Các bảng là kết quả chạy thật. Historical V0 chỉ là tham chiếu in-sample. "
    report += "Ba temporal folds lần lượt test tháng 7–8, 9–10, 11–12; shape, mask và Ridge chỉ fit từ tháng trước.\n\n"
    report += "## Reconstruction (mỗi trạm có trọng số bằng nhau)\n\n```text\n"+overview.to_string(index=False)+"\n```\n\n"
    report += "Peak distance là sai lệch cực đại của tổng EEM, không phải ground truth sinh hóa. "
    report += "Các mô hình mới có 6 Gaussian + nền; historical V0 có 4 Gaussian + nền, nên đây không phải ablation cùng rank.\n\n"
    report += "## Dự báo ngoài thời gian: median qua 10 trạm\n\n```text\n"+predictive.to_string()+"\n```\n\n"
    report += "R² OOF dùng toàn bộ các dự đoán held-out của mỗi trạm; skill so với baseline mean fit từ train. "
    report += "Ridge alpha=10 cố định, dùng tất cả amplitudes (kể cả nền), log1p và StandardScaler fit từ train. "
    report += "Không chọn component/rank bằng target test. Có EEM tháng test làm đầu vào: đây là hiệu chỉnh chất lượng nước, không phải dự báo trước khi đo EEM.\n\n"
    report += "## Hội tụ\n\n```text\n"+convergence.to_string()+"\n```\n\n"
    report += "Đạt max_steps không chứng minh đã hội tụ. Các starts dùng chung chiến lược region-seeded nhưng bootstrap/jitter khác nhau; "
    report += "diag_no_prior tắt penalty vị trí, vẫn giữ cùng khởi tạo để ablation có kiểm soát. Prior không xác nhận danh tính hóa học. "
    report += "Mask nonzero từ train là proxy thực nghiệm; metadata scatter/blank chưa được xác minh. "
    report += "Seed stability và temporal stability cần đọc cùng residual, không chỉ nhìn reconstruction tốt.\n"
    if group_mode:
        report = report.replace("10 trạm", f"{recon.station.nunique()} group").replace("trạm", "group")
        report += "\nHistorical V0 chỉ được thêm nếu có artifact fit trên đúng mẫu của group.\n"
    (out / "report.md").write_text(report)
    print(overview.to_string(index=False), flush=True)
    return {"output": out, "overview": overview, "prediction_overview": predictive,
            "convergence": convergence, "reconstruction_summary": by_station}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/diagnostics/v0_improved_10stations")
    args = parser.parse_args()
    run_experiment(args.output)
