"""Compare saved single-component LR/XGB results on the same ten stations."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from evaluate_v0_component_regression import markdown_table

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/diagnostics"
OUT = BASE / "model_comparison/improved_v0_vs_v0_parafac"
METHODS = ["PARAFAC", "V0_old", "diag_no_prior", "diag_soft_prior", "rho_soft_prior"]


def compare():
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = []
    improved = BASE / "v0_improved_10stations/component_regression"
    source_paths.append(improved / "station_best_component_r2.csv")
    current = pd.read_csv(source_paths[-1]).rename(columns={"variant": "method", "model": "regressor"})
    assert current.n_features.eq(1).all()
    current["selected_rank"] = 6
    frames = [current]
    sources = {"V0_old": BASE / "shared_gaussian_v0/by_station_top10_unmasked",
               "PARAFAC": BASE / "parafac/by_station_top10"}
    for method, folder in sources.items():
        for regressor, token in [("linear", "linear_cv"), ("xgb", "xgb")]:
            path = folder / f"station_best_component_{token}_r2.csv"
            source_paths.append(path)
            data = pd.read_csv(path).rename(columns={f"R2_{regressor}_{k}": f"R2_{k}"
                                                    for k in ["train", "cv_mean", "cv_std"]})
            data["method"], data["regressor"] = method, regressor
            frames.append(data)
    columns = ["method", "regressor", "station", "target", "component", "feature", "n",
               "selected_rank", "R2_train", "R2_cv_mean", "R2_cv_std", "cv_splits", "status"]
    data = pd.concat(frames, ignore_index=True)[columns]
    assert len(data) == 300 and data.status.eq("ok").all()
    assert data.cv_splits.eq(5).all()
    assert not data.component.isin(["background", "all_components"]).any()
    reference = data[data.method.eq("V0_old")].set_index(["regressor", "station", "target"]).n.sort_index()
    for method, table in data.groupby("method"):
        assert table.station.nunique() == 10
        pd.testing.assert_series_equal(table.set_index(["regressor", "station", "target"]).n.sort_index(), reference)
    # Check sample identities/order independently of aggregate sample counts.
    ids = None
    for folder in [*sources.values(), improved]:
        path = folder / "station_component_max_features.csv"
        source_paths.append(path)
        features = pd.read_csv(path)
        variants = features.groupby("variant") if "variant" in features else [("historical", features)]
        for _, table in variants:
            actual = table[["station", "sample_index"]].sort_values(["station", "sample_index"]).reset_index(drop=True)
            assert not actual.duplicated().any()
            if ids is None:
                ids = actual
            else:
                pd.testing.assert_frame_equal(actual, ids)
    data.to_csv(OUT / "best_components_long.csv", index=False)
    summary = data.groupby(["regressor", "method", "target"]).agg(
        train_mean=("R2_train", "mean"), cv_mean=("R2_cv_mean", "mean"),
        cv_median=("R2_cv_mean", "median"), positive_cv_stations=("R2_cv_mean", lambda s: int((s > 0).sum())),
        stations=("station", "nunique")).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    pairwise = []
    for regressor, table in data.groupby("regressor"):
        wide = table.pivot(index=["station", "target"], columns="method", values="R2_cv_mean")[METHODS]
        wide.reset_index().to_csv(OUT / f"{regressor}_cv_by_station.csv", index=False)
        for method in METHODS[2:]:
            for baseline in METHODS[:2]:
                delta = wide[method]-wide[baseline]
                for target in ["BOD", "COD", "TOC", "ALL"]:
                    d = delta if target == "ALL" else delta.xs(target, level="target")
                    pairwise.append({"regressor": regressor, "method": method, "baseline": baseline,
                        "target": target, "wins": int((d > 1e-10).sum()),
                        "ties": int((d.abs() <= 1e-10).sum()), "losses": int((d < -1e-10).sum()),
                        "mean_delta_cv": d.mean(), "median_delta_cv": d.median()})
    pairs = pd.DataFrame(pairwise)
    pairs.to_csv(OUT / "paired_comparisons.csv", index=False)
    manifest = {"comparison": "existing saved results; no refitting or model selection in this script",
        "aligned": "same 284 sample IDs, ten stations, targets, one Fmax, 5-fold shuffled CV seed 42",
        "decomposition_scope": "full station before regression CV for all methods",
        "candidate_components": "improved 6; old V0 4; PARAFAC 3 or 4",
        "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    (OUT / "provenance.json").write_text(json.dumps(manifest, indent=2))
    report = "# So sánh improved V0, V0 cũ và PARAFAC — từng component với target\n\n"
    report += "Đã đối chiếu cùng **284 sample IDs thuộc 10 trạm**, cùng target và số mẫu. "
    report += "Mỗi mô hình hồi quy chỉ dùng một Fmax; bỏ background. Component tốt nhất được chọn riêng theo CV mean "
    report += "cho mỗi phương pháp × trạm × target × estimator. Không ghép tên T/B với C1/C2 như cùng chất.\n\n"
    report += "`diag_soft_prior` là bản improved mặc định đang xem; trình bày thêm `diag_no_prior` và `rho_soft_prior` "
    report += "để thấy độ nhạy với prior/covariance. LR là LinearRegression; XGB có cùng cấu hình 80 cây, depth=2, lr=0.05. "
    report += "CV là KFold=5, shuffle=True, seed=42. Số liệu lấy từ artifact đã lưu; không fit lại mô hình trong script so sánh.\n\n"
    report += "## Tổng hợp\n\nMỗi trạm có trọng số bằng nhau. `cv_mean` là trung bình của mean 5-fold R² qua 10 trạm; "
    report += "`cv_median` là median qua trạm, không phải pooled OOF R².\n\n"
    report += markdown_table(summary) + "\n\n"
    report += "## So sánh cặp\n\nWins nghĩa là R² cao hơn, kể cả khi cả hai vẫn âm; không phải số trạm đã dự báo tốt.\n\n"
    report += markdown_table(pairs[pairs.target.eq("ALL")]) + "\n\n"
    report += "## Nhận xét\n\n"
    report += "- Improved V0 cải thiện mean CV so với hai phương pháp cũ ở cả ba target, nhưng mean CV vẫn âm. "
    report += "Prior diagonal không tốt hơn ở mọi trạm; Banwol là ví dụ suy giảm rõ.\n"
    report += "- Với XGB, lợi ích rõ nhất của `diag_soft_prior` là TOC: mean CV từ −0,368 (V0 cũ) / −0,758 (PARAFAC) "
    report += "lên −0,132; median gần 0 và 5/10 trạm có CV mean dương. BOD vẫn khó.\n"
    report += "- `rho_soft_prior` có mean CV tốt nhất trong năm phương pháp ở cả LR/XGB và cả ba target trong các artifact này; "
    report += "đây là quan sát khám phá, không phải xác nhận lựa chọn phương pháp trên test độc lập.\n"
    report += "- Ví dụ Anseong: Linear–COD improved diagonal 0,566, V0 cũ 0,441, PARAFAC 0,336. "
    report += "Nhưng XGB–TOC tại cùng trạm: improved diagonal 0,460, V0 cũ 0,598, PARAFAC 0,521. "
    report += "Một phương pháp không thắng đồng đều giữa estimator/target.\n"
    report += "- Reconstruction tốt hơn không tự động làm Fmax của một component dự đoán tốt hơn. "
    report += "Dữ liệu sinh hóa cần kiểm định riêng; CV target không xác nhận center đúng hóa học.\n\n"
    report += "## Giới hạn so sánh\n\n"
    report += "1. Decomposition của cả ba họ đều đã fit toàn bộ trạm trước CV hồi quy. Đây chưa phải end-to-end CV.\n"
    report += "2. Best component được chọn và báo trên cùng CV. Improved có 6 ứng viên, V0 cũ 4, PARAFAC 3–4: "
    report += "cơ hội chọn được điểm CV cao do ngẫu nhiên khác nhau. Không thể quy chênh lệch hoàn toàn cho mô hình peak tốt hơn.\n"
    report += "3. Tám trạm chỉ có 23 mẫu (fold test 4–5 mẫu), nên mean R² nhạy với phương sai nhỏ/outlier. "
    report += "Không dùng các mean này để kết luận ý nghĩa thống kê; đọc cùng median, SD và từng trạm.\n"
    report += "4. XGB có train R² cao hơn không có nghĩa tổng quát tốt hơn. "
    report += "Không so các số này với temporal Ridge CV trong notebook như cùng protocol.\n"
    report += "5. Các lượt improved đã hết ngân sách trước tiêu chí hội tụ. "
    report += "Chưa có reconstruction PARAFAC trên cùng metric/miền trong bảng này; không trộn relative error của PARAFAC với median relative RMSE của improved.\n\n"
    for regressor, table in data.groupby("regressor"):
        report += f"## {regressor.upper()}: R² CV theo từng trạm–target\n\n"
        wide = table.pivot(index=["station", "target"], columns="method", values="R2_cv_mean")[METHODS].reset_index()
        report += markdown_table(wide)+"\n\n"
        report += f"### {regressor.upper()}: component được chọn, train R² và CV ± SD\n\n"
        detail = table.copy()
        detail["CV ± SD"] = detail.apply(lambda r: f"{r.R2_cv_mean:.3f} ± {r.R2_cv_std:.3f}", axis=1)
        detail = detail[["station", "target", "method", "component", "R2_train", "CV ± SD"]].sort_values(["station", "target", "method"])
        report += markdown_table(detail)+"\n\n"
    (OUT / "report.md").write_text(report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    colors = ["#8c8c8c", "#bc7845", "#83b5d1", "#306ab0", "#36a080"]
    for row, regressor in enumerate(["linear", "xgb"]):
        for col, target in enumerate(["BOD", "COD", "TOC"]):
            values = summary[summary.regressor.eq(regressor) & summary.target.eq(target)].set_index("method").loc[METHODS]
            ax = axes[row, col]
            ax.bar(range(5), values.cv_mean, color=colors)
            ax.scatter(range(5), values.cv_median, marker="D", color="black", s=20, label="Median")
            ax.axhline(0, color="black", lw=.8)
            for i, v in enumerate(values.cv_mean):
                ax.text(i, v-.025, f"{v:.3f}", ha="center", va="top", fontsize=8)
            ax.set_ylim(-1, .05)
            ax.set_xticks(range(5), METHODS, rotation=35, ha="right", fontsize=8)
            ax.set_title(f"{regressor.upper()} — {target}")
            ax.set_ylabel("R² CV across stations")
            ax.legend(fontsize=8)
    fig.suptitle("Best single component per station/target — bars: mean; diamonds: median\nFixed-feature regression CV; unequal candidate component counts")
    fig.savefig(OUT / "cv_comparison.png", dpi=170)
    plt.close(fig)
    print(summary.round(3).to_string(index=False))
    print(pairs[pairs.target.eq("ALL")].round(3).to_string(index=False))
    print("Saved:", OUT)
    return data, summary, pairs


if __name__ == "__main__":
    compare()
