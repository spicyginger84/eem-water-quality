# EEM water-quality benchmark

Dự đoán BOD, COD, TOC (mg/L) từ phổ huỳnh quang EEM và 5 thông số hiện trường (Temp, pH, DO, SS, EC),
đánh giá trên **trạm mới**, tức trạm không có mẫu nào trong dữ liệu huấn luyện.

- **Dữ liệu:** 757 mẫu, 43 trạm, vùng Seoul–Gyeonggi, năm 2023.
- **Báo cáo:** [docs/station_cv_benchmark/report.md](docs/station_cv_benchmark/report.md).

## Pipeline

```
raw data ──scripts/data/build_processed.py──▶ data/processed ──eem-quality run CONFIG──▶ runs/<name>/
                (+ data/manifest)                 │
                                                  ├─ features.py   đặc trưng: peaks, FRI, field, season, oxygen, PARAFAC
                                                  ├─ models.py     các mô hình (bảng MODEL_SPECS)
                                                  ├─ protocols.py  cách chia (random, station*, loso), few-shot
                                                  └─ metrics.py    R², station-MAE…, bootstrap ghép cặp theo trạm
```

| Thành phần | Nội dung |
| --- | --- |
| `scripts/data/` | `raw_to_samples.py` ghép mỗi file EEM với đúng một dòng phòng thí nghiệm; lọc độ sâu ≤ 2 m; yêu cầu đủ BOD, COD, TOC, SS, EC. `build_processed.py` áp `data/manifest/` (mẫu bị loại, cờ nhãn), tạo các cách chia fold theo trạm, tính mask tán xạ và ghi `data/processed/` cùng `processed.zip` |
| `src/eem_water_quality/` | Pipeline gồm `data`, `features`, `models`, `cnn`, `protocols`, `metrics`, `runner`; dòng lệnh `eem-quality` |
| `configs/benchmark/` | `smoke.json` (chạy thử), `full.json` (21 mô hình × 7 scheme), `cnn_station.json`, `cnn.json` |
| `tests/` | 12 test trên dữ liệu giả |

**Mô hình.**
- Mô hình phi tuyến học trên log y; mô hình tuyến tính học trên mg/L.
- Siêu tham số chọn bằng kiểm định chéo bên trong, cùng kiểu chia với vòng ngoài.
- Hậu tố `+field` nghĩa là thêm 5 thông số hiện trường.

| ID | Mô hình |
| --- | --- |
| `peaks_mlr`, `fri_mlr`, `parafac_mlr` | Đỉnh Coble + chỉ số / 5 vùng FRI / PARAFAC 4 thành phần → hồi quy tuyến tính |
| `parafac_rf`, `parafac_xgb` | PARAFAC → random forest / XGBoost |
| `upls`, `npls`, `pca_svr` | EEM trải phẳng → PLS1; EEM 3 chiều → N-PLS1; 20 thành phần chính → SVR RBF |
| `cnn` | Ảnh log-EEM → CNN 2D, một mạng cho 3 chỉ tiêu (cần torch) |
| `xgb`, `svr` | XGBoost / SVR RBF trên khối bảng: peaks + FRI + field + season |
| `gard_svr`, `gard_matern` | SVR có trọng số theo nhóm đặc trưng, kernel RBF / Matérn-5/2 |
| `lgbm`, `lgbm_oxy` | LightGBM trên khối bảng; `lgbm_oxy` thêm khối `oxygen` (DO bão hòa, thiếu hụt oxy) |

**Đánh giá.**
- **Cách chia theo trạm:** `station` (seed 0) và `station_gkf5_seed{11,23,42,2026,3407}` là 6 cách chia
  5 fold, sinh bởi `station_folds()` trong `build_processed.py`.
  - Mỗi trạm nằm trọn trong một fold.
  - Các trạm được phân tầng theo loại trạm (suối công nghiệp hay sông/hồ, lấy từ sheet Location) và
    nước lợ (trung vị EC của trạm > 2000 µS/cm). Mỗi fold vì vậy có số trạm của từng tầng chênh nhau
    tối đa 1. Với dữ liệu hiện tại, mỗi fold có 148–155 mẫu.
  - Trong mỗi tầng, trạm lớn được xếp trước, vào fold đang có ít trạm của tầng đó nhất, rồi đến fold có
    ít mẫu nhất. Seed chỉ đổi thứ tự giữa các trạm có cùng số mẫu.
  - 6 cách chia gần như độc lập với nhau: ARI trung bình 0.05.
- **Cách chia khác:** `random` là KFold theo mẫu, chỉ để tham chiếu. `loso` là để lại một trạm mỗi lần.
- **Zero-shot:** chấm trên cả 757 mẫu.
- **Few-shot (K = 3):** dự đoán được cộng một độ lệch log riêng của trạm, có co rút. Ba thiết kế chọn nhãn: `early` (3 mẫu đầu), `rolling` (3 nhãn ngay trước mẫu cần dự đoán), `random` (dùng nhãn tương lai, chỉ để đối chiếu).

## Cài đặt và dữ liệu

```bash
python -m pip install -e ".[research,dev]"      # thêm ".[cnn]" nếu chạy CNN

# Cách 1: dùng bản dựng sẵn
unzip data/processed.zip -d data

# Cách 2: dựng lại từ dữ liệu thô (khoảng 1 phút); kết quả trùng bản dựng sẵn từng bit
unzip "2026ER_data_for Viet.Zip" -d /path/raw
python scripts/data/build_processed.py /path/raw --out data/processed

eem-quality inspect                             # kiểm tra: 757 mẫu, 43 trạm
```

## Tái lập kết quả của báo cáo

| Kết quả | Lệnh | Thời gian (10 luồng) |
| --- | --- | --- |
| Bảng 5 (zero-shot, cột chia ngẫu nhiên) và Bảng 6 (few-shot): các hàng PLS, PARAFAC, MLR, XGBoost, SVR, Group SVR; cùng `lgbm`, `lgbm_oxy` | `eem-quality run configs/benchmark/full.json --out runs/benchmark_full` | khoảng 13 phút |
| Hàng CNN 2D trong Bảng 5 và 6 | `eem-quality run configs/benchmark/cnn_station.json --out runs/benchmark_cnn` | chậm trên CPU |

Đọc kết quả trong thư mục run:
- `report.md`: các bảng R² và station-MAE.
- `summary.csv`: zero-shot và few-shot, trung bình và min qua 6 cách chia theo trạm (`regime`, `r2_mean`, `station_mae_mean`).
- `metrics.csv`: chỉ số theo từng scheme, gồm `random` và `mae_top10`.
- `predictions.parquet`: dự đoán từng mẫu.
- `paired.csv`: hiệu station-MAE so với `reference`, kèm KTC bootstrap.

So sánh hai mô hình trên một tập scheme tùy chọn:

```bash
eem-quality run configs/benchmark/full.json --out runs/x \
    --models gard_matern,lgbm_oxy --schemes station,station_gkf5_seed11,loso
```

**Phần của báo cáo không tái lập được bằng pipeline này:**
- **Bảng 5 và 6:** các hàng "Một đỉnh dạng mũ", MLP, GPR, KNN, Random forest, CatBoost, và LightGBM với tham số mặc định. Hàng LightGBM đó khác `lgbm` ở đây.
- **Mục 6–7:** phân tích sai số và 35 hướng cải tiến.

Các phần này được chạy bằng script thí nghiệm riêng, không nằm trong repo.

**Kết quả chính khi chạy `full.json`** (R², zero-shot, trung bình 6 cách chia theo trạm):

| Mô hình | BOD | COD | TOC |
| --- | --- | --- | --- |
| `gard_matern` | 0.49 | 0.66 | 0.67 |
| `lgbm_oxy` | 0.54 | 0.62 | 0.65 |
| `parafac_xgb+field` | 0.46 | 0.59 | 0.61 |

`lgbm_oxy` được thêm sau ngày báo cáo (22/10/2026); xem ghi chú cập nhật ở đầu báo cáo.

## Kiểm tra code

```bash
ruff check src tests scripts
pytest -q
```

`LOCAL/` (workspace) và `runs/` (kết quả chạy) không nằm trong Git.
