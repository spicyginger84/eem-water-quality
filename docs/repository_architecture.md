# Kiến trúc repository và quy chuẩn benchmark

Tài liệu này quy định cách tổ chức mã nguồn, dữ liệu, thí nghiệm và artifact cho
bài toán dự đoán chỉ số chất lượng nước từ EEM. Đây là cấu trúc mục tiêu để phát
triển dần; việc sắp xếp lại thư mục không được làm thay đổi kết quả của pipeline
hiện tại nếu chưa có một phiên bản migration riêng.

Phạm vi hiện tại chỉ gồm pipeline machine learning cổ điển (Linear Regression,
SVR, Decision Tree và XGBoost) với EEM raw, EEM PCA và tabular features. Các
phương pháp ngoài phạm vi benchmark sẽ được đặc tả trong tài liệu riêng khi có
đủ validation.

## Nguyên tắc

- Một nguồn sự thật cho mỗi loại thông tin: schema dữ liệu, cấu hình run,
  cách chia tập và metric không được định nghĩa lại trong notebook.
- Chia tập trước, sau đó mới `fit` mọi phép biến đổi có học tham số. PCA, scaler,
  imputer và mask học từ dữ liệu chỉ được fit trên tập train của protocol tương
  ứng.
- Benchmark mặc định là grouped 5-fold cross-validation. Mỗi fold có một nhóm
  test khác nhau; mọi experiment/model đều được tính điểm trên cả năm fold và
  lưu mean, độ lệch chuẩn và pooled score.
- Không dùng bất kỳ fold test nào để chọn model, chọn số thành phần hoặc điều
  chỉnh hyperparameter. LOMO/LOSO là protocol chẩn đoán riêng và không chạy
  thêm 5-fold bên trong.
- Mọi experiment phải tái lập được bằng seed, cấu hình, phiên bản code và hash
  dữ liệu.
- So sánh công bằng nghĩa là các model dùng cùng sample, cùng split, cùng feature
  schema và cùng tiêu chí đánh giá.
- Notebook dùng để khám phá và trình bày; logic có thể chạy lại phải nằm trong
  package hoặc script có tham số.

## Mẫu thiết kế phù hợp

- **Pipeline pattern:** mỗi stage nhận một object có schema rõ ràng và trả về
  object của stage kế tiếp. Stage không tự tìm file hoặc tự thay đổi config.
- **Strategy + Registry:** mỗi model classical (`linear`, `svr`, `tree`,
  `xgboost`) là một strategy cùng interface `fit`/`predict`; registry ánh xạ tên
  CLI sang factory. Thêm model mới không cần sửa vòng lặp đánh giá.
- **Transformer pattern:** PCA, scaler, mask và imputer có `fit`/`transform`,
  được gom vào một `PreprocessorBundle` và serialize cùng model.
- **Dataclass/schema boundary:** `FeatureSpec`, `SplitManifest` và
  `RunManifest` làm hợp đồng giữa các module; config không truyền dưới dạng
  dictionary tự do trong toàn bộ pipeline.
- **Adapter ở biên:** loader cho CSV, Parquet, NPY hoặc workbook chuyển dữ liệu
  về cùng `ProcessedDataset`; phần model không phụ thuộc định dạng file.

## Cấu trúc thư mục mục tiêu

```text
repo/
├── README.md
├── pyproject.toml
├── configs/                         # YAML/JSON cấu hình experiment
├── docs/
│   ├── repository_architecture.md   # tài liệu này
│   ├── data_contract.md             # schema và quy tắc dữ liệu
│   ├── benchmark_protocol.md        # protocol đánh giá
│   ├── Improvement_plan.md
│   └── phase2_report.md
├── src/eem_water_quality/
│   ├── __main__.py                  # entry point rất mỏng
│   ├── cli.py                       # parse CLI và dispatch
│   ├── data/
│   │   ├── io.py                    # đọc/ghi processed data
│   │   ├── schema.py                # dataclass/schema và validation
│   │   └── splitting.py             # random, grouped, two-way split
│   ├── preprocessing/
│   │   ├── eem.py                   # mask, reshape, chuẩn hóa EEM
│   │   └── tabular.py               # xử lý biến phụ trợ
│   ├── features/
│   │   ├── specs.py                 # FeatureSpec và feature schema
│   │   ├── interfaces.py            # FeatureTransformer contract
│   │   ├── raw_eem.py               # flattened raw EEM
│   │   ├── pca.py                   # EEM PCA representation
│   │   ├── tabular.py               # SS, EC, Temp, pH và combinations
│   │   └── registry.py              # feature representations và factory
│   ├── models/
│   │   ├── interfaces.py            # Regressor contract
│   │   ├── registry.py              # tên model và factory
│   │   └── classical.py             # LR, SVR, tree, XGBoost
│   ├── evaluation/
│   │   ├── metrics.py               # metric thuần, không có side effect
│   │   ├── baselines.py             # global/station baseline
│   │   ├── protocols.py             # benchmark và LOMO/LOSO holdout
│   │   └── cross_validation.py      # grouped 5-fold benchmark mặc định
│   ├── pipelines/
│   │   └── classical.py             # orchestration ML
│   └── artifacts/
│       ├── io.py                    # ghi CSV, JSON, model
│       └── provenance.py             # manifest, hash, phiên bản
├── scripts/
│   ├── data/process_raw_data.py
│   ├── evaluation/run_holdout.py
│   └── diagnostics/
│       ├── plot_eem_pca_groups.py
│       └── plot_correlations.py
├── notebooks/
│   ├── eda/
│   ├── experiments/
│   └── reports/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── data/
│   ├── raw/                         # dữ liệu gốc, thường không commit
│   ├── interim/                     # dữ liệu trung gian có thể tái tạo
│   └── processed/                   # input chính của model
└── runs/                            # output theo run, thường không commit
```

Hiện tại package vẫn dùng một số module phẳng như `data.py`, `features.py`,
`ml.py` và `artifacts.py`. Có thể giữ chúng trong giai đoạn chuyển tiếp; bảng
ánh xạ bên dưới cho biết vị trí đích mà không yêu cầu đổi tên ngay lập tức.

| Module hiện tại | Vị trí đích | Trách nhiệm |
|---|---|---|
| `data.py` | `data/io.py`, `data/schema.py`, `data/splitting.py` | đọc dữ liệu, schema, chia tập |
| `features.py` | `features/specs.py`, `features/raw_eem.py`, `features/pca.py`, `features/tabular.py`, `features/registry.py` | đặc tả và tạo feature |
| `ml.py` | `pipelines/classical.py`, `evaluation/protocols.py` | huấn luyện và đánh giá ML |
| `metrics.py` | `evaluation/metrics.py` | tính metric |
| `artifacts.py` | `artifacts/io.py`, `artifacts/provenance.py` | lưu output và provenance |
| `scripts/evaluate_holdouts.py` | `scripts/evaluation/run_holdout.py` | CLI chẩn đoán holdout |

## Đánh giá khả năng mở rộng

Cấu trúc trên có thể mở rộng nếu các module mới tuân theo interface hiện có và
không chèn logic riêng vào vòng lặp pipeline. Mỗi loại mở rộng có một điểm vào
riêng:

| Nhu cầu | Nơi thêm code | Thay đổi ngoài module đó |
|---|---|---|
| Model classical mới | `models/classical.py` và `models/registry.py` | thêm factory entry và unit test |
| EEM representation mới | module representation trong `features/` và `features/registry.py` | thêm `FeatureSpec`, không sửa evaluator |
| Tabular feature set mới | `features/specs.py` hoặc catalog config | thêm schema/validation |
| Protocol đánh giá mới | `evaluation/protocols.py` | thêm artifact schema và integration test |
| Metric mới | `evaluation/metrics.py` | cập nhật metric schema, không sửa model |
| Định dạng input mới | `data/io.py` adapter | trả về cùng `ProcessedDataset` |
| Cách lưu output mới | `artifacts/io.py` | giữ nguyên logical columns |

Để giữ tính mở rộng:

- `FeatureSpec` phải mô tả representation, tabular columns, target exclusions
  và preprocessing options; evaluator không được kiểm tra tên feature bằng
  các câu lệnh `if` rải rác.
- Mọi feature transformer phải tuân theo `FeatureTransformer` (`fit`,
  `transform`, `feature_schema`); mọi model phải tuân theo `Regressor` (`fit`,
  `predict`, `get_params`). Chúng không được đọc trực tiếp từ filesystem.
- Registry chỉ ánh xạ tên cấu hình sang implementation; registry không được
  chứa logic đánh giá hoặc logic ghi artifact.
- Protocol chỉ làm orchestration. Việc thêm model/feature mới không được yêu
  cầu sửa `grouped_5fold`, LOSO hoặc LOMO.
- Output dùng schema dạng dài với các cột `target`, `feature_set`, `model`,
  `protocol`, `fold` và metric; thêm model mới chỉ tạo thêm dòng, không tạo một
  format CSV mới.
- Mỗi implementation mới phải có unit test cho interface và integration test
  chạy được trên dataset nhỏ với cùng pipeline.

Vì vậy, việc tách `models`, `features`, `evaluation` và `artifacts` là hợp lý
cho mở rộng. Việc đổi tên module phẳng thành nhiều package con chỉ nên làm sau
khi đã có interface và test contract; nếu đổi thư mục trước, repo sẽ có nhiều
đường import nhưng chưa có ranh giới chức năng rõ ràng.

## Quy ước đặt tên

Tên function phải nói rõ đối tượng và hành động:

| Prefix | Dùng cho | Ví dụ |
|---|---|---|
| `load_`, `read_` | đọc dữ liệu | `load_processed_dataset` |
| `save_`, `write_` | ghi artifact | `write_run_manifest` |
| `validate_`, `check_` | kiểm tra điều kiện | `validate_processed_schema` |
| `split_` | tạo partition | `split_by_station` |
| `fit_` | học tham số từ train | `fit_eem_pca` |
| `transform_` | áp dụng tham số đã fit | `transform_features` |
| `build_`, `make_` | tạo object hoặc ma trận | `build_feature_matrix`, `make_model` |
| `compute_` | phép tính thuần | `compute_regression_metrics` |
| `evaluate_` | chấm điểm đã có prediction | `evaluate_predictions` |
| `run_` | orchestration cấp pipeline | `run_classical_pipeline` |
| `summarize_` | tổng hợp kết quả | `summarize_holdout_metrics` |

Class nên là danh từ, ví dụ `FeatureSpec`, `DatasetSchema`,
`SplitManifest`, `FeatureBuilder`, `ModelResult`, `RunArtifacts`. Tránh các tên
mơ hồ như `process`, `do_stuff`, `best_model` hoặc `data` khi có thể nêu rõ
ngữ nghĩa. Function không nên âm thầm đọc file, dùng global state hoặc fit lại
transformer trong `transform_*`; các phụ thuộc phải nhận qua đối số.

## Hợp đồng dữ liệu đầu vào

Processed dataset là một dataset theo sample, trong đó cùng một `sample_id`
liên kết EEM, metadata và target. Tối thiểu cần có:

```text
data/processed/
├── eem.npy                 # float32, shape (n_samples, n_emission, n_excitation)
├── samples.parquet|csv     # đúng n_samples dòng, metadata và target
├── wavelengths.npz         # emission và excitation, khớp hai chiều EEM
├── eem_mapping.csv         # provenance của từng EEM
└── processing_summary.json # version, filter và thống kê xử lý
```

Quy tắc bắt buộc:

- `sample_id` là khóa ổn định; không nối dữ liệu bằng vị trí dòng sau khi đã
  filter nếu không đồng thời lưu manifest.
- `eem[i]` phải tương ứng với dòng có cùng `sample_id` trong bảng metadata.
- EEM phải có kích thước chung, kiểu số hữu hạn và wavelength axes đầy đủ.
- Metadata nên có `station`, `month`/`date`, `round`, `depth`, đường dẫn EEM và
  trạng thái mapping.
- Quy tắc lọc như `depth <= 2` hoặc kiểm tra `COD >= BOD` phải là option có tên,
  được ghi trong manifest; không được lọc ngầm trong model code.
- Tên target và feature phải được chuẩn hóa một lần. Khi dự đoán `TOC`, không
  đưa `TOC` hoặc biến dẫn xuất trực tiếp của nó vào feature.

## Hợp đồng của pipeline

Mỗi pipeline phải tuân theo các stage sau. Với benchmark mặc định, các stage
được lặp độc lập cho từng fold:

1. `load_processed_dataset`: đọc dữ liệu và kiểm tra schema.
2. `validate_dataset`: kiểm tra số dòng, wavelength, finite values và metadata.
3. `make_evaluation_splits`: mặc định chạy 5-fold CV. `--split random` dùng
   KFold, `--split group` dùng GroupKFold. `--split two_way` là ngoại lệ: dùng
   một outer holdout hai chiều và không chạy cross-validation. Nếu có
   `--holdout-protocol loso` hoặc `--holdout-protocol lomo`, chỉ tạo split của
   holdout được chọn và tắt CV cùng các split khác.
4. `fit_preprocessors`: trong mỗi fold, fit mask, scaler, PCA hoặc imputer chỉ
   trên phần train; lưu feature schema và tham số của fold đó.
5. `transform_features`: áp dụng đúng bundle của fold cho train và held-out
   test; không fit lại trên held-out rows.
6. `fit_candidates`: fit từng model trên cùng train features.
7. `evaluate_fold`: dùng model đã fit trên train để dự đoán held-out test,
   tính metric và lưu prediction cho từng candidate/fold.
8. `aggregate_cv_metrics`: tính mean, standard deviation và pooled metric từ
   các held-out folds; không chọn candidate dựa trên test fold.
9. `write_artifacts`: ghi fold predictions, summary metrics, model/schema của
   từng fold và provenance.

### Chế độ đánh giá

`--evaluation-protocol cv` là mặc định và dùng `--cv-folds 5`. `--split` quyết
định loại cross-validation:

- `random`: KFold trên sample, phù hợp khi station có thể xuất hiện ở cả train
  và held-out fold;
- `group`: GroupKFold theo `--group-col` (mặc định `Point`), bảo đảm station
  không xuất hiện ở hai phía của một fold;
- `two_way`: một custom holdout duy nhất, trong đó test đồng thời thuộc station
  mới và month mới. Các rows chỉ có một group mới bị loại khỏi train để không
  làm yếu định nghĩa strict two-way holdout. `--cv-folds` không áp dụng.

Với `random` và `group`, năm fold dùng bốn phần để train và phần còn lại làm
held-out test. Với `two_way`, chỉ có một held-out test và không có fold summary
CV. Tất cả model và feature set dùng cùng fold/holdout manifest. Kết quả phải có
metric từng fold hoặc held-out group và summary:

- `mean`: trung bình số học của năm fold;
- `std`: độ biến thiên giữa các fold;
- `pooled`: gộp toàn bộ held-out predictions rồi tính một lần.

Đây là protocol so sánh model, không phải protocol chọn model. Mọi candidate đều
được chạy trên cả năm fold; pipeline không gọi `argmin` trên fold test và không
tạo một model được chọn từ kết quả CV.

`LOSO` (leave-one-station-out) và `LOMO` (leave-one-month-out) là các protocol
chẩn đoán định hướng trong **cùng pipeline chính**. Chúng được bật bằng flag
riêng `--holdout-protocol`, không lồng thêm random/group k-fold CV. Bản thân
LOSO/LOMO vẫn là các leave-one-group-out folds. Mục tiêu của chúng là xác định
độ nhạy với station/month mới, nên metric được báo cáo theo từng group và summary
riêng.

Một run được xác định bởi tích Descartes của ba danh sách:

```text
targets × features × models
```

`--evaluation-protocol` chỉ chọn cách đánh giá; `--split` chọn splitter khi
protocol là CV; `--holdout-protocol` là override chẩn đoán. Không option nào tự
chọn target, feature hay model. Để run tái lập được, phải ghi rõ cả ba danh
sách. Baseline global mean được pipeline thêm tự động và không tính là một
feature set.

Các default của pipeline classical:

| Flag | Default | Ý nghĩa |
|---|---|---|
| `--data` | `data/processed` | processed dataset |
| `--output` | `runs/<timestamp>_ml` | thư mục artifact mới |
| `--evaluation-protocol` | `cv` | cross-validation mặc định |
| `--split` | `group` | GroupKFold theo station |
| `--cv-folds` | `5` | số fold cho random/group CV |
| `--group-col` | `Point` | group chính |
| `--secondary-group-col` | `Month` | group thứ hai cho two-way |
| `--holdout-protocol` | `none` | `loso` hoặc `lomo`; override CV khi được bật |
| `--targets` | `BOD_COD` | target mặc định an toàn |
| `--features` | `EEMpca` | feature mặc định, không chứa target đo |
| `--models` | `linear tree xgboost` | các model classical mặc định |
| `--seed` | `42` | seed tái lập |
| `--n-jobs` | `1` | số worker mặc định |
| `--test-size` | `0.2` | chỉ dùng cho two-way/holdout outer split |
| `--val-size` | `0.2` | chỉ dùng cho single split tương thích |
| `--pca-components` | `30` | số thành phần của EEM PCA |

`--test-size` chỉ áp dụng cho two-way hoặc holdout có outer split; nó không thay
đổi số fold của random/group CV. `--val-size` chỉ áp dụng cho `single_split`
tương thích và không được dùng trong benchmark CV.

Thứ tự ưu tiên khi nhiều option cùng xuất hiện là:

```text
--holdout-protocol loso|lomo  >  --split two_way  >  --evaluation-protocol cv
```

Do đó, `--split two_way` tự chuyển run thành một two-way holdout và bỏ qua
`--cv-folds`; không được kết hợp nó với `--holdout-protocol`. Nếu bật
`--holdout-protocol`, `--split`, `--evaluation-protocol` và `--cv-folds` không
được dùng để tạo thêm một protocol khác; config thực tế phải được ghi vào
`run.json`.

Ví dụ interface CLI mục tiêu:

```bash
# grouped 5-fold mặc định, hai feature representations và ba model
python -m eem_water_quality ml \
  --data data/processed \
  --output runs/grouped5fold_eempca \
  --targets BOD COD TOC BOD_COD \
  --features EEMpca EEMpca_SS_EC \
  --models linear svr xgboost \
  --group-col Point --seed 42 --pca-components 30 --n-jobs 1

# leave-one-station-out: một held-out fold cho mỗi station
python -m eem_water_quality ml \
  --data data/processed \
  --output runs/loso_eempca \
  --targets BOD COD TOC BOD_COD \
  --features EEMpca EEMpca_SS_EC \
  --models linear svr xgboost \
  --holdout-protocol loso --group-col Point \
  --seed 42 --pca-components 30 --n-jobs 1

# leave-one-month-out: một held-out fold cho mỗi month
python -m eem_water_quality ml \
  --data data/processed \
  --output runs/lomo_eempca \
  --targets BOD COD TOC BOD_COD \
  --features EEMpca EEMpca_SS_EC \
  --models linear svr xgboost \
  --holdout-protocol lomo --group-col Month \
  --seed 42 --pca-components 30 --n-jobs 1

# unseen station + unseen month diagnostic
python -m eem_water_quality ml \
  --data data/processed --output runs/two_way_eempca \
  --targets BOD COD TOC BOD_COD \
  --features EEMpca EEMpca_SS_EC \
  --models linear svr xgboost \
  --split two_way --group-col Point --secondary-group-col Month \
  --seed 42 --pca-components 30 --n-jobs 1
```

`--holdout-protocol` là flag riêng cho LOMO/LOSO và chỉ nhận `loso` hoặc `lomo`.
Không dùng đồng thời các boolean flag `--loso` và `--lomo`. Không truyền flag
nghĩa là chạy grouped 5-fold mặc định. Khi truyền flag, pipeline chỉ chạy
protocol tương ứng: không chạy grouped 5-fold, không tạo single split và không
chạy two-way holdout. `--group-col` phải là `Point` cho LOSO và `Month` cho
LOMO. Two-way vẫn dùng `--split two_way` như một protocol chẩn đoán riêng và
không được kết hợp với `--holdout-protocol`. Script trong `scripts/evaluation/`
chỉ nên là wrapper mỏng hoặc công cụ phân tích kết quả, không được có một
implementation holdout khác với pipeline.

Khi dự đoán `TOC`, phải loại mọi feature set có `TOC` trong tên hoặc
trong `tabular` specification, chẳng hạn `EEMpca_TOC` và `TOC_SS_EC`. Ví dụ
trên chỉ dùng `EEMpca` và `EEMpca_SS_EC`, nên dùng được cho cả bốn target mà
không đưa TOC đo trong phòng thí nghiệm vào feature.

Nếu cần protocol train/validation/test một lần cho mục đích tương thích, phải
đặt tên rõ là `single_split`; nó không thay thế benchmark grouped 5-fold và
không được trộn vào bảng kết quả CV.

## Hợp đồng artifact đầu ra

Mỗi lần chạy có một `run_id` bất biến:

```text
runs/<run_id>/
├── run.json                 # config, git commit, seed, data hash, version
├── splits.csv               # sample_id, partition, station, month, fold
├── feature_schema.json      # feature names, shape, wavelength, mask
├── metrics/
│   ├── cv_fold_metrics.csv  # metric của từng model/features/fold
│   ├── cv_summary.csv       # mean, std và pooled của CV mặc định
│   ├── validation_metrics.csv  # chỉ cho protocol single_split
│   ├── test_metrics.csv        # chỉ cho protocol single_split
│   └── holdout_metrics.csv     # LOSO, LOMO và two_way
├── predictions/
│   ├── cv_predictions.csv  # held-out prediction, có cột fold
│   ├── validation_predictions.csv  # chỉ cho protocol single_split
│   ├── test_predictions.csv        # chỉ cho protocol single_split
│   └── holdout_predictions.csv     # LOSO, LOMO và two_way
├── models/<target>/<features>/<model>/
│   └── fold_<id>/model.joblib
└── diagnostics/             # PCA, residual và warning reports
```

Tên file hiện tại có thể còn nằm trực tiếp dưới `target_00`, nhưng schema nên
được chuẩn hóa trước khi đổi layout. Không cần `selected.json` trong protocol
này vì pipeline không chọn model bằng validation hoặc CV. Mỗi dòng metric nên có
ít nhất: `target`, `features`, `model`, `protocol`, `fold`, `n_train`,
`n_test`, `r2`, `rmse`, `mae`. Dòng summary bổ sung `r2_mean`, `r2_std` và
`r2_pooled`. Với grouped holdout cần lưu cả R² trung bình theo group và R² pooled,
vì chúng lần lượt coi mỗi group hoặc mỗi observation có trọng số ngang nhau.

Prediction phải có `sample_id`, `target`, `features`, `model`, `protocol`,
`fold`, `y_true`, `y_pred` và các group keys (`station`, `month`) nếu có. Không
ghi prediction mà thiếu khóa để truy ngược về sample gốc.

Với `loso` và `lomo`, cột `fold` hoặc `heldout_group` phải xác định station/month
đang bị giữ lại. `holdout_metrics.csv` phải có metric của từng group và các dòng
summary `mean`, `std`, `pooled`; không ghi chung vào `cv_summary.csv`.

## Chuẩn benchmark và kiểm thử

- Mặc định dùng grouped 5-fold; cùng một fold manifest cho tất cả model và
  feature set trong một bảng so sánh.
- Baseline global mean luôn được ghi; station mean chỉ dùng khi station có mặt
  trong train và phải ghi rõ đây là group baseline.
- Fit preprocessing trên train của từng fold; kiểm tra leakage bằng test tự động
  rằng train và held-out test không giao group.
- Report metric từng fold, `R²_mean`, `R²_std` và `R²_pooled`; với LOMO/LOSO
  report thêm metric theo từng group.
- Không chọn model bằng fold test. Mọi candidate phải được chạy trên cùng năm
  fold và được lưu prediction tương ứng.
- Không áp dụng grouped 5-fold bên trong LOMO/LOSO hoặc two-way holdout.
- Ghi số sample bị loại và lý do; không bỏ qua warning mapping hoặc target thiếu.
- Unit test cho schema, split và metric; integration test chạy pipeline trên
  dataset nhỏ; regression test kiểm tra số sample, shape EEM và schema artifact.
- Mỗi run ghi seed, số thread, phiên bản package, hash dữ liệu và command đã
  dùng. Không dùng timestamp làm thông tin duy nhất để tái lập run.

## Lộ trình chuyển đổi không đổi kết quả

1. Thêm các dataclass `DatasetSchema`, `FeatureSpec`, `SplitManifest` và
   `RunManifest` quanh API hiện tại.
2. Bổ sung validation đầu vào và output schema; giữ nguyên tên CLI hiện có.
3. Tách các function thuần từ module phẳng sang package con, thêm import tương
   thích trong giai đoạn chuyển tiếp.
4. Đưa grouped 5-fold vào protocol mặc định; thêm integration tests kiểm tra
   train-only fitting, group disjointness, fold aggregation và cấm test-based
   selection.
5. Đưa LOMO/LOSO vào cùng pipeline qua `--holdout-protocol`; giữ two-way là
   protocol chẩn đoán riêng qua `--split two_way`. Cả hai loại holdout đều
   không được lồng grouped CV.
6. Chuẩn hóa layout artifact rồi mới đổi tên thư mục `target_00` thành tên
   target nếu cần.
7. Sau khi mọi test và một run mẫu khớp kết quả cũ, mới xóa module/đường dẫn cũ.

Các bước này giúp tổ chức lại repo mà không trộn lẫn thay đổi kiến trúc với
thay đổi thuật toán hoặc thay đổi protocol đánh giá.
