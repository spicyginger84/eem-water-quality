# Tra cứu tài liệu và tính mới tiềm năng cho bài báo

*21/10/2026. Phạm vi: dự đoán BOD/COD/TOC (và DOC) từ huỳnh quang EEM, khả năng chuyển giao giữa các vị trí, và các thành phần phương pháp liên quan.*

## 1. Phương pháp tra cứu và giới hạn

- **Công cụ:** tìm kiếm web học thuật (chế độ mở rộng), 15 truy vấn, chia thành 5 nhóm:
  1. EEM / huỳnh quang + BOD / COD / TOC / DOC + các họ mô hình (MLR, PLS, N-PLS, RF, XGBoost, LightGBM, SVR, ANN, GPR, CNN, transformer);
  2. kiểm chứng ngoài và chuyển giao giữa vị trí, lưu vực, thiết bị;
  3. nước lợ và cửa sông;
  4. thành phần phương pháp (kernel rừng, chọn mẫu Kennard–Stone, đặc trưng thiếu hụt oxy, TabPFN);
  5. độ tin cậy và bất định của dự đoán.
- **Giới hạn:**
  - Phần lớn kết luận dựa trên tóm tắt và đoạn trích, **chưa đọc toàn văn**.
  - Chưa tra Scopus hay Web of Science.
  - Một số thông tin thư mục còn mâu thuẫn giữa các nguồn, ví dụ tên tạp chí của nghiên cứu TOC trên sông Hàn.

  Trước khi nộp bài cần: (i) tra Scopus/WoS bằng chuỗi truy vấn ở Mục 6; (ii) đọc toàn văn các nghiên cứu ở Bảng 1 để xác nhận cách họ chia dữ liệu.

## 2. Bản đồ các nghiên cứu liên quan

**Bảng 1. Nghiên cứu dự đoán chỉ tiêu hữu cơ từ huỳnh quang.**

| Nghiên cứu | Dữ liệu | Đầu vào | Mô hình | Cách đánh giá (theo tóm tắt) | Kết quả |
|---|---|---|---|---|---|
| Hur & Cho 2012, *Sensors* | 1 sông đô thị, Hàn Quốc | PARAFAC + UV220/254 | Hồi quy | Một sông | UV220, C3, C1 tương quan tốt nhất với BOD, COD, TN |
| 22 nhà máy xử lý nước thải, Hàn Quốc, *Sensors* 2014 | 22 nhà máy | PARAFAC + SS | Hồi quy đa biến | Gộp mẫu | Thêm SS giúp dự đoán TOC/BOD/COD |
| Christian và cs. 2017, *Water Environ. Res.* | 1 cơ sở xử lý nước thải | COD, TOC, huỳnh quang | Hồi quy mũ | Một cơ sở | Ước lượng được BOD₅ từ TOC hoặc huỳnh quang |
| Hồ cao nguyên Vân Nam, *Environ. Sci. Ecotechnol.* 2023 | 68 hồ | Hấp thụ + huỳnh quang DOM | Hồi quy | **Kiểm chứng ngoài trên 6 hồ lớn, 2014–2016** | Proxy cho COD_Mn và BOD₅ |
| Sensors 2025 (Đại học Thiên Tân) | — | EEM → PARAFAC | Random forest | Chưa rõ | BOD₅ R² 0.878 |
| Anal. Bioanal. Chem. 2025 | Nước thải | EEM + UV-Vis | PLS, unfolded PLS, SVR | Tập dự đoán riêng | UPLS tốt nhất cho BOD (R² 0.845); SVR cho COD (0.738) |
| A-TEEM–PARAFAC–PLS, *Anal. Sci.* 2024 | Nước mặt | Hấp thụ–truyền qua + EEM | PARAFAC–PLS | Kiểm định phương pháp | BOD |
| **TOC sông Hàn**, 2025/2026 (*Environ. Res.* hoặc *Water Res.*, cần kiểm tra) | Lưu vực sông Hàn, nhiều loại thủy vực | 5 đỉnh EEM + 4 thông số hiện trường | XGBoost, MLR, SHAP | **Chưa rõ cách chia (cần đọc toàn văn)** | XGBoost R² 0.756 so với MLR 0.460; EC và đỉnh A, C quan trọng nhất |
| TOC + thông số địa không gian, STOTEN 2024 | Các tiểu lưu vực sông Geumho | Thông số chất lượng nước + địa không gian | XGBoost | Mùa khô và mưa | Địa không gian cải thiện TOC; EC chi phối ở sông công nghiệp |
| **Yang và cs. 2026**, *J. Environ. Manage.* 416, 130846 (đã đọc toàn văn) | Thâm Quyến: 107 mẫu nhà máy xử lý, 203 cống, 99 sông, 99 biển; 04/2023–12/2024 | EEM (PARAFAC 3 thành phần) + UV-Vis (PCA) | LR, KNN, SVR, RF, XGB, DNN, LSTM, Transformer; XGB tốt nhất | Lặp 20 lần: mô hình riêng từng loại nước; mô hình chung; **giữ lại cả loại nước (LOMO)**; **chia theo thời gian** (20 mẫu cuối mỗi loại); baseline trung bình theo loại nước | DOC: mô hình chung R² 0.96; LOMO 0.88 / 0.93 / **0.60** / 0.84 (nhà máy xử lý / cống / **sông** / biển); chia theo thời gian 0.84 / 0.90 / 0.53 / 0.84 |
| DOC + ANN, *Chemosphere* 2023 | Nhà máy xử lý nước thải | Cường độ huỳnh quang | ANN | — | Cải thiện so với hồi quy |
| DOC + cây boosting, *Water* 2021 | Biển Hoa Đông | EEM | Cây hồi quy boosting | — | DOC nước biển |
| fDOM, *Biogeosciences* 2026 | 2 hồ | Cảm biến | RF, XGB, LightGBM, CatBoost, KNN, SVR, tuyến tính | — | So sánh nhiều mô hình |
| CNN trên EEM, *Sci. Rep.* 2021 | Nước cấp | EEM | CNN | — | Dự đoán sản phẩm phụ khử trùng |
| EEM + học sâu truy nguồn ô nhiễm (Lee, Hur, Kim), *Ecol. Indic.* 2025 | 1 lưu vực, Hàn Quốc | Ảnh EEM | Học sâu | Một lưu vực | F1 0.91 (bài toán phân loại nguồn) |
| Dập tắt huỳnh quang làm chẩn đoán độ tin cậy (Hu, Jacquin, Morgenroth), ES&T 2025 | Nước xám | PARAFAC + chất dập tắt | Chỉ số F0/F | — | Gắn cờ khi quan hệ huỳnh quang–nồng độ bị phá vỡ |

**Bằng chứng về khả năng chuyển giao:**
- **Hiệu chỉnh riêng cho từng vị trí luôn tốt nhất.** Nghiên cứu cảm biến huỳnh quang ở nhiều nhà máy xử lý nước thải cho thấy điều này. Gộp các vị trí vẫn dùng được, nhưng kém hơn.
- **Mô hình surrogate thường không chuyển giao được.** Một nghiên cứu dữ liệu tần suất cao (PLOS ONE) nói thẳng rằng các mô hình này "thường đặc thù theo vị trí". Họ thử một mô hình hỗn hợp chung cho nhiều vị trí, nhưng cho bùn cát và dinh dưỡng, không phải BOD/COD.
- **Hiệu chỉnh độ đục cho cảm biến fDOM phụ thuộc vị trí** (USGS).
- **Trong nước lợ**, thành phần humic-like trộn bảo toàn theo độ mặn, còn protein-like thì không (nghiên cứu cửa sông Trung Quốc). Một hiệu chỉnh duy nhất, bỏ qua độ mặn, vì vậy có rủi ro.

**Khung phương pháp về rò rỉ không gian:**
- Chia ngẫu nhiên đánh giá quá cao hiệu năng trên dữ liệu địa lý: chương sách về cross-validation không gian trong GeoAI; nghiên cứu viễn thám về phụ thuộc không gian giữa tập huấn luyện và kiểm tra.
- Water Research 2026 cảnh báo rò rỉ do vị trí trùng lặp và tương quan thời gian khi chuyển giao giữa các thủy vực.

## 3. Đối chiếu baseline với repo

**Bảng 2. Các họ mô hình xuất hiện trong tài liệu, và trạng thái trong repo.**

| Họ mô hình | Xuất hiện trong tài liệu | Trong repo | Cần bổ sung |
|---|---|---|---|
| Hồi quy một đỉnh (đỉnh T hoặc C), dạng tuyến tính hoặc mũ | Rất phổ biến: Christian 2017, nhiều nghiên cứu sông | **Chưa có** | **Có**: baseline đơn giản nhất mà phản biện sẽ hỏi |
| MLR trên đỉnh, FRI, PARAFAC | Hur & Cho 2012; 22 nhà máy 2014; sông Hàn (MLR) | Có (`peaks_mlr`, `fri_mlr`, `parafac_mlr`, cả bản `+field`) | — |
| PARAFAC + PLS, unfolded PLS, N-PLS | Anal. Bioanal. Chem. 2025; A-TEEM 2024; nghiên cứu UASB | Có (`upls`, `npls`) | Nên thêm PLS trên điểm PARAFAC (tùy chọn) |
| PCA/PCR + SVR | Phổ học | Có (`pca_svr`) | — |
| SVR trên đặc trưng | Anal. Bioanal. Chem. 2025 | Có (`svr`, `gard_svr`, `gard_matern`) | — |
| Random forest | Sensors 2025 (BOD); truy nguồn DOM | Có (`parafac_rf`) | Nên thêm RF trên đỉnh, FRI và thông số hiện trường |
| XGBoost | TOC sông Hàn; STOTEN 2024 | Có (`xgb`, `parafac_xgb`) | — |
| LightGBM, CatBoost | fDOM, Biogeosciences 2026 | Không có trong pipeline hiện tại (pipeline cũ có LightGBM) | **Có**: chi phí thấp |
| ANN/MLP | DOC Chemosphere 2023; nhiều nghiên cứu BOD | **Chưa có** | **Có**: rất phổ biến |
| Hồi quy quá trình Gauss (GPR) | Chất lượng nước (Gaza WWTP), bất định | **Chưa có** | **Có**: cũng là baseline về bất định |
| KNN | fDOM 2026 | Chưa có | Có thể thêm |
| CNN trên ảnh EEM | Sci. Rep. 2021; Ecol. Indic. 2025; J. Hazard. Mater. 2026 | Có code (`cnn.py`, `runs/litbench_cnn_v1`) nhưng chưa vào `benchmark_v2` | **Có**: chạy lại trên dữ liệu đã sửa |
| TabPFN (mô hình nền tảng cho dữ liệu bảng) | Hiệu chỉnh phổ cận hồng ngoại (arXiv 2026) | Đã chạy trên Kaggle (dữ liệu cũ) | Chạy lại trên dữ liệu đã sửa |
| Có UV254/UV220 hoặc A-TEEM | Hur & Cho 2012; A-TEEM 2024; COD từ UV-Vis | Không có dữ liệu | Ghi rõ là ngoài phạm vi |

**Kết luận về baseline:** để phản biện không chỉ ra baseline bị bỏ sót, cần thêm 6 mô hình:
1. hồi quy một đỉnh dạng mũ;
2. MLP;
3. GPR;
4. LightGBM hoặc CatBoost;
5. RF trên bộ đặc trưng đầy đủ;
6. CNN trên EEM.

Đồng thời chạy lại TabPFN trên dữ liệu đã sửa. Tất cả dùng cùng giao thức zero-shot và few-shot.

## 4. Khoảng trống và tính mới tiềm năng

| # | Khoảng trống | Bằng chứng từ tra cứu | Đóng góp có thể nêu | Mức độ mới (ước lượng) |
|---|---|---|---|---|
| G1 | **Chưa có đánh giá giữ lại từng trạm cho BOD/COD/TOC (chỉ tiêu tổng, gồm cả phần dạng hạt) từ EEM + đầu vào đo được tại hiện trường, trên một mạng lưới nước mặt nhiều trạm.** Lưu ý: Yang 2026 đã đánh giá giữ lại cả *loại nước* và chia theo *thời gian* cho DOC, nên tổng quát hóa nói chung không còn là khoảng trống. | Các nghiên cứu ở Bảng 1 dùng một vị trí, một lưu vực hoặc gộp mẫu. Gần nhất là kiểm chứng ngoài trên 6 hồ (Vân Nam), nhưng dùng hồi quy tuyến tính trên proxy quang học. Tài liệu cảm biến khuyến nghị hiệu chỉnh tại chỗ. | Benchmark 43 trạm, zero-shot theo trạm và few-shot nhân quả. Định lượng mức đánh giá quá cao của chia ngẫu nhiên (0.05 tới hơn 0.8 R²). | **Cao.** Cần đọc toàn văn nghiên cứu sông Hàn để chắc họ không dùng chia theo trạm. |
| G2 | **Chưa có phân tích giới hạn thông tin** (vì sao không vượt được R² nhất định) | Không thấy nghiên cứu nào làm phép đo oracle hay phân rã sai số theo trạm và theo mùa | Trần oracle; 4% mẫu chiếm 60–75% sai số; các đợt chất hữu cơ không phát huỳnh quang mùa lạnh ở suối công nghiệp/nước lợ | **Cao** |
| G3 | **Kernel lai rừng + Matérn trong SVR** | Kernel rừng có trong tài liệu học máy (Breiman; Davies & Ghahramani 2014; arXiv 2009.00089, 2012.10737). Kernel lai trong GPR có cho dữ liệu đất. **Không thấy** RF-kernel + Matérn trong SVR. | HTK-R: kernel cây học trên phần dư của kernel trơn | **Trung bình–cao** (mới về phương pháp; mức cải thiện khiêm tốn, chỉ cho BOD) |
| G4 | **Đặc trưng động học oxy cho dự đoán BOD từ EEM** | Thiếu hụt DO có làm đặc trưng cho chỉ số chất lượng nước (Sci. Rep. 2026); **không thấy** đi kèm EEM để dự đoán BOD | DO_sat (Benson–Krause, hiệu chỉnh độ mặn) + độ thiếu hụt; BOD tốt hơn có ý nghĩa thống kê | **Trung bình–cao** |
| G5 | **Chọn mẫu hiệu chỉnh cho trạm mới** | Kennard–Stone phổ biến khi chuyển giao hiệu chỉnh giữa *thiết bị* (cận hồng ngoại, huỳnh quang laser). **Không thấy** dùng để chọn mẫu phòng thí nghiệm cho *trạm quan trắc mới* | Chọn đa dạng kiểu Kennard–Stone: cải thiện cả 3 chỉ tiêu so với "3 mẫu đầu" | **Trung bình–cao** (mới về ứng dụng; nên so thêm SPXY) |
| G6 | **Nước lợ và suối công nghiệp** | Hành vi trộn khác nhau giữa các thành phần; chưa thấy mô hình dự đoán cho nước lợ | Phân tích sai số theo loại trạm; đặc trưng độ mặn và oxy | Trung bình |
| G7 | **Độ tin cậy, cảnh báo khi EEM "bị mù"** | ES&T 2025 dùng dập tắt huỳnh quang (cần thêm thao tác thí nghiệm). Tài liệu E. coli nêu thiếu định lượng bất định. | Cảnh báo dựa trên dị thường SS, mùa và loại nước; khoảng conformal theo trạm | Trung bình (cần thêm thí nghiệm) |
| G8 | **Kiểm soát chất lượng dữ liệu bằng dấu vân tay phổ** | Không thấy | Đã phát hiện một lỗi tráo file thật | Thấp–trung bình (phù hợp làm mục phụ) |

## 5. Đề xuất bài báo Q1

**Tên dự kiến:** *How transferable are fluorescence-based organic matter models? A station-held-out benchmark, information limits and physics-informed hybrid kernels for BOD, COD and TOC*

**Đóng góp chính:**
- G1 + G2: benchmark và giới hạn thông tin.
- G3 + G4: HTK-R với đặc trưng oxy.
- G5: chọn mẫu hiệu chỉnh cho trạm mới.

**Tạp chí phù hợp (Q1):**
- Water Research
- Environmental Science & Technology (hoặc ES&T Letters cho bản rút gọn G1 + G2)
- Science of the Total Environment
- Journal of Hazardous Materials
- Environmental Science and Ecotechnology
- Water Research X

**Rủi ro và cách giảm thiểu:**

1. **Cải thiện của phương pháp khiêm tốn.** BOD zero-shot chỉ tăng +0.03 R² (station-MAE tốt hơn có ý nghĩa ở 3–4/6 cách chia); COD/TOC không đổi.
   - Đặt trọng tâm vào G1, G2 và G5. Trình bày G3/G4 như đóng góp phụ, có ablation đầy đủ.
2. **Baseline bị chê thiếu.** Bổ sung 6 mô hình ở Mục 3.
3. **Chỉ một năm dữ liệu, một vùng.** Thêm kiểm định LOSO; kiểm định theo khối thời gian (huấn luyện các tháng trước, kiểm tra các tháng sau); và nếu được, dùng một bộ dữ liệu ngoài.
4. **Nhãn nghi vấn.** Báo cáo kết quả có và không có 5 dòng bị gắn cờ.
5. **Độ ổn định của kết quả.** Chạy nhiều seed cho các mô hình ngẫu nhiên.

## 6. Chuỗi truy vấn đề xuất cho Scopus / Web of Science

```
TITLE-ABS-KEY(("excitation-emission matri*" OR EEM OR "fluorescence spectr*" OR PARAFAC)
  AND (BOD OR "biochemical oxygen demand" OR COD OR "chemical oxygen demand" OR TOC OR DOC OR "organic carbon")
  AND (predict* OR estimat* OR model* OR surrogate))
AND (( "cross-site" OR "external validation" OR transferab* OR "leave-one" OR "independent site*" OR generaliz*))
```

Chuỗi phụ:

```
("random forest kernel" OR "forest kernel" OR "proximity kernel") AND (regression OR SVR)
("Kennard-Stone" OR "sample selection") AND (recalibration OR "site-specific calibration") AND (sensor OR water)
("oxygen deficit" OR "oxygen saturation") AND (BOD) AND ("machine learning" OR fluorescence)
```

## Nguồn

- [Hur & Cho 2012 – BOD, COD, TN từ EEM-PARAFAC + UV](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3279249/)
- [22 nhà máy xử lý nước thải Hàn Quốc – EEM-PARAFAC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3926638/)
- [Christian và cs. 2017 – COD, TOC, huỳnh quang để ước lượng BOD](https://onlinelibrary.wiley.com/doi/abs/10.2175/106143016X14504669768976)
- [Hồ cao nguyên Vân Nam – proxy quang học cho COD_Mn và BOD₅](https://www.sciencedirect.com/science/article/pii/S2666498423000911)
- [Sensors 2025 – BOD từ huỳnh quang 3D + học máy](https://doi.org/10.3390/s25030711)
- [Anal. Bioanal. Chem. 2025 – BOD/COD từ UV-Vis và EEM](https://link.springer.com/article/10.1007/s00216-025-06267-3)
- [A-TEEM–PARAFAC–PLS cho BOD](https://link.springer.com/article/10.1007/s44211-024-00605-8)
- [TOC sông Hàn – XGBoost + huỳnh quang + thông số tại chỗ](https://pubmed.ncbi.nlm.nih.gov/41448416/)
- [TOC + thông số địa không gian (STOTEN 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0048969724038907)
- [ANN cho DOC từ huỳnh quang ở nhà máy xử lý nước thải](https://www.sciencedirect.com/science/article/abs/pii/S0045653523012997)
- [EEM + cây boosting cho DOC](https://doi.org/10.3390/w13243612)
- [Học máy cho DOM, Biogeosciences 2026](https://bg.copernicus.org/articles/23/2661/2026/)
- [DOC từ proxy độ thơm trong nước nội địa](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12355942/)
- [CNN dự đoán sản phẩm phụ khử trùng từ huỳnh quang](https://www.nature.com/articles/s41598-021-03881-w)
- [EEM + học sâu cho ô nhiễm hữu cơ ở lưu vực hỗn hợp](https://www.sciencedirect.com/science/article/pii/S1470160X25012105)
- [Dập tắt huỳnh quang để chẩn đoán độ tin cậy (ES&T 2025)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12444986/)
- [Dự đoán thiếu oxy mùa hè ở sông từ huỳnh quang DOM với học máy diễn giải được](https://www.researchgate.net/publication/403549152_Predicting_Summer_River_Hypoxia_from_DOM_Fluorescence_Signatures_Using_Interpretable_Machine_Learning_and_Kinetic_Validation)
- [Truy nguồn DOM sông bằng huỳnh quang và học máy](https://www.nature.com/articles/s41545-024-00370-1)
- [Mô hình surrogate tần suất cao, tính đặc thù vị trí (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0215503)
- [USGS – hiệu chỉnh huỳnh quang theo từng vị trí](https://www.usgs.gov/publications/clearing-waters-evaluating-need-site-specific-field-fluorescence-corrections-based)
- [Cross-validation không gian (GeoAI)](https://www.acsu.buffalo.edu/~yhu42/papers/2023_GeoAIHandbook_SpatialCV.pdf)
- [Phụ thuộc không gian giữa tập huấn luyện và kiểm tra](https://link.springer.com/article/10.1007/s10994-021-05972-1)
- [Chuyển giao giữa thủy vực – cảnh báo rò rỉ (Water Research 2026)](https://www.sciencedirect.com/science/article/abs/pii/S004313542601064X)
- [Random Forest kernel (arXiv 2009.00089)](https://arxiv.org/pdf/2009.00089)
- [Kernel từ cây quyết định (arXiv 2012.10737)](https://arxiv.org/pdf/2012.10737)
- [GPR với kernel lai cho cacbon hữu cơ trong đất](https://github.com/Lbyyyyyyy/Hybrid-kernel-Gaussian-process-regression_Matlab)
- [Kennard–Stone trong chuyển giao hiệu chỉnh huỳnh quang laser (OSTI)](https://www.osti.gov/pages/servlets/purl/2376340)
- [Hiệu chỉnh chuyển giao trong hóa phân tích – tổng quan](https://www.spectroscopyonline.com/view/calibration-transfer-chemometrics-part-ii-review-subject)
- [Chỉ số chất lượng nước từ đặc trưng oxy, gồm thiếu hụt DO (Sci. Rep. 2026)](https://www.nature.com/articles/s41598-026-36156-3)
- [TabPFN cho hiệu chỉnh phổ cận hồng ngoại](https://arxiv.org/pdf/2605.21544)
- [GPR cho chất lượng nước nhà máy xử lý nước thải Gaza](https://pubs.rsc.org/en/content/articlehtml/2024/va/d3va00170a)
- [Dự đoán BOD bằng mạng nơ-ron và cây ensemble ở Hàn Quốc](https://www.sciencedirect.com/science/article/abs/pii/S0301479720307659)
