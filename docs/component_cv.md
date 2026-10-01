# CV component GMM và PARAFAC

Hai script đánh giá bắt đầu từ EEM thô trong `data/processed/eem.npy`, lọc target thiếu rồi chia KFold. Mỗi fold fit component và mask trên train, suy ra biên độ NNLS trên các bề mặt cố định cho train/test, rồi fit hồi quy trên train. Cache chỉ dùng lại khi tập train giống hệt nhau. Không đọc component hay feature full-group để tính CV.

- GMM: `scripts/diagnostics/evaluate_v0_component_regression.py --output runs/diagnostics/v0_improved_groups`. Dùng `cv_steps`, `cv_starts`, regions và biến thể trong config của experiment.
- PARAFAC: `scripts/diagnostics/evaluate_parafac_all_components.py --rank 4 --include-single`. Rank phải khai báo trước, không lấy rank đã chọn bằng toàn bộ group. Solver CV là `fit_peak_model` nonnegative CP có mask, raw loss; khác solver TensorLy của phần phân rã mô tả trong notebook. Mặc định 3 starts, 500 steps.

Chạy bằng `.venv/bin/python`. Các notebook đã gọi cùng luồng đánh giá. Chạy lại các cell CV và bảng phụ thuộc để cập nhật CSV/biểu đồ; các artifact cũ không tự đổi khi sửa mã.

`R2_train` và file feature full-group chỉ mô tả fit in-sample; `R2_cv_mean` và `R2_oof` lấy từ fit riêng trong fold. PARAFAC C1..Crank được sắp theo đỉnh Em rồi Ex của train, không đối chiếu với fit toàn nhóm; cùng nhãn giữa các fold chưa đảm bảo cùng danh tính hóa học.

Best single component vẫn được chọn trên chính CV báo cáo, có selection optimism. Cần nested CV hoặc tập test độc lập nếu muốn đánh giá toàn bộ quy trình chọn component. Shuffled KFold cũng không đánh giá khả năng dự đoán thời gian tương lai; dùng temporal protocol cho mục tiêu đó.
