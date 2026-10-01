# GMM phân rã phổ EEM: ý tưởng, giả định và cơ sở toán học

> Cập nhật mã đánh giá CV: GMM và PARAFAC fit component, mask và các đại lượng chuẩn hóa riêng trên train của từng fold; held-out chỉ suy ra biên độ NNLS với component cố định. Các kết quả số đã lưu từ protocol full-group cũ chưa được chạy lại và không đại diện cho protocol mới.

## 1. Câu hỏi nghiên cứu

**Tổng các bề mặt Gaussian dùng chung có biểu diễn được cấu trúc EEM và tạo đặc trưng hữu ích để ước lượng BOD, COD, TOC hay không? So với PARAFAC, biểu diễn này khác gì về hình dạng thành phần và hiệu quả thực nghiệm?**

Trong báo cáo, **GMM** là tên tạm của mô hình `improved v0`: sáu bề mặt Gaussian và một nền phẳng. Tên này không chỉ thuật toán GMM phân cụm thông thường.

## 2. Ý tưởng và giả định

Mỗi EEM được xem là tổng các vùng tín hiệu chồng lấp. Mỗi vùng có tâm, độ rộng và độ nghiêng dùng chung giữa các mẫu. Cường độ đóng góp của từng vùng thay đổi theo mẫu.

Mô hình dựa trên năm giả định:

1. **Tính cộng:** tín hiệu kỳ vọng bằng tổng các thành phần và nền.
2. **Hình dạng chung:** trong một nhóm, tâm và hình dạng thành phần không đổi giữa các mẫu.
3. **Dạng Gaussian:** mỗi thành phần được xấp xỉ bằng một vùng trơn, đơn đỉnh.
4. **Đóng góp không âm:** biên độ và bề mặt không âm; dữ liệu quan sát vẫn có thể âm do xử lý hoặc sai số.
5. **Nền phẳng:** mỗi mẫu có một mức nền cộng không đổi theo bước sóng.

Các giả định này mô tả tín hiệu phổ. Chúng không xác nhận mỗi Gaussian là một chất hóa học riêng biệt.

## 3. Cơ sở toán học

### 3.1. Bề mặt Gaussian

Gọi \(Y_{nj}\) là cường độ mẫu \(n\) tại ô bước sóng \(z_j=(Ex_j,Em_j)\). Tập ô hợp lệ là \(\mathcal M\), gồm \(J\) ô.

Thành phần \(k\) có tâm \(\mu_k\) và ma trận hình dạng

\[
\mu_k=\begin{pmatrix}\mu_{Ex,k}\\\mu_{Em,k}\end{pmatrix},\qquad
\Sigma_k=\begin{pmatrix}
\sigma_{Ex,k}^2 & \rho_k\sigma_{Ex,k}\sigma_{Em,k}\\
\rho_k\sigma_{Ex,k}\sigma_{Em,k} & \sigma_{Em,k}^2
\end{pmatrix}.
\]

Bề mặt chưa chuẩn hóa và bề mặt chuẩn hóa:

\[
g_k(z_j)=\exp\left[-\frac12(z_j-\mu_k)^\top\Sigma_k^{-1}(z_j-\mu_k)\right],
\qquad
\phi_{kj}=\frac{g_k(z_j)}{\sum_{r\in\mathcal M}g_k(z_r)}.
\]

Do đó \(\phi_{kj}\ge0\) và \(\sum_{j\in\mathcal M}\phi_{kj}=1\). Chuẩn hóa được tính trên mask của train.

- \(\mu_k\): vị trí trung tâm vùng phổ.
- \(\sigma_{Ex,k},\sigma_{Em,k}\): độ rộng theo hai trục.
- \(\rho_k\): độ nghiêng; \(\rho_k=0\) cho Gaussian tách biến theo Ex và Em.

### 3.2. Mô hình quan sát và quá trình sinh dữ liệu

Quá trình sinh dữ liệu trả lời câu hỏi: **nếu biết hình dạng các component và mức đóng góp của chúng trong một mẫu, ta tạo ra EEM của mẫu đó như thế nào?** Đây là giả định toán học về cách hình thành tín hiệu, chưa phải mô tả đầy đủ các phản ứng hóa học trong nước.

#### Bước 1. Xác định các component tín hiệu chung

Mỗi component là một gò hình chuông trên bản đồ Ex–Em. Tâm quyết định gò nằm ở đâu; độ rộng và độ nghiêng quyết định tín hiệu lan ra quanh tâm như thế nào. Công thức Gaussian ở phần 3.1 cho giá trị của gò tại từng ô bước sóng.

Sau chuẩn hóa, \(\phi_{kj}\) cho biết **phần đóng góp của component \(k\) phân bổ vào ô \(j\)**, với

\[
\sum_{j\in\mathcal M}\phi_{kj}=1.
\]

Ví dụ, \(\phi_{kj}=0.02\) nghĩa là ô đó nhận 2% tổng đóng góp của component trên miền đang xét. Sáu hình dạng này được giữ chung cho các mẫu trong cùng nhóm.

#### Bước 2. Gán mức đóng góp riêng cho từng mẫu

Với mẫu \(n\), gán sáu hệ số không âm \(a_{n1},\ldots,a_{n6}\). Hệ số \(a_{nk}\) cho biết component \(k\) đóng góp tổng cộng bao nhiêu tín hiệu. Đóng góp tại từng ô là

\[
D_{nkj}=a_{nk}\phi_{kj}.
\]

Nếu \(a_{nk}=100\) và \(\phi_{kj}=0.02\), component đóng góp 2 đơn vị cường độ tại ô đó. Tăng \(a_{nk}\) lên 200 làm đóng góp tăng thành 4, nhưng không đổi tâm hay độ rộng. Nếu \(a_{nk}=0\), component không xuất hiện trong tín hiệu tái tạo của mẫu.

Như vậy, mỗi component có **một hệ số cho cả mẫu**, không có một hệ số tự do riêng ở từng ô. Cấu trúc giữa các ô được quyết định bởi hình dạng Gaussian.

#### Bước 3. Cộng các component và nền tại từng ô

Tại mỗi cặp Ex–Em, cộng đóng góp của cả sáu component. Thêm một nền phẳng có tổng đóng góp \(a_{n,bg}\), được chia đều cho \(J\) ô:

\[
m_{nj}=\sum_{k=1}^{6}a_{nk}\phi_{kj}+\frac{a_{n,bg}}{J}.
\]

\(m_{nj}\) là cường độ mà mô hình kỳ vọng tại ô \(j\). Thực hiện phép tính này cho mọi ô sẽ thu được toàn bộ EEM tín hiệu của mẫu. Một ô có thể nhận đóng góp từ nhiều component chồng lấp; không cần chọn ô đó thuộc component nào.

#### Bước 4. Thêm sai số để thu được EEM quan sát

\[
Y_{nj}=m_{nj}+\varepsilon_{nj}.
\]

Sai số \(\varepsilon_{nj}\) chứa ảnh hưởng của phép đo, tiền xử lý và phần cấu trúc phổ mà mô hình chưa mô tả được. Sai số có thể dương hoặc âm. Vì vậy, EEM sau xử lý có thể có giá trị âm dù phần tín hiệu \(m_{nj}\) không âm.

Ví dụ minh họa tại **một ô bước sóng**, với \(J=100\) ô hợp lệ:

| Nguồn đóng góp | Phân bổ tại ô | Tổng đóng góp trong mẫu | Cường độ tại ô |
|---|---:|---:|---:|
| Component 1 | \(\phi_{1j}=0.02\) | \(a_{n1}=100\) | 2.0 |
| Component 2 | \(\phi_{2j}=0.01\) | \(a_{n2}=300\) | 3.0 |
| Component 3–6 | — | Biên độ bằng 0 trong ví dụ | 0.0 |
| Nền | \(1/J=0.01\) | \(a_{n,bg}=20\) | 0.2 |
| **Tín hiệu kỳ vọng** | | | **5.2** |
| Sai số | | | -0.1 |
| **Giá trị quan sát** | | | **5.1** |

Các số trên chỉ minh họa phép tính, không phải số liệu thực nghiệm. Ở ô khác, các giá trị \(\phi_{kj}\) thay đổi theo vị trí bước sóng, còn các biên độ của mẫu giữ nguyên.

#### Bước 5. Tạo mẫu khác bằng cách thay mức đóng góp

Mẫu tiếp theo dùng lại sáu hình dạng, nhưng có bộ biên độ và mức nền khác. Chẳng hạn, nếu component 1 tăng còn component 2 giảm, EEM sẽ sáng hơn quanh vùng thứ nhất và yếu hơn quanh vùng thứ hai. Peak của tổng tín hiệu có thể dịch chuyển dù tâm từng component không đổi.

| Dùng chung trong một nhóm | Riêng cho từng mẫu |
|---|---|
| Sáu tâm, độ rộng và độ nghiêng | Sáu biên độ component |
| Lưới bước sóng và mask đánh giá | Một biên độ nền |
| Các hình dạng chuẩn hóa \(\phi_k\) | Sai số tại các ô |

**Sinh dữ liệu và học mô hình là hai chiều ngược nhau.** Chiều sinh đi từ hình dạng và biên độ đến EEM. Khi fit, ta đã có EEM và cần tìm hình dạng cùng biên độ sao cho tổng tái tạo gần quan sát nhất. Với mẫu mới, giữ hình dạng đã học và chỉ suy ra biên độ từ EEM mới.

Triển khai hiện tại chưa quy định phân bố để lấy ngẫu nhiên các biên độ, cũng chưa học phân bố sai số đầy đủ. Vì thế, đây là mô hình cộng tín hiệu có sai số; chưa phải bộ sinh EEM ngẫu nhiên hoàn chỉnh. BOD/COD/TOC không tham gia quá trình tạo hoặc học các component này.

#### Liên hệ với tên GMM

Đặt \(A_n=\sum_{k=1}^{6}a_{nk}+a_{n,bg}\). Vì mỗi hình dạng có tổng bằng một, \(A_n\) là tổng cường độ **kỳ vọng** trên mask. Khi \(A_n>0\):

\[
\frac{m_{nj}}{A_n}=\sum_{k\in\{1,\ldots,6,bg\}}\pi_{nk}\phi_{kj},
\qquad \pi_{nk}=\frac{a_{nk}}{A_n},\qquad \sum_k\pi_{nk}=1,
\qquad \phi_{bg,j}=\frac1J.
\]

Tín hiệu chuẩn hóa có dạng hỗn hợp các Gaussian và nền đều. Đây là cơ sở dùng tên GMM. Thuật toán thực tế fit cường độ bằng bình phương sai số, không dùng EM và không buộc tổng cường độ tái tạo bằng tổng cường độ quan sát.

### 3.3. Hàm mục tiêu

Mô hình tối thiểu hóa bình phương sai số có trọng số:

\[
\mathcal L_{data}=\frac1N\sum_n\frac{\|Y_{n,\mathcal M}-m_{n,\mathcal M}\|_2^2}{s_n^2},
\qquad
s_n=\max\left(\|Y_{n,\mathcal M}\|_2,q_{0.05},10^{-12}\right),
\]

trong đó \(q_{0.05}\) là phân vị 5% của chuẩn EEM trên train. Cách chia này giảm sự chi phối của mẫu có cường độ lớn.

Có thể diễn giải loss bằng giả định sai số Gaussian có phương sai tỷ lệ với \(s_n^2\). Tuy nhiên, \(s_n\) được tính từ dữ liệu; đây là diễn giải của weighted least squares, chưa phải một mô hình nhiễu đã được kiểm chứng.

Hàm mục tiêu đầy đủ:

\[
\mathcal L=\mathcal L_{data}
+\frac{\lambda_\mu}{2K}\sum_{k,d}\left(\frac{d_{kd}}{h_d}\right)^2
+\frac{\lambda_\rho}{K}\sum_k\rho_k^2,
\qquad (h_{Ex},h_{Em})=(15,25)\text{ nm}.
\]

Ở đây, \(d_{kd}\) là khoảng cách từ tâm đến cửa sổ tham chiếu theo trục \(d\), bằng zero khi tâm ở trong cửa sổ. Penalty vị trí cho phép tâm ra ngoài vùng nếu giảm sai số đủ lớn. Penalty \(\rho\) hạn chế độ nghiêng. “Prior mềm” chỉ các penalty này; mô hình không tính hậu nghiệm Bayesian.

### 3.4. Số component và liên hệ với peak quan sát

Cấu hình hiện tại đặt **K = 6 Gaussian**, tương ứng sáu cửa sổ B, T, M, A, C, UV ở phần 5.1, và thêm **một component nền**. Như vậy mỗi mẫu có 7 biên độ. Số 6 được đặt trước theo danh sách vùng tham chiếu trong mã, không được suy ra từ số cực đại của từng EEM và chưa được xác nhận là số thành phần tối ưu.

**Component là một bề mặt đóng góp; peak quan sát là cực đại của tổng các bề mặt.** Với Gaussian liên tục không bị cắt miền, tâm \(\mu_k\) là vị trí cực đại của thành phần. Trên lưới đo có mask, cực đại khả dụng có thể khác tâm.

\[
\widehat Y_n(z)=\sum_{k=1}^{6}a_{nk}\phi_k(z)+a_{n,bg}\phi_{bg}(z).
\]

Từ phép cộng này, số peak nhìn thấy không nhất thiết bằng số component:

- Hai Gaussian gần nhau có thể tạo thành **một peak rộng** hoặc một peak có vai.
- Component có biên độ nhỏ có thể bị thành phần khác che lấp; NNLS cũng có thể cho biên độ bằng zero.
- Một vùng phổ bất đối xứng hoặc đa đỉnh có thể cần nhiều Gaussian để xấp xỉ.
- Nền phẳng đóng góp cường độ nhưng không tạo peak riêng.

Ví dụ, hai thành phần B và T chồng lấp có thể tạo một cực đại nằm giữa hai tâm. Khi tỷ lệ biên độ B/T thay đổi, cực đại tổng có thể dịch chuyển dù cả hai tâm vẫn cố định. Vì vậy, vị trí peak của EEM không trực tiếp cho biết tâm hoặc số thành phần bên dưới.

Khi khởi tạo, mô hình chọn điểm mạnh trong từng cửa sổ và ghi nhận điểm đó có phải cực đại cục bộ hay không. **Một cửa sổ không có peak rõ vẫn được khởi tạo một Gaussian**; trạng thái này được lưu bằng `initial_local_max_supported`. Sau fit, cần xem đồng thời tâm, biên độ, bản đồ đóng góp và residual để đánh giá component có được dữ liệu hỗ trợ hay không. Nhãn B/T/M/A/C/UV vẫn là nhãn vùng, không phải danh tính hóa học đã xác nhận.

Muốn xác định số component phù hợp, cần so sánh các cấu hình số lượng/vùng trên cùng tập validation, xét sai số tái tạo và độ ổn định thành phần. Sáu Gaussian hiện tại là cấu hình thử nghiệm, không phải bằng chứng EEM chứa đúng sáu peak hoặc sáu chất.

### 3.5. Ước lượng và đặc trưng

Hình dạng và biên độ được tối ưu bằng Adam, kết hợp giải lại biên độ bằng NNLS. Bài toán chung không lồi. Khi bề mặt cố định, NNLS là bài toán lồi:

\[
\widehat a_n=\arg\min_{a\ge0}
\left\|Y_{n,\mathcal M}-\sum_{k\in\{1,\ldots,K,bg\}}a_k\phi_{k,\mathcal M}\right\|_2^2.
\]

Mẫu test dùng bề mặt và mask đã học; chỉ biên độ được suy ra từ EEM test.

| Đại lượng | Công thức | Ý nghĩa |
|---|---|---|
| Amplitude | \(a_{nk}\) | Tổng đóng góp của thành phần trên mask |
| Share | \(a_{nk}/A_n\) | Tỷ lệ trong tín hiệu tái tạo |
| Fmax | \(a_{nk}\max_{j\in\mathcal M}\phi_{kj}\) | Cực đại đóng góp của thành phần |
| Residual | \(Y_n-\widehat Y_n\) | Phần tín hiệu chưa được giải thích |

Amplitude và Fmax là đại lượng phổ, chưa phải nồng độ chất. BOD/COD/TOC không tham gia học các bề mặt.

## 4. Dữ liệu và cách chia nhóm

Dữ liệu gồm **760 EEM, 45 trạm**, mỗi EEM có kích thước **271 Em × 57 Ex**.

Trạm được gán vào nhóm cố định theo tên sông/kênh, lưu vực, loại thủy vực và vị trí địa lý. Các trạm thiếu căn cứ ghép được giữ riêng. Đây là phân nhóm theo metadata, không phải phân cụm EEM.

Ánh xạ có 19 nhóm. Benchmark giữ nhóm có ít nhất **8 mẫu và 8 tháng quan sát**, còn **16 nhóm, 752 mẫu**. Ba nhóm bị loại là Asan lake 3 và hai trạm Cheongpyeong dam1/dam3. Mỗi nhóm được fit một mô hình riêng; các trạm trong nhóm dùng chung bề mặt.

### Sáu nhóm có nhiều mẫu nhất

Số trong ngoặc là số mẫu từng trạm. Cả sáu nhóm đều có quan sát ở 12 tháng; không phải mọi trạm thành viên đều đủ 12 tháng.

| Nhóm | Số trạm | Số mẫu | Thành viên (số mẫu) | Căn cứ ghép |
|---|---:|---:|---|---|
| `han_corridor` | 10 | **152** | Dogok (12), Guri (11), Amsa (12), Jamsil (12), Ttukdo (9), Bogwang (12), Yeongdeungpo (11), Gayang (49), Kimpo (12), Paju (12) | Nhãn River/Han và dải tọa độ; chưa xác minh mọi điểm thuộc cùng dòng chính |
| `anseong` | 4 | **119** | Geonji stream (23), Cheonglyong stream (22), Jang-an stream (23), Anseong stream3 (51) | Nhóm Anseong mở rộng, gồm ba suối công nghiệp và một trạm sông |
| `sihwa` | 5 | **115** | Gunja stream (23), Okgu stream (23), Jeongwang stream (23), Shingil stream (23), Banwol stream (23) | Hệ tiếp nhận Sihwa theo tài liệu rà soát |
| `northern_industrial` | 4 | **90** | Sin stream-1 (23), Sin stream-2 (23), Pocheon stream (21), Yeongpyeong stream (23) | Loại thủy vực và vùng tọa độ; liên hệ cụ thể của Pocheon cần xác minh |
| `gyeongan` | 5 | **60** | Kyungan stream 2, 3, 5, 7, 11 (mỗi trạm 12) | Cùng tên sông và dải tọa độ liên tục |
| `hantan` | 3 | **36** | Hantangang2, 3, 4 (mỗi trạm 12) | Cùng tên sông và vị trí gần nhau |
| **Tổng** | **31** | **572** | **76.1% số mẫu benchmark** | |

Nguồn: [ánh xạ nhóm](../data/processed/station_groups.csv), [số mẫu](../data/processed/samples.csv), [căn cứ và giới hạn ghép nhóm](station_grouping_analysis.md). Cùng nhóm chưa đồng nghĩa cùng thành phần hóa học. Pipeline hiện tại chưa hiệu chỉnh riêng theo trạm.

## 5. Triển khai

### 5.1. Cấu hình mô hình

Sáu cửa sổ dùng để khởi tạo và đặt prior:

| Nhãn | Ex (nm) | Em (nm) |
|---|---:|---:|
| B | 270–280 | 305–320 |
| T | 275–285 | 320–350 |
| M | 290–315 | 370–420 |
| A | 230–265 | 380–480 |
| C | 320–360 | 420–480 |
| UV | 220–240 | 315–365 |

Các nhãn chỉ vùng giả thuyết, không xác nhận danh tính hóa học.

| Biến thể | \(\lambda_\mu\) | \(\rho\) | \(\lambda_\rho\) |
|---|---:|---|---:|
| `diag_no_prior` | 0 | 0 | 0 |
| `diag_soft_prior` | 0.002 | 0 | 0 |
| `rho_soft_prior` | 0.002 | \(|\rho|<0.3\) | 0.002 |

Cả ba dùng khởi tạo theo vùng. `diag_no_prior` chỉ tắt penalty vị trí.

Quy trình chính:

1. Tạo mask từ các ô có giá trị khác zero trong train; chưa phải mask tán xạ vật lý đã xác minh.
2. Khởi tạo tâm từ bản đồ trung bình có bootstrap và làm trơn. Phần âm chỉ bị cắt ở bước khởi tạo.
3. Ràng buộc tâm trong miền đo; \(\sigma_{Ex}\in(5,80)\) nm và \(\sigma_{Em}\in(5,100)\) nm.
4. Chạy Adam, giữ tâm và \(\rho\) cố định 50 bước đầu; giải lại NNLS mỗi 25 bước.
5. Chọn nghiệm có objective thấp nhất: full fit dùng 3 starts × 400 bước; mỗi temporal fold dùng 2 starts × 300 bước.

Trong tối ưu, bề mặt được chuẩn hóa L2. Sau fit, bề mặt được chuẩn hóa tổng bằng một và biên độ được suy lại từ EEM gốc. Đổi thang này không đổi dạng tín hiệu có thể tái tạo.

### 5.2. Đánh giá

Đánh giá thời gian dùng nửa đầu các tháng quan sát để train, chia nửa sau thành ba block test và mở rộng train. Với đủ 12 tháng, ba block test là **7–8, 9–10, 11–12**. Nhóm thiếu tháng dùng lịch tương ứng trong [folds.csv](../runs/diagnostics/v0_improved_groups/folds.csv).

Mask, bề mặt, scaler và hồi quy đều được fit từ train. Hồi quy dùng **7 amplitudes gồm nền → log1p → StandardScaler → Ridge, \(\alpha=10\)**.

Đầu vào test vẫn là EEM đo tại tháng test. Đây là ước lượng chất lượng nước từ EEM ở tháng mới, không phải dự báo khi chưa đo EEM.

## 6. Quan hệ toán học với PARAFAC

PARAFAC biểu diễn tensor EEM bằng

\[
Y_{nie}\approx\sum_{k=1}^{R}A_{nk}B_{ik}C_{ek},
\]

với \(A\) là score mẫu, \(B\) là phổ phát xạ và \(C\) là phổ kích thích. Mỗi thành phần là tích ngoài của hai phổ. Cơ sở ứng dụng cho EEM được trình bày trong [Murphy và cộng sự, 2013](https://doi.org/10.1039/C3AY41160E).

Với GMM diagonal:

\[
\rho_k=0\quad\Rightarrow\quad g_k(Ex,Em)=u_k(Ex)v_k(Em).
\]

Do đó, mỗi Gaussian diagonal là một bề mặt rank một với hai loading bị ép dạng Gaussian. Khi \(\rho_k\ne0\), bề mặt nhìn chung không tách được thành một tích ngoài.

| Thuộc tính | GMM | PARAFAC không âm |
|---|---|---|
| Thành phần | Gaussian 2D | Tích ngoài hai phổ 1D |
| Hình dạng | Trơn, đơn đỉnh; bản rho cho phép nghiêng | Loading không bị ép Gaussian, có thể đa đỉnh |
| Tham số hình dạng danh nghĩa | 4 hoặc 5/thành phần | \(271+57=328\)/thành phần trên lưới hiện tại |
| Thông tin vị trí | Khởi tạo và prior theo vùng | Không bắt buộc prior vị trí |
| Giới hạn chính | Gaussian có thể quá cứng | Chất lượng phụ thuộc rank, ràng buộc và độ ổn định nghiệm |

Số tham số khác nhau chưa đủ để suy ra phương pháp nào tốt hơn. Cả hai đều cần kiểm định trước khi diễn giải thành phần theo hóa học.

## 7. So sánh kết quả GMM–PARAFAC trên sáu nhóm lớn nhất

So sánh dùng **572 mẫu** thuộc Han, Anseong, Sihwa, phía bắc, Gyeongan và Hantan. Đã đối chiếu cùng sample IDs, thứ tự mẫu và số mẫu giữa hai phương pháp.

**GMM** trong các bảng là `diag_soft_prior`, với 6 Gaussian. PARAFAC dùng rank 3–4 theo kết quả đã lưu. Mỗi hồi quy dùng **một Fmax**, chọn component tốt nhất riêng cho từng nhóm, target và bộ hồi quy. Cả hai dùng KFold 5 phần, shuffle, seed 42; XGBoost có cùng cấu hình.

**R² train** đo độ khớp khi hồi quy được fit và chấm trên toàn bộ mẫu của nhóm; không phải trung bình R² train của các fold. **R² CV** là trung bình R² trên phần held-out của 5 folds. Hai điểm dùng cùng component đã chọn theo CV, không chọn lại theo train. Decomposition đã fit toàn bộ nhóm, nên CV này chỉ đánh giá bước hồi quy, chưa phải end-to-end hoặc ngoài thời gian.

### 7.1. Component có R² CV cao nhất trong từng trường hợp

Mỗi trường hợp là một **nhóm × target × bộ hồi quy**. GMM và PARAFAC được chọn component riêng theo R² CV cao nhất; bảng ghi R² train và R² CV của component đó. Không lấy trung bình giữa các nhóm hoặc component. R² CV vẫn là mean của 5 folds theo giao thức đã nêu, không phải điểm của fold tốt nhất.

#### Linear Regression

| Nhóm | Target | GMM train | GMM CV | PARAFAC train | PARAFAC CV |
| --- | --- | ---: | ---: | ---: | ---: |
| han_corridor | BOD | 0.163 | 0.032 | 0.068 | -0.015 |
| han_corridor | COD | 0.356 | 0.307 | 0.388 | 0.338 |
| han_corridor | TOC | 0.279 | 0.232 | 0.341 | 0.288 |
| anseong | BOD | 0.338 | 0.299 | 0.214 | 0.105 |
| anseong | COD | 0.301 | 0.330 | 0.126 | 0.109 |
| anseong | TOC | 0.322 | 0.293 | 0.247 | 0.226 |
| sihwa | BOD | 0.173 | 0.073 | 0.041 | -0.019 |
| sihwa | COD | 0.189 | 0.041 | 0.034 | -0.030 |
| sihwa | TOC | 0.204 | 0.167 | 0.061 | 0.034 |
| northern_industrial | BOD | 0.386 | 0.205 | 0.428 | 0.154 |
| northern_industrial | COD | 0.657 | 0.529 | 0.678 | 0.550 |
| northern_industrial | TOC | 0.675 | 0.585 | 0.710 | 0.625 |
| gyeongan | BOD | 0.082 | -0.133 | 0.130 | -0.100 |
| gyeongan | COD | 0.403 | 0.274 | 0.389 | 0.256 |
| gyeongan | TOC | 0.322 | 0.091 | 0.324 | 0.076 |
| hantan | BOD | 0.433 | 0.093 | 0.378 | 0.076 |
| hantan | COD | 0.589 | 0.439 | 0.487 | 0.385 |
| hantan | TOC | 0.347 | 0.105 | 0.262 | 0.023 |

#### XGBoost

| Nhóm | Target | GMM train | GMM CV | PARAFAC train | PARAFAC CV |
| --- | --- | ---: | ---: | ---: | ---: |
| han_corridor | BOD | 0.463 | 0.132 | 0.397 | 0.070 |
| han_corridor | COD | 0.611 | 0.368 | 0.547 | 0.315 |
| han_corridor | TOC | 0.526 | 0.287 | 0.518 | 0.297 |
| anseong | BOD | 0.582 | 0.310 | 0.465 | 0.040 |
| anseong | COD | 0.567 | 0.312 | 0.445 | 0.136 |
| anseong | TOC | 0.584 | 0.288 | 0.521 | 0.211 |
| sihwa | BOD | 0.531 | 0.245 | 0.614 | 0.335 |
| sihwa | COD | 0.585 | 0.253 | 0.676 | 0.363 |
| sihwa | TOC | 0.552 | 0.274 | 0.654 | 0.354 |
| northern_industrial | BOD | 0.709 | 0.272 | 0.668 | 0.138 |
| northern_industrial | COD | 0.762 | 0.515 | 0.770 | 0.514 |
| northern_industrial | TOC | 0.813 | 0.614 | 0.813 | 0.617 |
| gyeongan | BOD | 0.548 | -0.475 | 0.479 | -0.316 |
| gyeongan | COD | 0.594 | 0.130 | 0.590 | 0.150 |
| gyeongan | TOC | 0.535 | -0.056 | 0.531 | -0.035 |
| hantan | BOD | 0.761 | -0.018 | 0.732 | -0.066 |
| hantan | COD | 0.803 | 0.445 | 0.671 | 0.311 |
| hantan | TOC | 0.674 | -0.015 | 0.599 | -0.009 |

Tên component chỉ có ý nghĩa trong mô hình của từng nhóm; không đồng nhất GMM B/T với PARAFAC C1/C2.

Nguồn: GMM [Linear](../runs/diagnostics/v0_improved_groups/component_regression/group_best_component_linear_cv_r2.csv), [XGBoost](../runs/diagnostics/v0_improved_groups/component_regression/group_best_component_xgb_r2.csv); PARAFAC [Linear](../runs/diagnostics/parafac/by_group/group_best_component_linear_cv_r2.csv), [XGBoost](../runs/diagnostics/parafac/by_group/group_best_component_xgb_r2.csv).

### 7.2. Phạm vi kết luận

Kết quả chỉ so sánh khả năng hồi quy từ một thành phần phổ trong sáu nhóm trên. Component được chọn và báo điểm trên cùng CV; GMM có nhiều ứng viên hơn PARAFAC. Vì vậy, chênh lệch chưa chứng minh ưu thế dự đoán trên dữ liệu độc lập.


## 8. So sánh kết quả GMM–PARAFAC trên 10 trạm

Bộ kết quả này gồm **284 mẫu của 10 trạm**, fit một hệ component riêng cho từng trạm. Đây là benchmark theo trạm, khác mô hình fit chung theo nhóm ở phần 7; các mẫu có thể trùng với bộ nhóm nên hai bảng không phải hai tập kiểm định độc lập.

GMM dùng `diag_soft_prior`. Mỗi case là **trạm × target × bộ hồi quy**; chọn riêng component có R² CV cao nhất cho từng phương pháp và báo cả R² train, CV của chính component đó. R² train tính trên toàn bộ mẫu của trạm; R² CV là mean 5 folds, shuffle, seed 42. Không lấy trung bình qua trạm hoặc component.

Đã đối chiếu cùng sample IDs giữa hai phương pháp. Decomposition được fit toàn bộ trạm trước CV hồi quy; đây chưa phải end-to-end CV hoặc đánh giá ngoài thời gian.

### 8.1. Linear Regression

| Trạm | Số mẫu | Target | GMM train | GMM CV | PARAFAC train | PARAFAC CV |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Anseong stream3 | 51 | BOD | 0.201 | -0.007 | 0.336 | 0.028 |
| Anseong stream3 | 51 | COD | 0.634 | 0.566 | 0.362 | 0.336 |
| Anseong stream3 | 51 | TOC | 0.624 | 0.517 | 0.662 | 0.529 |
| Gayang | 49 | BOD | 0.102 | -0.369 | 0.178 | -0.247 |
| Gayang | 49 | COD | 0.381 | 0.115 | 0.190 | -0.039 |
| Gayang | 49 | TOC | 0.154 | -0.097 | 0.083 | -0.171 |
| Banwol stream | 23 | BOD | 0.903 | -0.273 | 0.920 | -0.195 |
| Banwol stream | 23 | COD | 0.857 | -0.293 | 0.920 | -0.172 |
| Banwol stream | 23 | TOC | 0.835 | -0.116 | 0.844 | -0.067 |
| Geonji stream | 23 | BOD | 0.269 | -0.246 | 0.294 | -0.197 |
| Geonji stream | 23 | COD | 0.441 | -0.088 | 0.400 | -0.218 |
| Geonji stream | 23 | TOC | 0.525 | -0.024 | 0.489 | -0.070 |
| Gunja stream | 23 | BOD | 0.425 | -0.953 | 0.333 | -1.732 |
| Gunja stream | 23 | COD | 0.579 | -0.237 | 0.567 | -0.460 |
| Gunja stream | 23 | TOC | 0.417 | -0.259 | 0.446 | -0.136 |
| Jang-an stream | 23 | BOD | 0.406 | -0.921 | 0.519 | -0.519 |
| Jang-an stream | 23 | COD | 0.185 | -0.541 | 0.019 | -0.661 |
| Jang-an stream | 23 | TOC | 0.077 | -0.203 | 0.359 | 0.181 |
| Je4ganseon-sulo | 23 | BOD | 0.563 | -0.422 | 0.218 | -1.213 |
| Je4ganseon-sulo | 23 | COD | 0.394 | -1.975 | 0.337 | -3.155 |
| Je4ganseon-sulo | 23 | TOC | 0.502 | -0.526 | 0.467 | -0.997 |
| Jeongwang stream | 23 | BOD | 0.248 | -0.215 | 0.235 | -0.324 |
| Jeongwang stream | 23 | COD | 0.375 | -0.807 | 0.230 | -1.002 |
| Jeongwang stream | 23 | TOC | 0.238 | 0.164 | 0.289 | 0.074 |
| Man-u stream | 23 | BOD | 0.021 | -2.375 | 0.000 | -2.436 |
| Man-u stream | 23 | COD | 0.301 | -0.201 | 0.175 | -0.255 |
| Man-u stream | 23 | TOC | 0.001 | -0.775 | 0.005 | -1.106 |
| Okgu stream | 23 | BOD | 0.247 | -0.077 | 0.259 | -0.054 |
| Okgu stream | 23 | COD | 0.029 | -0.065 | 0.071 | -0.046 |
| Okgu stream | 23 | TOC | 0.024 | -1.078 | 0.021 | -1.107 |

### 8.2. XGBoost

| Trạm | Số mẫu | Target | GMM train | GMM CV | PARAFAC train | PARAFAC CV |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Anseong stream3 | 51 | BOD | 0.674 | -0.191 | 0.702 | 0.028 |
| Anseong stream3 | 51 | COD | 0.825 | 0.520 | 0.822 | 0.491 |
| Anseong stream3 | 51 | TOC | 0.775 | 0.460 | 0.819 | 0.521 |
| Gayang | 49 | BOD | 0.493 | -0.801 | 0.653 | -0.147 |
| Gayang | 49 | COD | 0.647 | 0.044 | 0.540 | -0.021 |
| Gayang | 49 | TOC | 0.551 | -0.147 | 0.581 | -0.119 |
| Banwol stream | 23 | BOD | 0.949 | 0.264 | 0.960 | 0.439 |
| Banwol stream | 23 | COD | 0.933 | -0.064 | 0.950 | 0.479 |
| Banwol stream | 23 | TOC | 0.845 | 0.263 | 0.851 | 0.143 |
| Geonji stream | 23 | BOD | 0.763 | -0.197 | 0.644 | -0.283 |
| Geonji stream | 23 | COD | 0.697 | 0.194 | 0.768 | 0.181 |
| Geonji stream | 23 | TOC | 0.720 | 0.173 | 0.765 | 0.210 |
| Gunja stream | 23 | BOD | 0.626 | -1.385 | 0.474 | -1.678 |
| Gunja stream | 23 | COD | 0.762 | -1.342 | 0.690 | -1.283 |
| Gunja stream | 23 | TOC | 0.800 | -0.056 | 0.681 | -0.552 |
| Jang-an stream | 23 | BOD | 0.684 | -1.121 | 0.738 | -0.563 |
| Jang-an stream | 23 | COD | 0.555 | -0.534 | 0.352 | -1.124 |
| Jang-an stream | 23 | TOC | 0.524 | -0.355 | 0.709 | -0.326 |
| Je4ganseon-sulo | 23 | BOD | 0.751 | -0.214 | 0.495 | -1.823 |
| Je4ganseon-sulo | 23 | COD | 0.736 | -0.932 | 0.718 | -3.755 |
| Je4ganseon-sulo | 23 | TOC | 0.766 | 0.226 | 0.488 | -1.457 |
| Jeongwang stream | 23 | BOD | 0.609 | -0.428 | 0.648 | -0.725 |
| Jeongwang stream | 23 | COD | 0.743 | -0.132 | 0.593 | -2.017 |
| Jeongwang stream | 23 | TOC | 0.610 | 0.046 | 0.557 | -1.035 |
| Man-u stream | 23 | BOD | 0.771 | -1.242 | 0.530 | -2.093 |
| Man-u stream | 23 | COD | 0.572 | -0.391 | 0.710 | -1.027 |
| Man-u stream | 23 | TOC | 0.408 | -0.849 | 0.398 | -1.152 |
| Okgu stream | 23 | BOD | 0.602 | -0.206 | 0.628 | -0.365 |
| Okgu stream | 23 | COD | 0.685 | -0.013 | 0.432 | -0.474 |
| Okgu stream | 23 | TOC | 0.479 | -1.087 | 0.619 | -3.808 |

### 8.3. Phạm vi diễn giải

Tên component là nhãn riêng của từng lần fit, không bảo đảm cùng chất giữa các trạm hoặc phương pháp. Component được chọn trên cùng CV báo điểm; GMM có 6 ứng viên, PARAFAC có 3–4. Tám trạm chỉ có 23 mẫu, khiến mỗi test fold có khoảng 4–5 mẫu và R² dễ biến động.

Nguồn: [kết quả từng case](../runs/diagnostics/model_comparison/improved_v0_vs_v0_parafac/best_components_long.csv), [manifest đối chiếu](../runs/diagnostics/model_comparison/improved_v0_vs_v0_parafac/provenance.json). Số liệu lấy từ artifact đã lưu, không fit lại.
