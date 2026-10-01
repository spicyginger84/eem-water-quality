# Thông tin các trạm và hướng ghép nhóm

## Nhóm đã áp dụng trong processed data

Đã thêm `Group` vào `data/processed/samples.csv` và `samples.parquet`. Đây là phân nhóm làm việc theo metadata để train/dự đoán trong nhóm, chưa xác nhận đồng nhất sinh học. Phần đề xuất bên dưới được giữ làm căn cứ và ghi nhận các giới hạn.

| Group | Số trạm | Số mẫu |
|---|---:|---:|
| `anseong` | 4 | 119 |
| `ara` | 2 | 22 |
| `gyeongan` | 5 | 60 |
| `han_corridor` | 10 | 152 |
| `hantan` | 3 | 36 |
| `northern_industrial` | 4 | 90 |
| `sihwa` | 5 | 115 |
| 12 nhóm `single_*`, mỗi trạm một nhóm riêng | 12 | 166 |
| **Tổng** | **45** | **760** |

Chọn Anseong mở rộng gồm Geonji, Cheonglyong, Jang-an và Anseong stream3. Giữ `Point` để mô hình có thể hiệu chỉnh riêng từng trạm. Các nhãn ứng viên Han/phía bắc/Sihwa được dùng làm nhóm thử nghiệm; bỏ hậu tố candidate chỉ để tên ngắn, không thay đổi mức độ chắc chắn. Các trạm chưa rõ quan hệ và các hồ được gán nhóm riêng, không gộp vào một nhóm review chung.

Mapping chính thức cho bộ processed hiện tại: [station_groups.csv](../data/processed/station_groups.csv). Không thay đổi thứ tự mẫu, các cột cũ hoặc EEM. Cột Group là metadata; chưa tự động thay đổi cách train hay thêm feature vào mô hình hiện tại. Muốn train một nhóm, lọc Group trước rồi chia theo thời gian; không dùng GroupKFold theo Group để đánh giá mục tiêu dự đoán trong chính nhóm đó.

Sau khi tái tạo processed data, áp dụng lại mapping bằng:

```bash
.venv/bin/python scripts/data/assign_station_groups.py
```

Script kiểm tra CSV/Parquet khớp nhau, từ chối trạm không có mapping và kiểm tra lại các cột cũ sau khi ghi. Nếu bổ sung trạm mới, cần phân nhóm trạm đó trong mapping trước.

## Phạm vi dữ liệu

- 760 EEM, 45 trạm, quan trắc trong năm 2023.
- 34/45 trạm có đủ 12 tháng; 9 trạm có 6–11 tháng quan trắc (1 trạm có 6 tháng, 2 trạm có 9 tháng, 6 trạm có 11 tháng); 2 trạm chỉ có 1 mẫu (`Cheongpyeong dam1`, `Cheongpyeong dam3`).
- Hai trạm có nhiều mẫu bất thường so với phần còn lại: `Anseong stream3` (51) và `Gayang` (49). 17 trạm có 21–23 mẫu; 16 trạm có đúng 12 mẫu.
- Các tên trạm được nối với `locations.csv` sau khi chuẩn hóa chữ hoa/thường, khoảng trắng và ký tự phân cách. `Etc` và `Etc.` được xem là cùng nhãn nhóm phân tích, nhưng giá trị gốc vẫn được giữ trong cột `Watershed`.

## Nhóm metadata gốc — chỉ để kiểm kê, không phải nhóm huấn luyện đã xác nhận

| Nhóm phân tích | Số trạm | Tổng mẫu | Trung vị mẫu/trạm | Khoảng mẫu/trạm | Ý nghĩa |
|---|---:|---:|---:|---:|---|
| Han River / river | 14 | 197 | 12 | 9–49 | nhãn rộng, chứa cả Hantan/Imjin; không đồng nghĩa cùng dòng chính |
| Etc / industrial_stream | 9 | 194 | 23 | 11–23 | nhóm công nghiệp còn lại; nhãn lưu vực không xác định |
| Han River / industrial_stream | 6 | 136 | 23 | 21–23 | suối/kênh công nghiệp trong lưu vực Han; nên giữ riêng khỏi river |
| Anseong stream / industrial_stream | 3 | 68 | 23 | 22–23 | các suối công nghiệp gần Anseong; nhóm nhỏ nhưng hợp lý |
| Namhan River / river | 5 | 60 | 12 | 12–12 | các trạm Kyungan/Gyeongan; nên đặt tên theo sông cụ thể |
| Anseong stream / river | 1 | 51 | 51 | 51–51 | trạm sông Anseong; hiện chỉ một trạm nhưng nhiều mẫu |
| Etc / river | 2 | 22 | 11 | 11–11 | nhóm river không xác định lưu vực; nên dùng thận trọng |
| Etc / lake | 1 | 12 | 12 | 12–12 | một trạm, 12 mẫu; giữ riêng khỏi stream |
| Bukhan River / river | 1 | 12 | 12 | 12–12 | chỉ một trạm, không đủ để ước lượng hiệu ứng nhóm |
| Anseong stream / lake | 1 | 6 | 6 | 6–6 | một trạm, 6 mẫu; không nên gộp với river/industrial stream |
| Bukhan River / lake | 2 | 2 | 1 | 1–1 | hai trạm, mỗi trạm một mẫu; chỉ dùng metadata/kiểm tra, không fit riêng |

## Danh sách từng trạm

Bảng chi tiết gồm số mẫu, số ngày, số tháng, thời gian đầu/cuối, lưu vực, loại hình và nhóm phân tích được lưu tại [`docs/assets/station_inventory.csv`](assets/station_inventory.csv).

| Trạm | Mẫu | Tháng | Từ | Đến | Lưu vực | Loại | Nhóm phân tích |
|---|---:|---:|---:|---:|---|---|---|
| Cheonglyong stream | 22 | 12 | 2023.01.02 | 2023.12.19 | Anseong stream | Industrial stream | Anseong stream / industrial_stream |
| Geonji stream | 23 | 12 | 2023.01.02 | 2023.12.19 | Anseong stream | Industrial stream | Anseong stream / industrial_stream |
| Jang-an stream | 23 | 12 | 2023.01.02 | 2023.12.19 | Anseong stream | Industrial stream | Anseong stream / industrial_stream |
| Asan lake 3 | 6 | 6 | 2023.03.20 | 2023.11.29 | Anseong stream | Lake | Anseong stream / lake |
| Anseong stream3 | 51 | 12 | 2023.01.02 | 2023.12.26 | Anseong stream | River | Anseong stream / river |
| Cheongpyeong dam1 | 1 | 1 | 2023.02.13 | 2023.02.13 | Bukhan River | Lake | Bukhan River / lake |
| Cheongpyeong dam3 | 1 | 1 | 2023.02.13 | 2023.02.13 | Bukhan River | Lake | Bukhan River / lake |
| Nam-i island | 12 | 12 | 2023.01.09 | 2023.12.06 | Bukhan River | River | Bukhan River / river |
| Banwol stream | 23 | 12 | 2023.01.02 | 2023.12.20 | Etc | Industrial stream | Etc / industrial_stream |
| Gumun stream | 22 | 12 | 2023.01.02 | 2023.12.20 | Etc | Industrial stream | Etc / industrial_stream |
| Gunja stream | 23 | 12 | 2023.01.04 | 2023.12.20 | Etc | Industrial stream | Etc / industrial_stream |
| Je4ganseon-sulo | 23 | 12 | 2023.01.04 | 2023.12.20 | Etc | Industrial stream | Etc / industrial_stream |
| Jeongwang stream | 23 | 12 | 2023.01.04 | 2023.12.20 | Etc | Industrial stream | Etc / industrial_stream |
| Okgu stream | 23 | 12 | 2023.01.04 | 2023.12.20 | Etc | Industrial stream | Etc / industrial_stream |
| Shingil stream | 23 | 12 | 2023.01.04 | 2023.12.20 | Etc | Industrial stream | Etc / industrial_stream |
| Wonjeongli-sulo | 23 | 12 | 2023.01.02 | 2023.12.20 | Etc | Industrial stream | Etc / industrial_stream |
| seonhaeng stream | 11 | 11 | 2023.02.20 | 2023.12.05 | Etc. | Industrial stream | Etc / industrial_stream |
| Nam-yang lake 3 | 12 | 12 | 2023.01.02 | 2023.12.04 | Etc | Lake | Etc / lake |
| Ara stream 1 | 11 | 11 | 2023.02.07 | 2023.12.05 | Etc. | River | Etc / river |
| Ara stream 2 | 11 | 11 | 2023.02.07 | 2023.12.05 | Etc. | River | Etc / river |
| Man-u stream | 23 | 12 | 2023.01.03 | 2023.12.18 | Han River | Industrial stream | Han River / industrial_stream |
| Pocheon stream | 21 | 11 | 2023.02.08 | 2023.12.18 | Han River | Industrial stream | Han River / industrial_stream |
| Seung-gi stream | 23 | 12 | 2023.01.04 | 2023.12.20 | Han River | Industrial stream | Han River / industrial_stream |
| Sin stream-1 | 23 | 12 | 2023.01.03 | 2023.12.18 | Han River | Industrial stream | Han River / industrial_stream |
| Sin stream-2 | 23 | 12 | 2023.01.03 | 2023.12.18 | Han River | Industrial stream | Han River / industrial_stream |
| Yeongpyeong stream | 23 | 12 | 2023.01.03 | 2023.12.18 | Han River | Industrial stream | Han River / industrial_stream |
| Amsa | 12 | 12 | 2023.01.10 | 2023.12.12 | Han River | River | Han River / river |
| Bogwang | 12 | 12 | 2023.01.10 | 2023.12.12 | Han River | River | Han River / river |
| Dogok | 12 | 12 | 2023.01.10 | 2023.12.12 | Han River | River | Han River / river |
| Gayang | 49 | 12 | 2023.01.10 | 2023.12.26 | Han River | River | Han River / river |
| Guri | 11 | 11 | 2023.02.14 | 2023.12.12 | Han River | River | Han River / river |
| Hantangang2 | 12 | 12 | 2023.01.19 | 2023.12.11 | Han River | River | Han River / river |
| Hantangang3 | 12 | 12 | 2023.01.19 | 2023.12.11 | Han River | River | Han River / river |
| Hantangang4 | 12 | 12 | 2023.01.19 | 2023.12.11 | Han River | River | Han River / river |
| Imjin stream 2 | 9 | 9 | 2023.04.25 | 2023.12.11 | Han River | River | Han River / river |
| Jamsil | 12 | 12 | 2023.01.10 | 2023.12.12 | Han River | River | Han River / river |
| Kimpo | 12 | 12 | 2023.01.10 | 2023.12.12 | Han River | River | Han River / river |
| Paju | 12 | 12 | 2023.01.09 | 2023.12.13 | Han River | River | Han River / river |
| Ttukdo | 9 | 9 | 2023.03.16 | 2023.11.08 | Han River | River | Han River / river |
| Yeongdeungpo | 11 | 11 | 2023.01.10 | 2023.11.08 | Han River | River | Han River / river |
| Kyungan stream 2 | 12 | 12 | 2023.01.18 | 2023.12.13 | Namhan River | River | Namhan River / river |
| Kyungan stream 3 | 12 | 12 | 2023.01.18 | 2023.12.13 | Namhan River | River | Namhan River / river |
| Kyungan stream 5 | 12 | 12 | 2023.01.18 | 2023.12.13 | Namhan River | River | Namhan River / river |
| Kyungan stream 7 | 12 | 12 | 2023.01.18 | 2023.12.13 | Namhan River | River | Namhan River / river |
| Kyungan stream11 | 12 | 12 | 2023.01.18 | 2023.12.13 | Namhan River | River | Namhan River / river |

## Rà soát và đề xuất cập nhật ngày 2026-10-01

Mục tiêu là ước lượng chỉ số từ EEM tại lần quan trắc sau ở chính trạm/nhóm trạm đã có lịch sử. Các nhóm dưới đây là giả thuyết để đánh giá theo thời gian, chưa phải các nhóm sinh học được xác nhận. Chưa huấn luyện mô hình hoặc chọn nhóm theo kết quả target.

### Những điểm cần sửa khi diễn giải metadata

- `Etc`/`Etc.` là nhãn còn lại, không phải một lưu vực. Không gộp 9 industrial streams chỉ vì cùng nhãn này.
- `Han River / river` gồm cả Hantangang và Imjin; không thể gọi cả 14 trạm là một đoạn sông chính đồng nhất.
- Năm trạm Kyungan thuộc cùng tên sông Gyeongan (경안천). Gyeongan chảy vào hồ Paldang; không nên diễn giải nhãn `Namhan River` trong file thành năm điểm trên dòng chính Namhan. [Nguồn Academy of Korean Studies](https://encykorea.aks.ac.kr/Article/E0002645).
- `Seung-gi stream` và `Je4ganseon-sulo` có tọa độ trùng hoàn toàn: 37°23′47.9″N, 126°40′23.6″E. Đây là dấu hiệu cần đối chiếu file gốc; chưa biết dòng nào sai, không tự sửa hoặc coi hai trạm là một.
- Seung-gi được cơ quan Incheon mô tả là sông qua Namdong/Yeonsu. Vì vậy nhãn `Han River` chưa đủ căn cứ để ghép nó với Sin/Pocheon/Yeongpyeong. [Nguồn Hội đồng Incheon](https://www.icouncil.go.kr/webzine/webzineno/webzineNoDetail.do?menu_no=004&menumain=04&no_seq=1351&pgno=8).
- Mã `Lebel=Ha14` bị trùng giữa Guri và Hantangang 3. Ghép bảng bằng tên chuẩn hóa có kiểm tra tính duy nhất, không dùng mã này như khóa duy nhất.
- File locations có 50 trạm nhưng samples chỉ có 45: Kyungan stream 9, Asan lake 1/2, Imjin stream 7 và Cheongpyeong dam 2 chưa có mẫu trong bộ processed. Không tính chúng vào số mẫu huấn luyện.
- Nguồn ngoài xác nhận đặc điểm hệ sông, chưa xác nhận độc lập toàn bộ 45 tọa độ lấy mẫu hoặc thành phần sinh học năm 2023. Giữ nguyên metadata gốc; đề xuất nằm trong bảng riêng.

### Các nhóm nên thử

Các số train/validation/test tính trực tiếp từ samples.csv, lần lượt tháng 1–6 / 7–8 / 9–12. Đây là một cấu hình minh họa; khi chọn kiến trúc chỉ xem validation.

| Nhóm | Thành viên | Tổng mẫu | Train / Val / Test | Cơ sở và giới hạn |
|---|---|---:|---:|---|
| Hành lang Han — ứng viên | Dogok, Guri, Amsa, Jamsil, Ttukdo, Bogwang, Yeongdeungpo, Gayang, Kimpo, Paju | 152 | 75 / 27 / 50 | Cùng nhãn River/Han và dải tọa độ; chưa xác minh mọi điểm nằm trên cùng dòng chính. Cần kiểm tra khác biệt thượng/hạ lưu và vị trí Paju trước khi khẳng định đồng nhất. |
| Gyeongan/Kyungan | Kyungan stream 2, 3, 5, 7, 11 | 60 | 30 / 10 / 20 | Cùng tên sông, dải tọa độ liên tục; ưu tiên cao về cơ sở nhóm nhưng ít mẫu cho CNN. |
| Hantan | Hantangang2, 3, 4 | 36 | 18 / 6 / 12 | Cùng tên sông và gần nhau; không tự thêm Imjin chỉ vì nhãn Han. |
| Suối công nghiệp phía bắc — ứng viên | Sin stream-1, Sin stream-2, Pocheon stream, Yeongpyeong stream | 90 | 46 / 12 / 32 | Cùng loại và vùng tọa độ. Có cơ sở hệ Hantan cho Sin/Yeongpyeong; quan hệ cụ thể của Pocheon và ảnh hưởng nguồn thải cần xác minh. |
| Sihwa — ứng viên | Gunja stream, Okgu stream, Jeongwang stream, Shingil stream, Banwol stream | 115 | 60 / 15 / 40 | Nguồn nghiên cứu nhận diện các sông vào Sihwa; cùng hệ tiếp nhận không bảo đảm cùng nguồn DOM. |
| Suối công nghiệp Anseong | Geonji stream, Cheonglyong stream, Jang-an stream | 68 | 36 / 9 / 23 | Cùng lưu vực/loại theo metadata và gần nhau; nhóm ban đầu riêng với dòng sông Anseong. |
| Anseong riêng | Anseong stream3 | 51 | 25 / 9 / 17 | Đối chứng mô hình riêng trạm. Có thể thử thêm vào nhóm Anseong ở trên thành 119 mẫu (61 / 18 / 40), như một thí nghiệm mở rộng có hiệu chỉnh trạm/loại thủy vực. |
| Ara | Ara stream 1, Ara stream 2 | 22 | 10 / 4 / 8 | Cùng tên kênh; giữ riêng vì vận hành kênh có thể khác sông. Số mẫu quá ít để ưu tiên CNN. |

Nguồn cho hệ phía bắc: [Thông báo quy hoạch Hantan/Sin, có mô tả Yeongpyeong hợp lưu với Hantan](https://www.mcee.go.kr/hg/web/board/read.do?boardId=1870600&boardMasterId=342&menuId=1256). Thông tin này hỗ trợ prior địa lý, không xác nhận tương đồng sinh học tại các điểm trong dataset.

Nguồn Sihwa: [Báo cáo nghiên cứu tại Ansan Green Environment Center](https://www.agec.or.kr/data/study/5cde13896847d.pdf) phân biệt Okgu/Gunja/Jeongwang và kênh số 4 là các dòng nhân tạo, Shingil/Banwol là dòng tự nhiên. Vì vậy có thể thử nhóm lõi Gunja–Okgu–Jeongwang (69 mẫu) trước, sau đó mở rộng lên 115 mẫu. Je4ganseon-sulo là ứng viên bổ sung theo tên nhưng tạm để ngoài vì tọa độ trùng Seung-gi; nếu xác minh đúng, nhóm mở rộng có 138 mẫu.

Nguồn Ara: [K-water](https://www.kwater.or.kr/giwaterway/images/giwaterway/download/leaflet_2021.pdf) mô tả tuyến nối biển Tây với Han. Không suy ra độ mặn từng mẫu chỉ từ vị trí này.

### Các trạm chưa ghép

12 trạm còn lại có tổng 166 mẫu. `review_individual` trong CSV là trạng thái chờ rà soát, **không phải một nhóm để train chung**.

| Trạm | Mẫu | Xử lý đề xuất |
|---|---:|---|
| Je4ganseon-sulo | 23 | Xác minh tọa độ; ứng viên nhóm Sihwa sau khi đối chiếu. |
| Seung-gi stream | 23 | Giữ riêng; xác minh nhãn lưu vực và tọa độ. |
| Gumun stream | 22 | Chưa có bằng chứng cùng hệ sông với Sihwa; không gộp theo Etc. |
| Wonjeongli-sulo | 23 | Giữ riêng, kiểm tra cửa xả/hệ kênh trước khi nối với nhóm khác. |
| Man-u stream | 23 | Không tự ghép với nhóm công nghiệp phía bắc chỉ do nhãn Han. |
| seonhaeng stream | 11 | Tọa độ tách khỏi cụm Sihwa; chưa đủ căn cứ ghép. |
| Imjin stream 2 | 9 | Có thể xét nhóm mở rộng Hantan–Imjin sau khi kiểm tra kết nối/vị trí và validation; chưa đưa vào nhóm lõi. |
| Nam-i island | 12 | Giữ riêng theo Bukhan/River; không ghép với hồ chỉ vì cùng Bukhan. |
| Nam-yang lake 3 | 12 | Hồ riêng; thiếu cơ sở ghép với Asan chỉ theo loại Lake. |
| Asan lake 3 | 6 | Hồ riêng, rất ít mẫu. |
| Cheongpyeong dam1, dam3 | 1 mỗi trạm | Không đủ lịch sử cho đánh giá dự đoán theo thời gian; giữ để kiểm kê. |

### Cách quyết định giữ hay tách nhóm

1. Cố định nhóm metadata trước; không dùng target test để chọn thành viên. Nhóm theo EEM cần fit PCA/scaler và tính độ tương đồng chỉ trên train của mỗi cửa sổ thời gian.
2. Kiểm tra đồng thời hình dạng EEM đã chuẩn hóa theo mẫu (chỉ phục vụ so hình dạng), cường độ tuyệt đối và EC/pH nếu có. Giống hình dạng không đồng nghĩa giống nồng độ hoặc cùng sinh học. Chưa thực hiện phân tích này trong lần rà soát metadata.
3. So sánh mô hình riêng trạm với mô hình chung nhóm và mô hình chung có hiệu chỉnh trạm. Bắt đầu bằng Ridge/SVR với ít đặc trưng; CNN nhỏ là nhánh thử nghiệm vì ngay cả nhóm lớn nhất chỉ có 75 mẫu train trong ví dụ.
4. Với CNN, dùng encoder chung trong nhóm + intercept trạm có regularization. Nếu pooling làm MAE validation xấu đi ở một trạm, kiểm tra nhóm và mức co hiệu ứng trạm; không liên tục sửa nhóm theo test.
5. Cân bằng đóng góp theo trạm trong loss hoặc lấy mẫu theo trạm. Gayang chiếm 49/152 mẫu nhóm Han; báo cáo MAE trung bình đều qua trạm bên cạnh pooled MAE. Lịch đo đồng thời giữa các trạm làm số mẫu độc lập hữu hiệu nhỏ hơn số dòng.
6. Dùng chia tiến về thời gian với cùng mốc ngày cho mọi trạm. LOMO có thể dùng dữ liệu tương lai để dự đoán quá khứ, nên chỉ là đối chứng nội suy, không phải đánh giá chính cho mục tiêu này. Lịch sử target và intercept trạm chỉ được tính từ dữ liệu có trước thời điểm dự đoán.
7. Báo cáo MAE/RMSE từng trạm, pooled R² và cải thiện so với trung bình lịch sử/giá trị gần nhất của trạm; không chọn nhóm chỉ bằng pooled R². Với một năm dữ liệu, chưa thể xác nhận độ ổn định qua năm.

### Thứ tự thực hiện

- Ưu tiên kiểm tra nhóm **Gyeongan** để có một nhóm cùng sông rõ ràng, dù mô hình khởi đầu nên đơn giản.
- Ưu tiên thử CNN trên **Sihwa 115 mẫu** và **hành lang Han 152 mẫu**, sau khi kiểm tra điều kiện thành viên nêu trên; chưa coi đây là cỡ mẫu đủ bảo đảm CNN tốt.
- Với câu hỏi cụ thể về **Anseong stream3**, so sánh mô hình riêng 51 mẫu với nhóm Anseong mở rộng 119 mẫu, giữ hiệu ứng riêng từng trạm và kiểm tra MAE riêng Anseong stream3.
- Không gộp thêm các hồ hay các trạm Etc chưa rõ chỉ để đạt một ngưỡng số mẫu.

Danh sách máy đọc được: [station_group_proposals.csv](assets/station_group_proposals.csv). Bảng bao phủ đúng 45 trạm, mỗi trạm một dòng, tổng 760 mẫu; không thay đổi locations.csv hoặc các bảng kiểm kê gốc.

## Benchmark notebook theo Group

Các notebook hiện giữ lại là `parafac.ipynb` và `improved_v0_10stations.ipynb`, dùng `Group` cho benchmark phân rã/hồi quy. Notebook V0 cũ và EDA cũ không còn trong workflow hiện tại. Giữ tên notebook cũ để các liên kết không bị đứt. Mỗi group dùng toàn bộ mẫu thuộc các trạm thành viên để fit chung; cột Point gốc vẫn giữ trong processed data.

- PARAFAC: bộ kết quả hiện tại gồm sáu nhóm lớn nhất, có ít nhất 30 mẫu. Xem cấu hình đã lưu tại `runs/diagnostics/parafac/by_group/config.json`.
- V0 cải tiến: thêm điều kiện ít nhất 8 tháng quan sát; ghi đầy đủ điều kiện vào `group_eligibility.csv`. Fit mask/shape/hồi quy từ train trong từng group. Nhóm không đủ 12 tháng dùng nửa đầu số tháng quan sát làm train ban đầu, nửa sau chia ba block; lịch test có thể khác giữa nhóm.
- KFold hồi quy trên decomposition full-group trong các notebook phân rã vẫn là diagnostic, không phải end-to-end temporal evaluation. Đổi sang Group không loại bỏ hạn chế này.
- Output hiện tại: `parafac/by_group` và `v0_improved_groups` dưới `runs/diagnostics/`. Historical V0 của group chỉ được dùng khi có artifact với đúng các sample_index.

Output đã lưu trong notebook được xóa để tránh hiển thị kết quả theo trạm cũ như kết quả group. Chạy Run All để tạo benchmark đầy đủ; kiểm tra chạy thử giảm ngân sách không được xem là điểm benchmark chính thức.
