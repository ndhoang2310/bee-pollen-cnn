# Bài tập: Phân loại ong mang phấn (Bee Pollen) bằng CNN

- Dataset: https://www.kaggle.com/datasets/ivanfel/honey-bee-pollen/data
- Notebook em chạy trên Kaggle (GPU, đã chạy xong): https://www.kaggle.com/code/ndhoang2310/bee-pollen-cnn-baseline-to-correction
- Code: thư mục `src/`, file nhãn sau khi sửa: `data/pollen_data_corrected.csv`

---

# Phần I

## 1. Download và tìm hiểu dữ liệu

Em tải bộ Honey Bee Pollen trên Kaggle về. Bộ này gồm 714 ảnh ong, mỗi ảnh đã được gán nhãn sẵn trong file `pollen_data.csv`:

- Nhãn 1 (mang phấn, tên file bắt đầu bằng `P`): 369 ảnh
- Nhãn 0 (không mang phấn, tên file bắt đầu bằng `NP`): 345 ảnh

Hai lớp gần như cân bằng nên em không cần xử lý mất cân bằng lớp. Em có kiểm tra thử thì nhãn trong file csv khớp hết với chữ cái đầu của tên file.

Quan sát một số ảnh thì em thấy ong mang phấn thường có 2 cục phấn màu vàng nhạt/trắng dính ở 2 chân sau, nhìn khá rõ. Còn ong không mang phấn thì chân sau đen, không có cục gì. Một số ảnh hơi mờ hoặc bị con ong khác che nên nhìn khó.

## 2. Chia train/val/test theo tỉ lệ 7:1:2

Em chia stratified (giữ tỉ lệ 2 lớp ở mỗi tập) với `seed=42`:

- train: 499 ảnh (P 258 / NP 241)
- val: 72 ảnh (P 37 / NP 35)
- test: 143 ảnh (P 74 / NP 69)

## 3. Tự định nghĩa mô hình CNN

Em tự viết một CNN nhỏ, input ảnh resize về 128x128:

```
Conv(3→32) - BN - ReLU - MaxPool
Conv(32→64) - BN - ReLU - MaxPool
Conv(64→128) - BN - ReLU - MaxPool
Conv(128→256) - BN - ReLU - AdaptiveAvgPool
FC(256→128) - ReLU - Dropout(0.5) - FC(128→2)
```

Tổng khoảng 422k tham số. Em cố tình làm model nhỏ thôi vì train chỉ có 499 ảnh, model to quá dễ overfit.

Cách train: Adam lr=1e-3, batch 32, tối đa 30 epochs, giảm lr khi val không tăng, early stopping theo val_acc (patience=7).

## 4. Đánh giá mô hình (baseline, chưa augmentation)

| Tập | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| train (499) | 0.9459 | 0.9358 | 0.9612 | 0.9484 | 0.9910 |
| val (72) | 0.9583 | 0.9722 | 0.9459 | 0.9589 | 0.9907 |
| test (143) | 0.9021 | 0.8659 | 0.9595 | 0.9103 | 0.9730 |

*Nhận xét:* model học được rồi (AUC ~0.97 trở lên), nhưng test mới 90.2%, chủ yếu sai ở chỗ đoán nhầm ong không phấn thành có phấn (11 ca FP). Em chạy lại vài lần thì điểm test lúc lên lúc xuống (có lần lên tới 96%), vì tập test chỉ có 143 ảnh, lệch vài ảnh là điểm nhảy nhiều. Val cũng chỉ có 72 ảnh nên early stopping hơi hên xui. Nói chung baseline này tạm được nhưng khả năng khái quát chưa vững, cần cải tiến ở Phần II.

---

# Phần II

## 1. Tìm cách improve performance

Em giữ nguyên kiến trúc, chỉ thêm augmentation lúc train (val/test giữ nguyên): lật ngang ngẫu nhiên, xoay ±15 độ, chỉnh sáng/tương phản nhẹ. Kết quả:

| Tập | Acc | F1 | AUC |
|---|---|---|---|
| train | 0.9539 | 0.9555 | 0.9899 |
| val | 0.9444 | 0.9444 | 0.9799 |
| test | 0.9301 | 0.9351 | 0.9757 |

*Nhận xét sau bước cải tiến:* test lên từ 90.21% → 93.01% (đúng thêm 4 ảnh, số ca báo động giả giảm từ 11 xuống 8), val thì gần như giữ nguyên (chênh đúng 1 ảnh). Theo em đây là cải tiến thật vì nó trúng vào chỗ yếu của baseline (FP nhiều), mà augmentation sinh thêm biến thể ảnh nên model bớt học vẹt. Tuy vậy mức tăng còn nhỏ, muốn lên nữa chắc phải dùng transfer learning hoặc ensemble, nhưng trong phạm vi bài này em dừng ở augmentation.

## 2. Chia thêm tập train-dev để kiểm tra mismatched distribution

Em cắt tập train 499 ra làm 2: `train_small` 399 ảnh để train, `train-dev` 100 ảnh để đo (cùng phân phối với train), rồi vẫn đánh giá trên val/test cũ. Kết quả:

| train_small (399) | train-dev (100) | val (72) | test (143) |
|---|---|---|---|
| 0.9649 | 0.9300 | 0.9583 | 0.9021 |

*Nhận xét:* theo lý thuyết, nếu bị mismatched distribution thì train-dev phải cao ngang train và cao hơn hẳn val/test. Nhưng ở đây train-dev (93.0%) xấp xỉ val (95.8%) và test (90.2%), cả ba đều thấp hơn train_small (96.5%). Tức là **không bị mismatch** — train/val/test cùng một phân phối (cũng dễ hiểu vì em chia ngẫu nhiên). Độ chênh vài % còn lại chỉ là overfit nhẹ + tập nhỏ. Nên em không cần xử lý gì thêm cho vấn đề này, cứ tập trung cải thiện model là được.

## 3. Correct lại label, chia lại 7:1:2 rồi đánh giá

Cách em làm: train model 3 lần với seed khác nhau, lọc ra những ảnh bị đoán sai cả 3 lần để soi lại bằng mắt (căn cứ vào cục phấn ở chân sau). Kết quả có 13 ảnh sai cả 3 lần, đa số là ong **có phấn nhưng phấn nhỏ** nên model bỏ sót — mấy ảnh này nhãn đúng, không được sửa. Chỉ có đúng 1 ảnh sai nhãn rõ ràng là `P6961-88r.jpg`: nhãn gốc ghi là có phấn nhưng nhìn kỹ không thấy cục phấn nào, cả 3 lần model đều chắc chắn nó là không phấn (p ≈ 0.00). Em sửa ảnh này từ 1 → 0, còn mấy ảnh mờ/bị che thì em giữ nguyên vì không đủ chắc chắn.

Sau khi sửa (P còn 368, NP thành 346), em chia lại 7:1:2 với cùng seed=42 rồi train lại y hệt:

| Tập | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| train (499) | 0.9639 | 0.9377 | 0.9961 | 0.9660 | 0.9984 |
| val (72) | 0.9722 | 0.9487 | 1.0000 | 0.9737 | 1.0000 |
| test (143) | 0.9650 | 0.9481 | 0.9865 | 0.9669 | 0.9924 |

*Nhận xét sau bước correct:* val tăng 95.83% → 97.22% (đúng thêm 1 ảnh — chính là ảnh `P6961` nằm ở tập val, trước sửa nó bị tính là lỗi, sau sửa thành đúng, recall lớp P lên 1.0). Test lên 96.5%.

## 4. So sánh 2 bộ dữ liệu (chưa correct vs correct) và nhận xét chung

| Tập | Chưa correct | Correct | Chênh lệch |
|---|---|---|---|
| train | 0.9459 | 0.9639 | +1.8% |
| val | 0.9583 | 0.9722 | +1.4% (1 ảnh) |
| test | 0.9021 | 0.9650 | +6.3% (9 ảnh) |

*Nhận xét:* nhìn bảng thì bộ correct thắng hết, nhưng em phải nói thật là mức tăng ở test không thể đổ hết cho việc sửa 1 cái nhãn, vì nhãn của tập test không hề thay đổi — tăng 9 ảnh ở test chủ yếu là do mỗi lần train ngẫu nhiên khác nhau (tập nhỏ nên điểm nhảy nhiều, như baseline em chạy lúc 90% lúc 96%). Cái chắc chắn quy được cho việc correct label chỉ có 1 ảnh ở val và recall lớp P cải thiện.

Kết luận của em: bộ dữ liệu gốc vốn đã khá sạch (soi hết 714 ảnh mà chỉ bắt được 1 lỗi chắc chắn), nên sửa nhãn không phải là hướng giúp tăng điểm nhiều. Vấn đề thật của bài này là model hay bỏ sót phấn nhỏ và hay đoán nhầm ảnh mờ/bị che. Nếu làm tiếp thì em sẽ thử transfer learning hoặc tăng kích thước ảnh thay vì ngồi sửa nhãn hàng loạt — vì em thử rồi, cứ thấy model đoán sai mà sửa theo model thì toàn sửa oan, nhãn người ta đúng mà mình sửa thành sai thì còn tệ hơn.
