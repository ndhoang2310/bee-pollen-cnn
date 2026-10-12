# Bee Pollen (Ong mang phấn) — CNN: Baseline → Cải tiến → Chẩn đoán mismatch → Correct label

Dataset: https://www.kaggle.com/datasets/ivanfel/honey-bee-pollen (CC-BY-SA-4.0).
Notebook Kaggle (đã chạy, GPU): https://www.kaggle.com/code/ndhoang2310/bee-pollen-cnn-baseline-to-correction (COMPLETE).
Code trong `src/` (`dataset.py`, `model.py`, `train_utils.py`, `run_part1.py`, `run_part2.py`).
File nhãn đã correct: `data/pollen_data_corrected.csv`.
**Số liệu dưới đây là kết quả chạy chính thức trên Kaggle (GPU).**

## 1. Dữ liệu

- 714 ảnh JPG RGB, kích thước gốc ~180×300 (W×H), 2 lớp:
  - `P*` = pollen-bearing (mang phấn): 369 ảnh (51.7%)
  - `NP*` = non-pollen: 345 ảnh (48.3%)
- Nhãn trong `pollen_data.csv` khớp 100% với tiền tố tên file (0 mismatch filename-vs-CSV).
- Dấu hiệu nhận biết khi review thủ công: ong mang phấn có **khối phấn màu vàng nhạt/trắng rõ rệt ở 2 chân sau (corbiculae)** — xem ảnh mẫu `P10057-125r`, `P57380-120r`, `P63866-224r`. Không có khối này = NP.
- Chia stratified theo tỉ lệ 7:1:2, `seed=42`:
  - train 499 (P 258 / NP 241), val 72 (P 37 / NP 35), test 143 (P 74 / NP 69).

## 2. Phần I — Mô hình CNN tự định nghĩa + training

### Kiến trúc `BaselineCNN` (~422k params, input 3×128×128)

```
Conv(3→32,3×3,p1) - BN - ReLU - MaxPool(2)      # 128→64
Conv(32→64,3×3,p1) - BN - ReLU - MaxPool(2)     # 64→32
Conv(64→128,3×3,p1) - BN - ReLU - MaxPool(2)    # 32→16
Conv(128→256,3×3,p1) - BN - ReLU - AdaptiveAvgPool(1)
Flatten - FC(256→128) - ReLU - Dropout(0.5) - FC(128→2)
```

Huấn luyện: Adam `lr=1e-3`, `weight_decay=1e-4`, `CrossEntropyLoss`, batch 32,
`ReduceLROnPlateau(mode=max, factor=0.5, patience=3)`, EarlyStopping theo `val_acc` (`patience=7`, tối đa 30 epochs),
resize 128×128 + chuẩn hóa mean/std ImageNet. Chạy GPU Kaggle.

### Kết quả Phần I (baseline, không augmentation) — Kaggle GPU

| Tập | Acc | Prec | Rec | F1 | AUC | Confusion matrix [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|---|
| train (499) | 0.9459 | 0.9358 | 0.9612 | 0.9484 | 0.9910 | [[224,17],[10,248]] |
| val (72) | 0.9583 | 0.9722 | 0.9459 | 0.9589 | 0.9907 | [[34,1],[2,35]] |
| test (143) | **0.9021** | 0.8659 | 0.9595 | 0.9103 | 0.9730 | [[58,11],[3,71]] |

Nhận xét: baseline học được đặc trưng phấn (AUC ~0.97–0.99) nhưng còn báo động giả trên NP (11 FP ở test).
Điểm test dao động mạnh giữa các lần chạy (bản chạy local CPU đạt 96.5%) — do tập nhỏ (mỗi ảnh test ≈ 0.7%)
cộng EarlyStopping theo val nhỏ (72 mẫu). Đây chính là lý do Phần II cần cải tiến khả năng khái quát.

Chạy lại: `python3 src/run_part1.py --epochs 30 --img 128 --seed 42`

## 3. Phần II — Cải tiến, train-dev, correct label

### 3.1. Cải tiến (augmentation + regularization) — Kaggle GPU

Giữ nguyên kiến trúc, chỉ thêm augmentation lúc train:
`RandomHorizontalFlip(0.5) + RandomRotation(15°) + ColorJitter(0.2)` (val/test giữ nguyên).
Kết quả:

| Tập | Acc | F1 | AUC | CM |
|---|---|---|---|---|
| train | 0.9539 | 0.9555 | 0.9899 | [[229,12],[11,247]] |
| val | 0.9444 | 0.9444 | 0.9799 | [[34,1],[3,34]] |
| test | **0.9301** | 0.9351 | 0.9757 | [[61,8],[2,72]] |

Nhận xét từng bước:
- Test tăng **90.21% → 93.01%** (+4 ảnh đúng, FP giảm 11 → 8), val tương đương (95.83% → 94.44%, chênh 1 ảnh).
- Kết luận: augmentation **cải thiện khái quát** (đúng vai trò của nó trên tập nhỏ), dù mức tăng tuyệt đối
  vẫn nằm trong biên độ nhiễu — cần trung bình nhiều seeds để khẳng định chặt chẽ hơn.
  Hướng tiếp theo: transfer learning (ResNet18/MobileNet), ảnh 192px, TTA, ensemble 3 seeds.

Chạy: `python3 src/run_part2.py --step aug`

### 3.2. Tập train-dev — kiểm tra mismatched distribution — Kaggle GPU

Chia train 499 → `train_small` 399 + `train-dev` 100 (stratified, cùng phân phối với train),
train **chỉ trên 399**, đánh giá trên 4 tập:

| train_small (399) | train-dev (100) | val (72) | test (143) |
|---|---|---|---|
| 0.9649, F1 0.9670, [[180,13],[1,205]] | 0.9300, F1 0.9358, [[42,6],[1,51]] | 0.9583, F1 0.9589 | 0.9021, F1 0.9114 |

Luận giải theo phương pháp train-dev (Ng):
- Nếu **mismatch**: train-dev ≈ train ≫ val/test.
- Thực tế: train-dev (93.0%) ≈ val (95.8%) ≈ test (90.2%) < train_small (96.5%).
- Kết luận: **không có bằng chứng mismatched distribution** trên cách chia ngẫu nhiên;
  khoảng chênh là **overfit/phương sai do tập nhỏ** (bớt 100 mẫu train làm test rớt vài %),
  không phải do train/test khác phân phối. Vì vậy không cần biện pháp xử lý mismatch
  (thu thêm dữ liệu theo phân phối test, reweighting…); ưu tiên tăng dữ liệu/augmentation.

Chạy: `python3 src/run_part2.py --step train_dev`

### 3.3. Audit + correct label — Kaggle GPU

Quy trình (tránh confirmation bias — không flip hàng loạt theo model):
1. Train 3 seeds (42/7/123), lấy mẫu **sai ổn định ≥2/3 seeds** làm ứng viên.
   Chạy: `python3 src/run_part2.py --step audit`
2. Kết quả audit trên Kaggle: 13 mẫu sai 3/3 — hầu hết là **P bị bỏ sót** (`P6961, P63199, P12063, P21252,
   P11440, P21577, P11762, P11180, P13151, P9995, P14163`) + 2 NP (`NP27452, NP13576`, ảnh mờ/che khuất).
   Đối chiếu thủ công với định nghĩa "khối phấn ở chân sau":
   - Chỉ **1 mẫu sai nhãn rõ rệt**: `P6961-88r.jpg` — nhãn gốc P nhưng không có khối phấn,
     model 3/3 seeds đều tin cậy NP (p≈0.00) → flip **1→0**.
   - Các P bị bỏ sót còn lại **đúng là P** (phấn nhỏ, model chưa đủ nhạy — vấn đề recall, không phải nhãn sai).
     Các NP mờ/che khuất **giữ nguyên nhãn** (không đủ bằng chứng sửa).
3. Lưu `data/pollen_data_corrected.csv` (P 369→368, NP 345→346 — cũng là file output của notebook Kaggle),
   chia lại 7:1:2 cùng `seed=42`, train lại cùng cấu hình.

### 3.4. So sánh chưa-correct vs correct (cùng seed chia) — Kaggle GPU

| Tập | Chưa correct (gốc) | Correct (1 nhãn) |
|---|---|---|
| train (499) | acc 0.9459, F1 0.9484, [[224,17],[10,248]] | acc 0.9639, F1 0.9660, [[225,17],[1,256]] |
| val (72) | acc 0.9583, F1 0.9589, [[34,1],[2,35]] | acc **0.9722**, F1 0.9737, [[33,2],[0,37]] |
| test (143, nhãn không đổi) | acc 0.9021, F1 0.9103, [[58,11],[3,71]] | acc **0.9650**, F1 0.9669, [[65,4],[1,73]] |

Nhận xét trung thực:
- Val tăng 1 mẫu (mẫu `P6961-88r` nằm ở val: từ "lỗi" thành "đúng" sau khi sửa — đúng như kỳ vọng của correct label),
  recall lớp P lên 1.000, AUC val = 1.000.
- Test tăng 90.21% → 96.50% (+9 ảnh) **không thể quy hết cho 1 nhãn được sửa** (nhãn test không đổi):
  phần lớn là **phương sai huấn luyện** (khởi tạo/shuffle/EarlyStopping trên tập nhỏ).
  Kết luận đúng đắn là: bộ dữ liệu gốc **vốn đã sạch** (~1/714 lỗi xác thực được),
  nút thắt là **recall phấn nhỏ + mẫu khó/che khuất**, không phải nhiễu nhãn.
  Để so sánh chặt chẽ cần trung bình nhiều seeds; không flip hàng loạt theo model (sẽ bơm nhiễu vào dữ liệu).

Chạy: `python3 src/run_part2.py --step corrected`

## 4. Kết luận chung

1. Baseline CNN tự định nghĩa (~422k params) đạt **90–96% test** tùy lần chạy (phương sai lớn do tập nhỏ) — học được đặc trưng phấn (AUC ~0.97–0.99).
2. Augmentation cải thiện test **90.2% → 93.0%** (giảm FP), đúng vai trò khái quát trên tập nhỏ.
3. Thí nghiệm train-dev **bác bỏ giả thuyết mismatched distribution** (train-dev ≈ val ≈ test).
4. Audit 3-seed + review thủ công chỉ xác thực **1 lỗi nhãn** (`P6961-88r` P→NP); sau correct, val 95.8% → 97.2%,
   test về mức cao nhất 96.5% — nhưng chênh lệch test chủ yếu là nhiễu huấn luyện, chứng tỏ dữ liệu Kaggle hiện tại đã sạch,
   không nên kỳ vọng tăng điểm lớn từ việc sửa nhãn.
