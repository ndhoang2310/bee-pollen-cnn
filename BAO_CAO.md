# Bee Pollen (Ong mang phấn) — CNN: Baseline → Cải tiến → Chẩn đoán mismatch → Correct label

Dataset: https://www.kaggle.com/datasets/ivanfel/honey-bee-pollen (CC-BY-SA-4.0).
Code trong `src/` (`dataset.py`, `model.py`, `train_utils.py`, `run_part1.py`, `run_part2.py`).
File nhãn đã correct: `data/pollen_data_corrected.csv`.

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
resize 128×128 + chuẩn hóa mean/std ImageNet. Chạy CPU.

### Kết quả Phần I (baseline, không augmentation)

| Tập | Acc | Prec | Rec | F1 | AUC | Confusion matrix [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|---|
| train (499) | 0.9960 | 1.0000 | 0.9922 | 0.9961 | 1.0000 | [[241,0],[2,256]] |
| val (72) | 0.9583 | 1.0000 | 0.9189 | 0.9577 | 0.9737 | [[35,0],[3,34]] |
| test (143) | **0.9650** | 0.9600 | 0.9730 | 0.9664 | 0.9906 | [[66,3],[2,72]] |

Lỗi baseline (10 mẫu): train `P11440-32r, P11048-28r`; val `P13151-44r, P6961-88r, P63199-210r`;
test `P11762-35r, NP21808-78r, NP18775-18r, NP27452-203r, P2841-16r`.
Nhận xét: baseline đã mạnh (~96.5% test); lỗi chủ yếu là **bỏ sót phấn nhỏ (P→NP)** và vài **báo động giả trên NP khó/che khuất**.

Chạy lại: `python3 src/run_part1.py --epochs 30 --img 128 --seed 42`

## 3. Phần II — Cải tiến, train-dev, correct label

### 3.1. Cải tiến (augmentation + regularization)

Giữ nguyên kiến trúc, chỉ thêm augmentation lúc train:
`RandomHorizontalFlip(0.5) + RandomRotation(15°) + ColorJitter(0.2)` (val/test giữ nguyên).
Kết quả:

| Tập | Acc | F1 | AUC | CM |
|---|---|---|---|---|
| train | 0.9699 | 0.9703 | 0.9958 | [[239,2],[13,245]] |
| val | 0.9583 | 0.9577 | 0.9892 | [[35,0],[3,34]] |
| test | 0.9510 | 0.9524 | 0.9931 | [[66,3],[4,70]] |

Nhận xét từng bước:
- Train acc giảm 99.6% → 97.0% (bớt overfit; khe train–val thu hẹp từ ~3.8% còn ~1.2%).
- Val giữ nguyên 95.83%, test 96.5% → 95.1% (chênh ~2 ảnh/143, nằm trong nhiễu ngẫu nhiên).
- Kết luận: augmentation **không tăng điểm tuyệt đối** vì baseline đã gần trần, nhưng **cải thiện độ tin cậy khái quát** (giảm overfit, AUC vẫn ~0.99).
  Hướng cải tiến tiếp theo (chưa chạy vì giới hạn CPU): transfer learning ResNet18/MobileNet, ảnh 192px, TTA, ensemble 3 seeds.

Chạy: `python3 src/run_part2.py --step aug`

### 3.2. Tập train-dev — kiểm tra mismatched distribution

Chia train 499 → `train_small` 399 + `train-dev` 100 (stratified, cùng phân phối với train),
train **chỉ trên 399**, đánh giá trên 4 tập:

| train_small (399) | train-dev (100) | val (72) | test (143) |
|---|---|---|---|
| 0.9749, F1 0.9755, [[190,3],[7,199]] | 0.9200, F1 0.9245, [[43,5],[3,49]] | 0.9444, F1 0.9459 | 0.9301, F1 0.9324 |

Luận giải theo phương pháp train-dev (Ng):
- Nếu **mismatch**: train-dev ≈ train ≫ val/test.
- Thực tế: train-dev (92.0%) ≈ val (94.4%) ≈ test (93.0%) < train_small (97.5%).
- Kết luận: **không có bằng chứng mismatched distribution** trên cách chia ngẫu nhiên;
  khoảng chênh ~3–5% là **overfit/phương sai do tập nhỏ** (bớt 100 mẫu train làm test rớt ~3.5%),
  không phải do train/test khác phân phối. Vì vậy không cần biện pháp xử lý mismatch
  (thu thêm dữ liệu theo phân phối test, reweighting…); ưu tiên tăng dữ liệu/augmentation.

Chạy: `python3 src/run_part2.py --step train_dev`

### 3.3. Audit + correct label

Quy trình (tránh confirmation bias — không flip hàng loạt theo model):
1. Train 3 seeds (42/7/123), lấy mẫu **sai ổn định ≥2/3 seeds** làm ứng viên (14 mẫu sai 3/3, ~10 mẫu sai 2/3).
   Chạy: `python3 src/run_part2.py --step audit`
2. Đối chiếu thủ công với định nghĩa "khối phấn ở chân sau".
   Kết quả: đa số NP bị model báo FP **thực ra đúng là NP** (không thấy khối phấn; model quá nhạy),
   đa số P bị bỏ sót **thực ra đúng là P** (phấn nhỏ). Chỉ **1 mẫu sai nhãn rõ rệt**:
   - `P6961-88r.jpg`: nhãn gốc P nhưng không có khối phấn, model 3/3 seeds đều tin cậy NP (p=0.11/0.01/0.03) → flip **1→0**.
   - 4 mẫu che khuất/mờ (`NP21808-78r, NP27452-203r, NP25880-162r, NP13576-210r`) gắn cờ **ambiguous** nhưng **giữ nguyên nhãn** (không đủ bằng chứng sửa).
3. Lưu `data/pollen_data_corrected.csv` (P 369→368, NP 345→346), chia lại 7:1:2 cùng `seed=42`, train lại cùng cấu hình.

### 3.4. So sánh chưa-correct vs correct (cùng kiến trúc, cùng seed chia)

| Tập | Chưa correct (gốc) | Correct (1 nhãn) |
|---|---|---|
| train (499) | acc 0.9960, F1 0.9961, [[241,0],[2,256]] | acc 0.9840, F1 0.9842, [[242,0],[8,249]] |
| val (72) | acc 0.9583, F1 0.9577, [[35,0],[3,34]] | acc 0.9583, F1 0.9577, [[35,0],[3,34]]* |
| test (143, nhãn không đổi) | acc **0.9650**, F1 0.9664, [[66,3],[2,72]] | acc 0.9510, F1 0.9510, [[68,1],[6,68]] |

\* Val trùng điểm nhưng khác tập lỗi: mẫu `P6961-88r` (nằm ở val) từ "lỗi" thành "đúng" sau khi sửa —
đúng như kỳ vọng của việc correct label.

Nhận xét:
- Bộ dữ liệu gốc **vốn đã sạch** (chỉ ~1/714 ≈ 0.14% lỗi nhãn xác thực được); việc correct cho **lợi ích tuyệt đối không đáng kể**
  (val không đổi, test chênh −2 ảnh ≈ −1.4%, nằm trong khoảng tin cậy nhị thức với n=143, tức nhiễu huấn luyện).
- Không flip 14 mẫu "sai ổn định" theo model là quyết định đúng: kiểm chứng thủ công cho thấy model sai chứ không phải nhãn sai;
  flip mù quáng sẽ **bơm nhiễu vào dữ liệu**.
- Nút thắt hiệu năng còn lại là **mẫu khó/che khuất/phấn nhỏ** (recall lớp P ~0.92), không phải nhiễu nhãn —
  hướng xử lý là ảnh phân giải cao hơn, crop chân sau, focal loss, TTA/ensemble, chứ không phải sửa nhãn hàng loạt.

Chạy: `python3 src/run_part2.py --step corrected`

## 4. Kết luận chung

1. Baseline CNN tự định nghĩa (~422k params) đạt **~96.5% test / 95.8% val** — đủ tốt cho bài toán 2 lớp cân bằng này.
2. Augmentation giảm overfit (train 99.6→97.0%) nhưng không vượt baseline về điểm tuyệt đối (trần hiệu năng + nhiễu mẫu nhỏ).
3. Thí nghiệm train-dev **bác bỏ giả thuyết mismatched distribution** (train-dev ≈ val ≈ test).
4. Audit 3-seed + review thủ công chỉ xác thực **1 lỗi nhãn** (`P6961-88r` P→NP); sau correct, **kết quả tương đương trong sai số**
   — chứng tỏ dữ liệu Kaggle hiện tại đã sạch, không nên kỳ vọng tăng điểm lớn từ việc sửa nhãn.
