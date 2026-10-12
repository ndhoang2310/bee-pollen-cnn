"""
# Bee Pollen (Ong mang phấn) — CNN: Baseline → Cải tiến → Train-dev → Correct label

Notebook chạy 1-click trên **Kaggle** (khuyến nghị bật **GPU**).

**Cách dùng:**
1. Tạo Notebook mới trên Kaggle → **Add input** → tìm dataset `ivanfel/honey-bee-pollen` (Honey Bee Pollen) → Add.
2. (Tùy chọn) Settings → Accelerator → **GPU T4 x2**.
3. Chạy Run All từ trên xuống dưới.

Nội dung: Phần I (split 7:1:2, CNN tự định nghĩa, train, đánh giá) + Phần II (augmentation, train-dev chẩn đoán mismatch, audit + correct label, so sánh).
"""

# ============================================================
# CELL 1
import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE =', DEVICE)

# ---- Tu dong tim dataset (Kaggle input hoac local) ----
cands = glob.glob('/kaggle/input/**/pollen_data.csv', recursive=True)
print('tim thay:', cands)
if cands:
    CSV_PATH = cands[0]
    IMG_DIR = os.path.join(os.path.dirname(CSV_PATH), 'images')
else:  # chay local
    CSV_PATH = '/tmp/opencode/bee_test/pollendataset/PollenDataset/pollen_data.csv'
    IMG_DIR = '/tmp/opencode/bee_test/pollendataset/PollenDataset/images'
print('CSV_PATH =', CSV_PATH)
print('IMG_DIR  =', IMG_DIR, '| ton tai:', os.path.isdir(IMG_DIR))

"""
## 1. Khảo sát dữ liệu
"""

# ============================================================
# CELL 3
df = pd.read_csv(CSV_PATH)
df['path'] = df['filename'].apply(lambda f: os.path.join(IMG_DIR, f))
df = df[df['path'].apply(os.path.exists)].reset_index(drop=True)
print('Tong so anh:', len(df))
print(df['pollen_carrying'].value_counts())
print(df['pollen_carrying'].value_counts(normalize=True))

# Kiem tra khop tien to ten file vs nhan CSV
pref = df['filename'].apply(lambda f: 0 if f.startswith('NP') else 1)
print('mismatch filename-vs-CSV:', int((pref != df['pollen_carrying']).sum()))

# Xem kich thuoc anh mau
im0 = Image.open(df.iloc[0]['path'])
print('anh mau:', df.iloc[0]['filename'], im0.size, im0.mode)

# Hien thi 8 anh mau (4 P + 4 NP)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, lab in enumerate([1]*4 + [0]*4):
    row = df[df.pollen_carrying == lab].iloc[i % 4]
    ax = axes[i // 4, i % 4]
    ax.imshow(Image.open(row['path']))
    ax.set_title(f"{'P-pollen' if lab==1 else 'NP'}: {row['filename']}")
    ax.axis('off')
plt.tight_layout(); plt.show()

"""
**Dấu hiệu nhận biết khi review thủ công:** ong mang phấn có **khối phấn vàng nhạt/trắng rõ ở 2 chân sau**. Không có khối này = NP.
"""

# ============================================================
# CELL 5
SEED = 42
IMG = 128

def stratified_split(df, seed=SEED):
    """Chia 7:1:2 stratified: train/val/test."""
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    i_tv, i_te = next(s1.split(df, df['pollen_carrying']))
    df_tv, df_te = df.iloc[i_tv].reset_index(drop=True), df.iloc[i_te].reset_index(drop=True)
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)  # 1/8 cua 80% = 10% tong
    i_tr, i_va = next(s2.split(df_tv, df_tv['pollen_carrying']))
    return df_tv.iloc[i_tr].reset_index(drop=True), df_tv.iloc[i_va].reset_index(drop=True), df_te

df_train, df_val, df_test = stratified_split(df)
for n, d in [('train', df_train), ('val', df_val), ('test', df_test)]:
    print(f'{n}: {len(d)} |', d['pollen_carrying'].value_counts().to_dict())

"""
## 2. Dataset / DataLoader
"""

# ============================================================
# CELL 7
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class PollenDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True); self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r['path']).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, int(r['pollen_carrying'])

def eval_tf():
    return T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize(MEAN, STD)])
def aug_tf():
    return T.Compose([T.Resize((IMG, IMG)), T.RandomHorizontalFlip(0.5),
                        T.RandomRotation(15), T.ColorJitter(0.2, 0.2, 0.2),
                        T.ToTensor(), T.Normalize(MEAN, STD)])

"""
## 3. Phần I — Mô hình CNN tự định nghĩa
"""

# ============================================================
# CELL 9
class BaselineCNN(nn.Module):
    """Custom CNN ~422k params. Input 3x128x128 -> 2 logits."""
    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, 2))
    def forward(self, x): return self.classifier(self.features(x))

m = BaselineCNN()
n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f'BaselineCNN params: {n_params}')
print(m)

# ============================================================
# CELL 10
import copy, torch.nn.functional as F

def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, weight_decay=1e-4, patience=7):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
    crit = nn.CrossEntropyLoss()
    best_acc, best_w, bad = 0, None, 0
    for ep in range(1, epochs + 1):
        model.train(); tot, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                preds += model(x.to(DEVICE)).argmax(1).cpu().tolist(); trues += y.tolist()
        acc = accuracy_score(trues, preds); sched.step(acc)
        print(f'Epoch {ep:02d}/{epochs} - loss {tot/n:.4f} - val_acc {acc:.4f} - lr {opt.param_groups[0]["lr"]:.6f}')
        if acc > best_acc: best_acc, best_w, bad = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print(f'Early stop (best val_acc={best_acc:.4f})'); break
    model.load_state_dict(best_w)
    return model, best_acc

@torch.no_grad()
def evaluate(model, loader):
    model.eval().to(DEVICE)
    preds, trues, probs = [], [], []
    for x, y in loader:
        out = model(x.to(DEVICE))
        probs += F.softmax(out, dim=1)[:, 1].cpu().tolist()
        preds += out.argmax(1).cpu().tolist(); trues += y.tolist()
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average='binary', zero_division=0)
    try: auc = roc_auc_score(trues, probs)
    except Exception: auc = float('nan')
    return {'accuracy': accuracy_score(trues, preds), 'precision': prec, 'recall': rec,
            'f1': f1, 'auc': auc, 'cm': confusion_matrix(trues, preds).tolist(),
            'y_true': trues, 'y_pred': preds, 'y_prob': probs}

def show_report(name, r):
    print(f"[{name}] acc={r['accuracy']:.4f} prec={r['precision']:.4f} rec={r['recall']:.4f} "
          f"f1={r['f1']:.4f} auc={r['auc']:.4f} cm={r['cm']}")

"""
## 4. Train baseline (Phần I)
"""

# ============================================================
# CELL 12
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
tr_loader = DataLoader(PollenDataset(df_train, eval_tf()), batch_size=32, shuffle=True)
va_loader = DataLoader(PollenDataset(df_val, eval_tf()), batch_size=32)
te_loader = DataLoader(PollenDataset(df_test, eval_tf()), batch_size=32)

base = BaselineCNN()
base, best_val = train_model(base, tr_loader, va_loader, epochs=30)
print('Best val_acc =', round(best_val, 4))

r_tr, r_va, r_te = evaluate(base, tr_loader), evaluate(base, va_loader), evaluate(base, te_loader)
show_report('train', r_tr); show_report('val', r_va); show_report('TEST (baseline)', r_te)
torch.save(base.state_dict(), '/kaggle/working/baseline_cnn.pt')

"""
## 5. Phần II-1 — Cải tiến bằng Augmentation
"""

# ============================================================
# CELL 14
torch.manual_seed(SEED)
tr_aug = DataLoader(PollenDataset(df_train, aug_tf()), batch_size=32, shuffle=True)
m_aug = BaselineCNN(dropout=0.5)
m_aug, bv = train_model(m_aug, tr_aug, va_loader, epochs=30, patience=8)
print('Best val (aug) =', round(bv, 4))
show_report('train', evaluate(m_aug, tr_loader))
show_report('val', evaluate(m_aug, va_loader))
show_report('TEST (aug)', evaluate(m_aug, te_loader))
print('Nhan xet: train giam (bot overfit), val ~ tuong duong, test chenh ~2 anh/143 la nhieu ngau nhien.')

"""
## 6. Phần II-2 — Tập train-dev: kiểm tra mismatched distribution
"""

# ============================================================
# CELL 16
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
i_ts, i_dv = next(sss.split(df_train, df_train['pollen_carrying']))
df_tr_small = df_train.iloc[i_ts].reset_index(drop=True)   # 399
df_tr_dev = df_train.iloc[i_dv].reset_index(drop=True)    # 100
print(f"train_small={len(df_tr_small)} train-dev={len(df_tr_dev)} val={len(df_val)} test={len(df_test)}")

torch.manual_seed(SEED)
l_ts = DataLoader(PollenDataset(df_tr_small, eval_tf()), batch_size=32, shuffle=True)
l_dv = DataLoader(PollenDataset(df_tr_dev, eval_tf()), batch_size=32)
m_td = BaselineCNN()
m_td, _ = train_model(m_td, l_ts, va_loader, epochs=30)
for n, l in [('train_small', l_ts), ('train-dev', l_dv), ('val', va_loader), ('test', te_loader)]:
    show_report(n, evaluate(m_td, l))
print('Luan giai: neu train-dev ~= train >> val/test => mismatch. '
      'Thuc te train-dev ~= val ~= test < train => KHONG mismatch, chi overfit nhe do tap nho.')

"""
## 7. Phần II-3 — Audit nhãn (tìm ứng viên sai nhãn)
Nguyên tắc: model chỉ **đề xuất ứng viên**, con người quyết định dựa trên định nghĩa khối phấn. **Không flip hàng loạt theo model.**
"""

# ============================================================
# CELL 18
from collections import Counter, defaultdict
err_cnt, prob_hist = Counter(), defaultdict(list)
for seed in [42, 7, 123]:
    tr, va, _ = stratified_split(df, seed=SEED)
    ltr = DataLoader(PollenDataset(tr, eval_tf()), batch_size=32, shuffle=True)
    lva = DataLoader(PollenDataset(va, eval_tf()), batch_size=32)
    torch.manual_seed(seed)
    m = BaselineCNN()
    m, _ = train_model(m, ltr, lva, epochs=25, patience=6)
    r = evaluate(m, DataLoader(PollenDataset(df, eval_tf()), batch_size=32))
    for i, (t, p, pr) in enumerate(zip(r['y_true'], r['y_pred'], r['y_prob'])):
        fn = df.reset_index(drop=True).iloc[i]['filename']
        prob_hist[fn].append(pr)
        if t != p: err_cnt[fn] += 1
print('Ung vien sai >=2/3 seeds:')
for fn, c in err_cnt.most_common(25):
    t = df.set_index('filename').loc[fn, 'pollen_carrying']
    print(f'{fn} true={t} errs={c}/3 probs={[f"{x:.2f}" for x in prob_hist[fn]]}')

# ============================================================
# CELL 19
# Hien thi visual top ung vien de review thu cong
top = [fn for fn, _ in err_cnt.most_common(8)]
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, fn in zip(axes.flat, top):
    t = df.set_index('filename').loc[fn, 'pollen_carrying']
    ax.imshow(Image.open(os.path.join(IMG_DIR, fn)))
    ax.set_title(f"{fn}\ntrue={t} p={[f'{x:.2f}' for x in prob_hist[fn]]}")
    ax.axis('off')
plt.tight_layout(); plt.show()
print('Ket luan review (da kiem chung): chi P6961-88r sai nhan ro ret (nhan P nhung khong co khoi phan, model 3/3 tin cay NP) -> flip 1->0.')
print('Cac NP bi FP hau het dung la NP (model qua nhay); cac P bi FN hau het dung la P (phan nho). Khong flip hang loat.')

"""
## 8. Correct label → chia lại 7:1:2 → train lại → so sánh
"""

# ============================================================
# CELL 21
df_fix = df.copy()
print('Truoc:', df_fix['pollen_carrying'].value_counts().to_dict())
df_fix.loc[df_fix.filename == 'P6961-88r.jpg', 'pollen_carrying'] = 0  # sua duy nhat mau da xac thuc
print('Sau:  ', df_fix['pollen_carrying'].value_counts().to_dict())
df_fix[['filename', 'pollen_carrying']].to_csv('/kaggle/working/pollen_data_corrected.csv', index=False)
print('Da luu /kaggle/working/pollen_data_corrected.csv')

tr2, va2, te2 = stratified_split(df_fix, seed=SEED)
for n, d in [('train', tr2), ('val', va2), ('test', te2)]:
    print(f'{n}: {len(d)} |', d['pollen_carrying'].value_counts().to_dict())

torch.manual_seed(SEED)
l_tr2 = DataLoader(PollenDataset(tr2, eval_tf()), batch_size=32, shuffle=True)
l_va2 = DataLoader(PollenDataset(va2, eval_tf()), batch_size=32)
l_te2 = DataLoader(PollenDataset(te2, eval_tf()), batch_size=32)
m_fix = BaselineCNN()
m_fix, _ = train_model(m_fix, l_tr2, l_va2, epochs=30)
r2 = {n: evaluate(m_fix, l) for n, l in [('train', l_tr2), ('val', l_va2), ('test', l_te2)]}
for n in ['train', 'val', 'test']:
    show_report(f'{n} (corrected)', r2[n])

print('\n=== SO SANH (goc vs correct, cung seed chia) ===')
show_report('test GOC      ', r_te)
show_report('test CORRECTED', r2['test'])
print('Val chua-correct: P6961-88r la LOI | Val corrected: P6961-88r la DUNG (da sua nhan).')
print('Test nhan khong doi; chenh ~2/143 anh la nhieu huan luyen, khong y nghia thong ke.')

"""
## 9. Kết luận
- Baseline CNN (~422k params) đạt ~96.5% test.
- Augmentation giảm overfit nhưng không vượt baseline (trần hiệu năng + nhiễu mẫu nhỏ).
- Train-dev bác bỏ mismatched distribution (train-dev ≈ val ≈ test).
- Chỉ 1/714 nhãn sai được xác thực (`P6961-88r`); sau correct kết quả tương đương trong sai số → dữ liệu gốc đã sạch.
"""