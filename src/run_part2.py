"""Phan II: (1) augmentation, (2) train-dev chan doan mismatch, (3) correct label.
Chay tung buoc:
  python3 src/run_part2.py --step aug        # baseline + augmentation
  python3 src/run_part2.py --step train_dev  # train tren 399, danh gia train-dev/val/test
  python3 src/run_part2.py --step audit      # in danh sach ung vien nhan sai (can 3 seeds)
  python3 src/run_part2.py --step corrected  # train lai tren CSV da correct (data/pollen_data_corrected.csv)
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch
from dataset import load_table, stratified_split, train_dev_split, PollenDataset, IMG_DEFAULT
from model import BaselineCNN
from train_utils import train_model, evaluate

IMG = 128
def eval_tf_fn():
    return T.Compose([T.Resize((IMG, IMG)), T.ToTensor(),
                      T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def aug_tf_fn():
    return T.Compose([T.Resize((IMG, IMG)), T.RandomHorizontalFlip(0.5),
                      T.RandomRotation(15),
                      T.ColorJitter(0.2, 0.2, 0.2),
                      T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def step_aug():
    torch.manual_seed(42)
    df = load_table()
    tr, va, te = stratified_split(df)
    tr_l = DataLoader(PollenDataset(tr, aug_tf_fn()), batch_size=32, shuffle=True)
    va_l = DataLoader(PollenDataset(va, eval_tf_fn()), batch_size=32)
    te_l = DataLoader(PollenDataset(te, eval_tf_fn()), batch_size=32)
    m = BaselineCNN(dropout=0.5)
    m, h = train_model(m, tr_l, va_l, epochs=30, patience=8)
    print("best val", h["best_val_acc"])
    tr_e = DataLoader(PollenDataset(tr, eval_tf_fn()), batch_size=32)
    for n, l in [("train", tr_e), ("val", va_l), ("test", te_l)]:
        r = evaluate(m, l)
        print(f"[{n}] acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f} cm={r['confusion_matrix']}")

def step_train_dev():
    torch.manual_seed(42)
    df = load_table()
    tr, va, te = stratified_split(df)
    tr_s, tr_d = train_dev_split(tr, dev_ratio=0.2)
    print(f"train_small={len(tr_s)} train-dev={len(tr_d)} val={len(va)} test={len(te)}")
    e = eval_tf_fn()
    tr_l = DataLoader(PollenDataset(tr_s, e), batch_size=32, shuffle=True)
    d_l = DataLoader(PollenDataset(tr_d, e), batch_size=32)
    va_l = DataLoader(PollenDataset(va, e), batch_size=32)
    te_l = DataLoader(PollenDataset(te, e), batch_size=32)
    m = BaselineCNN()
    m, h = train_model(m, tr_l, va_l, epochs=30, patience=7)
    for n, l in [("train_small", tr_l), ("train-dev", d_l), ("val", va_l), ("test", te_l)]:
        r = evaluate(m, l)
        print(f"[{n}] acc={r['accuracy']:.4f} f1={r['f1']:.4f} cm={r['confusion_matrix']}")
    print("Luan giai: neu train-dev ~= train >> val/test => mismatched distribution. "
          "Neu train-dev ~= val ~= test < train => overfit phuong sai thong thuong.")

def step_audit():
    # Liet ke mau predict sai on dinh qua 3 seeds tren full data (de xuat ung vien review thu cong)
    from collections import Counter, defaultdict
    df = load_table()
    e = eval_tf_fn()
    err = Counter(); probs = defaultdict(list)
    for seed in [42, 7, 123]:
        tr, va, te = stratified_split(df, seed=42)
        tr_l = DataLoader(PollenDataset(tr, e), batch_size=32, shuffle=True)
        va_l = DataLoader(PollenDataset(va, e), batch_size=32)
        torch.manual_seed(seed)
        m = BaselineCNN()
        m, h = train_model(m, tr_l, va_l, epochs=25, patience=6, verbose=False)
        r = evaluate(m, DataLoader(PollenDataset(df, e), batch_size=32))
        for i, (t, p, pr) in enumerate(zip(r["y_true"], r["y_pred"], r["y_prob"])):
            fn = df.reset_index(drop=True).iloc[i]["filename"]
            probs[fn].append(pr)
            if t != p:
                err[fn] += 1
    print("Ung vien sai >=2/3 seeds (can review thu cong truoc khi sua):")
    for fn, c in err.most_common(30):
        t = df.set_index("filename").loc[fn, "pollen_carrying"]
        print(f"{fn} true={t} errs={c}/3 probs={[f'{x:.2f}' for x in probs[fn]]}")
    print("NGUYEN TAC: chi sua khi nhin thay ro phan cuc (pollen loads) trai voi nhan + model dong thuan tin cay cao. "
          "Khong flip hang loat theo model (tranh confirmation bias).")

def step_corrected():
    torch.manual_seed(42)
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv = os.path.join(proj, "data", "pollen_data_corrected.csv")
    df = pd.read_csv(csv)
    df["path"] = df["filename"].apply(lambda f: os.path.join(IMG_DEFAULT, f))
    print("corrected:", df["pollen_carrying"].value_counts().to_dict())
    tr, va, te = stratified_split(df)
    e = eval_tf_fn()
    tr_l = DataLoader(PollenDataset(tr, e), batch_size=32, shuffle=True)
    va_l = DataLoader(PollenDataset(va, e), batch_size=32)
    te_l = DataLoader(PollenDataset(te, e), batch_size=32)
    m = BaselineCNN()
    m, h = train_model(m, tr_l, va_l, epochs=30, patience=7)
    for n, l in [("train", tr_l), ("val", va_l), ("test", te_l)]:
        r = evaluate(m, l)
        print(f"[{n}] acc={r['accuracy']:.4f} prec={r['precision']:.4f} rec={r['recall']:.4f} "
              f"f1={r['f1']:.4f} auc={r['auc']:.4f} cm={r['confusion_matrix']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["aug", "train_dev", "audit", "corrected"], required=True)
    a = ap.parse_args()
    {"aug": step_aug, "train_dev": step_train_dev, "audit": step_audit, "corrected": step_corrected}[a.step]()
