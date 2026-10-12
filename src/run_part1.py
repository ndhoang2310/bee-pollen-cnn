"""Phan I: baseline CNN tren Bee Pollen, split 7:1:2 (stratified), seed=42.
Chay: python3 src/run_part1.py [--epochs 30] [--img 128]
Can dataset Kaggle ivanfel/honey-bee-pollen (CSV + images/). Mac dinh doc tu /tmp/opencode/...,
doi CSV_DEFAULT/IMG_DEFAULT trong src/dataset.py neu duong dan khac.
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_table, stratified_split, PollenDataset
from model import BaselineCNN, count_params
from train_utils import train_model, evaluate
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    df = load_table()
    print(f"Total: {len(df)} (P={int((df.pollen_carrying==1).sum())}, NP={int((df.pollen_carrying==0).sum())})")
    tr, va, te = stratified_split(df, seed=a.seed)
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)} (ti le ~7:1:2, stratified)")
    tf = lambda: T.Compose([T.Resize((a.img, a.img)), T.ToTensor(),
                            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    eval_tf = tf()
    tr_l = DataLoader(PollenDataset(tr, eval_tf), batch_size=32, shuffle=True)
    va_l = DataLoader(PollenDataset(va, eval_tf), batch_size=32)
    te_l = DataLoader(PollenDataset(te, eval_tf), batch_size=32)
    m = BaselineCNN()
    print(f"BaselineCNN params: {count_params(m)}")
    print(m)
    m, h = train_model(m, tr_l, va_l, epochs=a.epochs, lr=1e-3, weight_decay=1e-4, patience=7)
    print(f"Best val_acc={h['best_val_acc']:.4f}")
    for n, l in [("train", tr_l), ("val", va_l), ("test", te_l)]:
        r = evaluate(m, l)
        print(f"[{n}] acc={r['accuracy']:.4f} prec={r['precision']:.4f} rec={r['recall']:.4f} "
              f"f1={r['f1']:.4f} auc={r['auc']:.4f} cm={r['confusion_matrix']}")

if __name__ == "__main__":
    main()
