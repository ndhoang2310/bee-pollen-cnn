"""Training + evaluation helpers."""
import torch, copy, time
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

def train_model(model, train_loader, val_loader, device="cpu", epochs=30, lr=1e-3,
                weight_decay=1e-4, patience=7, verbose=True):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    crit = nn.CrossEntropyLoss()
    best_acc, best_w, bad = 0, None, 0
    hist = {"train_loss": [], "val_acc": []}
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        # val
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                p = model(x.to(device)).argmax(1).cpu().tolist()
                preds += p; trues += y.tolist()
        acc = accuracy_score(trues, preds)
        sched.step(acc)
        hist["train_loss"].append(tot / n); hist["val_acc"].append(acc)
        if verbose:
            print(f"Epoch {ep:02d}/{epochs} - loss {tot/n:.4f} - val_acc {acc:.4f} - lr {opt.param_groups[0]['lr']:.6f}")
        if acc > best_acc:
            best_acc, best_w, bad = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"Early stop at epoch {ep} (best val_acc={best_acc:.4f})")
                break
    if best_w is not None:
        model.load_state_dict(best_w)
    return model, {"best_val_acc": best_acc, **hist}

@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval().to(device)
    preds, trues, probs = [], [], []
    import torch.nn.functional as F
    for x, y in loader:
        out = model(x.to(device))
        pr = F.softmax(out, dim=1)[:, 1].cpu().tolist()
        p = out.argmax(1).cpu().tolist()
        preds += p; trues += y.tolist(); probs += pr
    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(trues, probs)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(trues, preds).tolist()
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "confusion_matrix": cm, "y_true": trues, "y_pred": preds, "y_prob": probs}
