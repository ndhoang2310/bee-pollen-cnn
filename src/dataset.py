"""Bee Pollen CNN - Dataset utilities: stratified 7:1:2 split + train-dev split."""
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import torch
from torch.utils.data import Dataset

CSV_DEFAULT = "/tmp/opencode/bee_test/pollendataset/PollenDataset/pollen_data.csv"
IMG_DEFAULT = "/tmp/opencode/bee_test/pollendataset/PollenDataset/images"

def load_table(csv_path=CSV_DEFAULT, img_dir=IMG_DEFAULT):
    df = pd.read_csv(csv_path)
    # only keep rows whose file exists
    df["path"] = df["filename"].apply(lambda f: os.path.join(img_dir, f))
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    return df

def stratified_split(df, seed=42):
    """Split 7:1:2 -> train/val/test, stratified by pollen_carrying."""
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    idx_trainval, idx_test = next(sss1.split(df, df["pollen_carrying"]))
    df_trainval = df.iloc[idx_trainval].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)
    # within trainval (80%), take val = 1/8 = 10% of total
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    idx_train, idx_val = next(sss2.split(df_trainval, df_trainval["pollen_carrying"]))
    df_train = df_trainval.iloc[idx_train].reset_index(drop=True)
    df_val = df_trainval.iloc[idx_val].reset_index(drop=True)
    return df_train, df_val, df_test

def train_dev_split(df_train, dev_ratio=0.2, seed=42):
    """Further split train -> train / train-dev (same distribution) for mismatch diagnosis."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=dev_ratio, random_state=seed)
    idx_t, idx_d = next(sss.split(df_train, df_train["pollen_carrying"]))
    return df_train.iloc[idx_t].reset_index(drop=True), df_train.iloc[idx_d].reset_index(drop=True)

class PollenDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        label = int(row["pollen_carrying"])
        if self.transform:
            img = self.transform(img)
        return img, label
