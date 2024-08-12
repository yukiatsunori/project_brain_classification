import os

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from termcolor import cprint
from torchmetrics import Accuracy
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed


def objective(trial):
    set_seed(1234)  # 再現性のために固定のシードを設定

    # チューニングするハイパーパラメータ
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    num_epochs = 80
    num_workers = 4

    # データローダー
    loader_args = {"batch_size": batch_size, "num_workers": num_workers}
    train_set = ThingsMEGDataset("train", "data")
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", "data")
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", "data")
    test_loader = torch.utils.data.DataLoader(
        test_set,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # モデル
    model = BasicConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels
    ).to("cuda:0")

    # オプティマイザ
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to("cuda:0")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to("cuda:0"), y.to("cuda:0")

            y_pred = model(X)

            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to("cuda:0"), y.to("cuda:0")

            with torch.no_grad():
                y_pred = model(X)

            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        mean_val_acc = np.mean(val_acc)
        if mean_val_acc > max_val_acc:
            max_val_acc = mean_val_acc
            cprint("New best.", "cyan")

    return max_val_acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print(f"Best trial: {study.best_trial.value}")
    print(f"Best hyperparameters: {study.best_trial.params}")
