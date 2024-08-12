import multiprocessing as mp
import os
import sys

import hydra
import numpy as np
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

# マルチプロセッシングのスタートメソッドを 'spawn' に設定
mp.set_start_method("spawn", force=True)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    device = get_device()
    print("Starting run function")
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        print("Initializing wandb")
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    print("Creating dataloaders")
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": 0,
    }  # num_workersを0に設定

    try:
        print("Loading train dataset")
        train_set = ThingsMEGDataset("train", args.data_dir, preprocess=False)
        print(f"Train dataset loaded with {len(train_set)} samples")
        train_loader = torch.utils.data.DataLoader(
            train_set, shuffle=True, **loader_args
        )

        print("Loading val dataset")
        val_set = ThingsMEGDataset("val", args.data_dir, preprocess=False)
        print(f"Val dataset loaded with {len(val_set)} samples")
        val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)

        print("Loading test dataset")
        test_set = ThingsMEGDataset("test", args.data_dir, preprocess=False)
        print(f"Test dataset loaded with {len(test_set)} samples")
        test_loader = torch.utils.data.DataLoader(
            test_set,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=0,
        )
    except Exception as e:
        print(f"Error in creating dataloaders: {e}")
        sys.exit(1)

    print("Creating model")
    model = BasicConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels
    ).to(device)

    print("Creating optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training")
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(device)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(device), y.to(device)

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
            X, y = X.to(device), y.to(device)

            with torch.no_grad():
                y_pred = model(X)

            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(
            f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}"
        )
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": np.mean(train_loss),
                    "train_acc": np.mean(train_acc),
                    "val_loss": np.mean(val_loss),
                    "val_acc": np.mean(val_acc),
                }
            )

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)

    print("Starting evaluation")
    model.load_state_dict(
        torch.load(os.path.join(logdir, "model_best.pt"), map_location=device)
    )

    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(model(X.to(device)).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    print("Starting main function")
    run()
