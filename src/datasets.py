import os
from typing import Tuple

import numpy as np
import psutil
import torch
from scipy.signal import butter, filtfilt
from termcolor import cprint


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        data_dir: str = "data",
        preprocess: bool = False,
        subset: int = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.device = device

        print(f"Loading data for split: {split}")
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        print(f"Loaded {split}_X.pt")

        self.subject_idxs = torch.load(
            os.path.join(data_dir, f"{split}_subject_idxs.pt")
        )
        print(f"Loaded {split}_subject_idxs.pt")

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            print(f"Loaded {split}_y.pt")
            assert (
                len(torch.unique(self.y)) == self.num_classes
            ), "Number of classes do not match."

        if subset:
            print(f"Using subset of data: first {subset} samples")
            self.X = self.X[:subset]
            self.subject_idxs = self.subject_idxs[:subset]
            if hasattr(self, "y"):
                self.y = self.y[:subset]

        if preprocess:
            print(f"Preprocessing data for split: {split}")
            self.check_memory("Before preprocessing")
            self.X = self.batch_preprocess(
                self.X, batch_size=10
            )  # バッチサイズを小さく設定
            print(f"Preprocessing completed for split: {split}")

        # データはここではCPU上に保持しておく
        # self.X = self.X.to(self.device)
        # self.subject_idxs = self.subject_idxs.to(self.device)
        # if hasattr(self, "y"):
        #     self.y = self.y.to(self.device)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            print(f"Fetching item {i}")
            X = self.X[i].to(self.device)
            y = self.y[i].to(self.device)
            subject_idxs = self.subject_idxs[i].to(self.device)
            return X, y, subject_idxs
        else:
            print(f"Fetching item {i}")
            X = self.X[i].to(self.device)
            subject_idxs = self.subject_idxs[i].to(self.device)
            return X, subject_idxs

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

    def preprocess_data(self, data):
        print("Starting preprocessing")
        self.check_memory("Before converting to numpy")
        data = data.numpy()  # テンソルを NumPy 配列に変換
        self.check_memory("After converting to numpy")
        print("Converted to numpy array")

        # 設定
        lowcut = 0.5
        highcut = 30.0
        fs = 200  # 元のサンプリングレート

        # バンドパスフィルタ
        data = self.bandpass_filter(data, lowcut, highcut, fs)
        print("Applied bandpass filter")

        # 正規化
        data = self.normalize_data(data)
        print("Applied normalization")

        # ベースライン補正
        data = self.baseline_correction(data)
        print("Applied baseline correction")

        print("Preprocessing finished")
        return torch.tensor(data, dtype=torch.float32)  # NumPy 配列をテンソルに変換

    def batch_preprocess(self, data, batch_size):
        num_samples = data.size(0)
        processed_data = []

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = data[start_idx:end_idx]
            processed_batch = self.preprocess_data(batch_data)
            processed_data.append(processed_batch)

        return torch.cat(processed_data, dim=0)

    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        return b, a

    def bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=2)  # 修正: データの軸を考慮
        return y

    def resample_data(self, data, original_fs, target_fs):
        ratio = target_fs / original_fs
        num_samples = int(data.shape[2] * ratio)
        resampled_data = np.zeros((data.shape[0], data.shape[1], num_samples))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                resampled_data[i, j, :] = np.interp(
                    np.linspace(0, data.shape[2], num=num_samples),
                    np.arange(data.shape[2]),
                    data[i, j, :],
                )
        return resampled_data

    def normalize_data(self, data):
        mean = np.mean(data, axis=2, keepdims=True)
        std = np.std(data, axis=2, keepdims=True)
        return (data - mean) / std

    def baseline_correction(self, data):
        baseline = np.mean(data[:, :, :50], axis=2, keepdims=True)
        return data - baseline

    def check_memory(self, message):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"{message} - Memory usage: {mem_info.rss / (1024 ** 2)} MB")


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
