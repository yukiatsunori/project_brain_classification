import os

import numpy as np
import psutil
import torch
from scipy.signal import butter, filtfilt


class ThingsMEGDatasetPreprocessor:
    def __init__(self, data_dir: str = "data", batch_size: int = 10):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def preprocess_and_save(self, split: str):
        X = torch.load(os.path.join(self.data_dir, f"{split}_X.pt")).cpu().numpy()
        subject_idxs = (
            torch.load(os.path.join(self.data_dir, f"{split}_subject_idxs.pt"))
            .cpu()
            .numpy()
        )

        if split in ["train", "val"]:
            y = torch.load(os.path.join(self.data_dir, f"{split}_y.pt")).cpu().numpy()

        # バッチ処理で前処理を行う
        processed_X = []
        for i in range(0, len(X), self.batch_size):
            batch_X = X[i : i + self.batch_size]
            processed_X.append(self.preprocess_data(batch_X))
        processed_X = np.concatenate(processed_X, axis=0)

        # 前処理済みデータを保存
        np.save(os.path.join(self.data_dir, f"{split}_X_preprocessed.npy"), processed_X)
        np.save(
            os.path.join(self.data_dir, f"{split}_subject_idxs_preprocessed.npy"),
            subject_idxs,
        )

        if split in ["train", "val"]:
            np.save(os.path.join(self.data_dir, f"{split}_y_preprocessed.npy"), y)

    def preprocess_data(self, data):
        print("Starting preprocessing")
        self.check_memory("Before converting to numpy")
        data = data  # テンソルを NumPy 配列に変換
        self.check_memory("After converting to numpy")
        print("Converted to numpy array")

        # 設定
        lowcut = 0.5
        highcut = 30.0
        fs = 200  # 元のサンプリングレート
        target_fs = 100  # 目標のサンプリングレート

        # バンドパスフィルタ
        data = self.bandpass_filter(data, lowcut, highcut, fs)
        print("Applied bandpass filter")

        # リサンプリング
        data = self.resample_data(data, fs, target_fs)
        print("Applied resampling")

        # 正規化
        data = self.normalize_data(data)
        print("Applied normalization")

        # ベースライン補正
        data = self.baseline_correction(data)
        print("Applied baseline correction")

        print("Preprocessing finished")
        return torch.tensor(data, dtype=torch.float32)  # NumPy 配列をテンソルに変換

    def batch_preprocess(self, data, batch_size):
        num_samples = data.shape[0]
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
        y = filtfilt(b, a, data, axis=2)
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


if __name__ == "__main__":
    preprocessor = ThingsMEGDatasetPreprocessor()
    preprocessor.preprocess_and_save("train")
    preprocessor.preprocess_and_save("val")
    preprocessor.preprocess_and_save("test")
