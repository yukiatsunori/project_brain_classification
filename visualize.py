import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# ThingsMEGDatasetクラスをインポート
from src.datasets import ThingsMEGDataset


def plot_selected_channels(eeg_data, channels, title="Selected EEG Channels"):
    plt.figure(figsize=(15, 5))
    for i, channel in enumerate(channels):
        plt.plot(eeg_data[channel] + i * 10, label=f"Channel {channel}")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


def plot_eeg_in_segments(eeg_data, segment_size=10, title="EEG Data"):
    num_channels = eeg_data.shape[0]
    num_segments = (num_channels + segment_size - 1) // segment_size

    plt.figure(figsize=(15, num_segments * 5))
    for i in range(num_segments):
        start = i * segment_size
        end = min(start + segment_size, num_channels)
        plt.subplot(num_segments, 1, i + 1)
        for j in range(start, end):
            plt.plot(eeg_data[j] + (j - start) * 10)
        plt.title(f"{title} (Channels {start}-{end-1})")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig):
    # データセットのロード
    dataset = ThingsMEGDataset("train", args.data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # サンプルの抽出
    num_samples = 5
    samples = []
    for i, (eeg_data, label, subject_idx) in enumerate(dataloader):
        if i >= num_samples:
            break
        samples.append((eeg_data.squeeze().numpy(), label.item(), subject_idx.item()))

    # サンプルのプロット
    for i, (eeg_data, label, subject_idx) in enumerate(samples):
        # 特定のチャネルを選択してプロット
        channels_to_plot = np.arange(
            dataset.num_channels
        )  # 選択されたチャネル全てをプロット
        plot_selected_channels(
            eeg_data,
            channels_to_plot,
            title=f"Sample {i+1} - Label: {label}, Subject: {subject_idx}",
        )

        # すべてのチャネルを複数の小さなプロットに分けて表示
        plot_eeg_in_segments(
            eeg_data,
            segment_size=10,
            title=f"Sample {i+1} - Label: {label}, Subject: {subject_idx}",
        )


if __name__ == "__main__":
    main()
