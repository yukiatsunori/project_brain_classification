# EEG Classification Project

## Updates

- **2024.06.18**: Added details about the dataset
  - The paper associated with the dataset is available here (you can also access it via the dataset link):
    - https://elifesciences.org/articles/82580
  - The distributed data has been processed according to the content in the "MEG data preprocessing and cleaning" section of the paper.
    - Therefore, the sampling rate is 200Hz.
  - As stated in the "MEG data acquisition" section, the coordinates of the channels follow the [CTF 275 MEG system](https://mne.tools/1.6/auto_examples/visualization/meg_sensors.html#ctf).
    - Please refer to this when incorporating channel coordinates into your model.
    - The reason why the distributed data contains only 271 channels is explained in the above two sections.

## Environment Setup

To set up your environment, follow these steps:

```bash
conda create -n dlbasics python=3.10
conda activate dlbasics
pip install -r requirements.txt
```

## Running the Baseline Model

### Training

```bash
python main.py
```

# Online visualization of the results (requires a wandb account)
```bash
python main.py use_wandb=True
```

## Task Details

- In this competition, the task is to classify **which class an image belongs to based on the EEG recorded while the subject is viewing the image**.

- The evaluation will be based on top-10 accuracy.
  - This means the model is considered correct if the correct class is included in the top 10 predicted probabilities.
  - The chance level is approximately 10 / 1,854 ≒ 0.54%.


