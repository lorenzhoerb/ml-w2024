import os.path as osp

from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
import pandas as pd

import knockoff.config as cfg

class EmotionDataset(ImageFolder):
    def __init__(self, train=True, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, 'emotion_dataset')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset'
            ))

        # Initialize the parent ImageFolder class
        super().__init__(root=osp.join(root, "dataset"), transform=transform, target_transform=target_transform)

        self.root = root

        # Load CSV file with image metadata
        csv_path = osp.join(self.root, "data.csv")
        if not osp.exists(csv_path):
            raise ValueError(f"Dataset CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        # Full image path
        df["path"] = df["path"].apply(lambda x: osp.join(self.root, "dataset", x))

        # Train-Test split
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

        # Select the correct DataFrame based on the train flag
        self.data_df = train_df if train else test_df

        # Creating imgs/samples from the CSV DataFrame
        self.imgs = [(row['path'], self.class_to_idx[row['label']]) for _, row in self.data_df.iterrows()]
        self.samples = self.imgs
