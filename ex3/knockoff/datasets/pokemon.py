import os.path as osp

from torchvision.datasets import ImageFolder

import knockoff.config as cfg

class PokemonDataset(ImageFolder):
    def __init__(self, train=True, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, 'pokemon-dataset-1000')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000'
            ))

        # Select the correct subfolder
        split = "train" if train else "test"
        dataset_path = osp.join(root, split)

        if not osp.exists(dataset_path):
            raise ValueError(f"Expected directory '{split}' not found in {root}")

        # Initialize the parent ImageFolder class
        super().__init__(root=dataset_path, transform=transform, target_transform=target_transform)