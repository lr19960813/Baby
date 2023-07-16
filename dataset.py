from PIL import Image
import clip
import os
import pandas as pd

from torch.utils.data import Dataset

import config as cfg



class BabyDataset(Dataset):
    def __init__(self, txt_path):
        with open(txt_path, 'r') as f:
            self.file_path = f.read().splitlines()
        _, preprocess = clip.load("ViT-B/32", device = 0, jit = False)
        self.preprocess = preprocess

    def __getitem__(self, idx):
        label = self.file_path[idx].split('/')[0]
        if label == 'wo_baby':
            label = 0
        elif label == 'w_baby':
            label = 1
        else:
            raise RuntimeError
        image = Image.open(os.path.join(cfg.path.img_data, self.file_path[idx]))
        image = self.preprocess(image)
        return image, label

    def __len__(self):
        return len(self.file_path)
