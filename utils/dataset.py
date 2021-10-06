import os
import torch
import numpy as np
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.data = []
        self.transform = transform
        COVER = np.array([1, 0])
        STEGO = np.array([0, 1])

        self.readData(img_dir + '/cover_png', COVER)
        self.readData(img_dir + '/stego_png', STEGO)

    def readData(self, path, label):
        file_names = os.listdir(path)
        for file_name in file_names:
            file_path = path + '/' + file_name
            self.data.append((file_path, label))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label