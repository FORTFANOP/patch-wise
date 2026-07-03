import cv2
import random
import torch
import torchvision
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


path = './data/cars'

class carsData(Dataset):
    def __init__(self, img_paths, base_transform=None, input_transform=None):
        self.img_paths = img_paths
        self.base_transform = base_transform
        self.input_transform = input_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")

        if self.base_transform:
            img = self.base_transform(img)
        target = img.clone()

        inp = img.clone()
        if self.input_transform:
            inp = self.input_transform(inp)

        return inp, target


class randomPatchify:
    def __init__(self, num_patches=(1, 5), patch_size=(10, 40), p=0.5):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.p = p

    def __call__(self, img):
        # img: Tensor [C, H, W]

        if random.random() > self.p:
            return img

        img = img.clone()

        _, H, W = img.shape

        n = random.randint(*self.num_patches)

        for patch in range(n):
            h = random.randint(*self.patch_size)
            w = random.randint(*self.patch_size)

            y = random.randint(0, max(0, H-h))
            x = random.randint(0, max(0,  W-w))

            img[:, y:y+h, x:x+w] = 0

        return img




