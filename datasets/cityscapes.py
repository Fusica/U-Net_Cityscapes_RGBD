import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from datasets import custom_transform as tr


class CityscapesRGBDDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.images_dir = os.path.join(root, 'leftImg8bit', split)
        self.depth_dir = os.path.join(root, 'disparity', split)
        self.labels_dir = os.path.join(root, 'gtFine', split)

        self.image_paths = []
        self.depth_paths = []
        self.label_paths = []

        for city in os.listdir(self.images_dir):
            if city == '.DS_Store':
                continue
            city_images_dir = os.path.join(self.images_dir, city)
            city_depth_dir = os.path.join(self.depth_dir, city)
            city_labels_dir = os.path.join(self.labels_dir, city)
            for file_name in os.listdir(city_images_dir):
                if file_name.endswith('_leftImg8bit.png'):
                    self.image_paths.append(os.path.join(city_images_dir, file_name))
                    depth_name = file_name.replace('_leftImg8bit.png', '_disparity.png')
                    self.depth_paths.append(os.path.join(city_depth_dir, depth_name))
                    label_name = file_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    self.label_paths.append(os.path.join(city_labels_dir, label_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path)
        label = Image.open(label_path)

        if image is None:
            raise FileNotFoundError(f"RGB image not found: {image_path}")
        if depth is None:
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        if label is None:
            raise FileNotFoundError(f"Label image not found: {label_path}")

        sample = {'image': image, 'depth': depth, 'label': label}

        if self.split == 'train':
            sample = self.transform_tr(sample)
        elif self.split == 'val':
            sample = self.transform_val(sample)
        elif self.split == 'test':
            sample = self.transform_ts(sample)

        image = sample['image']
        depth = sample['depth']
        label = sample['label']

        # Ensure depth is a single-channel image
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)

        rgbd = torch.cat([image, depth], dim=0)  # 将RGB和深度图像在通道维度上拼接

        return rgbd, label.long()

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.CropBlackArea(),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=1024, crop_size=(256, 512), fill=255),

            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.CropBlackArea(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.CropBlackArea(),
            tr.FixedResize(size=768),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)
