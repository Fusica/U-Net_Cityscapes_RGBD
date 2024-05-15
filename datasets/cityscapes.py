import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import matplotlib.pyplot as plt
import numpy as np


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

        image = cv2.imread(image_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"RGB image not found: {image_path}")
        if depth is None:
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        if label is None:
            raise FileNotFoundError(f"Label image not found: {label_path}")

        if self.transform:
            image = self.transform(image)
            depth = cv2.resize(depth, (image.shape[2], image.shape[1]))  # 调整深度图像的大小与RGB图像一致
            label = cv2.resize(label, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST)  # 调整标签图像的大小
            depth = transforms.ToTensor()(depth)  # 将深度图像转换为tensor并增加一个通道维度

        rgbd = torch.cat([image, depth], dim=0)  # 将RGB和深度图像在通道维度上拼接

        return rgbd, torch.from_numpy(label).long()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])
