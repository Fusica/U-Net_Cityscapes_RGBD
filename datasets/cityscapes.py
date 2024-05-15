import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from datasets.custom_transform import Normalize, ToTensor, CropBlackArea, RandomHorizontalFlip, RandomScaleCrop


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
        depth = Image.open(depth_path).convert('L')
        label = Image.open(label_path).convert('L')

        sample = {'image': image, 'depth': depth, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['depth'], sample['label'].long()


base_size = 1024
crop_size = 480

train_transform = transforms.Compose([
    CropBlackArea(),
    RandomHorizontalFlip(),
    RandomScaleCrop(base_size=base_size, crop_size=crop_size, fill=255),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensor()
])

val_transform = transforms.Compose([
    CropBlackArea(),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensor()
])

test_transform = transforms.Compose([
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensor()
])


def save_augmented_images(dataloader, save_dir, num_images=5):
    os.makedirs(save_dir, exist_ok=True)
    count = 0

    for batch in dataloader:
        images, depths, labels = batch
        for i in range(images.shape[0]):
            if count >= num_images:
                return
            rgb = images[i].numpy()
            depth = depths[i].numpy()
            label = labels[i].numpy()

            # 反归一化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb = std[:, None, None] * rgb + mean[:, None, None]
            rgb = np.clip(rgb, 0, 1)

            # 保存RGB图像
            plt.imsave(os.path.join(save_dir, f'image_{count}.png'), np.transpose(rgb, (1, 2, 0)))

            # 保存深度图像
            plt.imsave(os.path.join(save_dir, f'depth_{count}.png'), depth, cmap='gray')

            # 保存标签图像
            plt.imsave(os.path.join(save_dir, f'label_{count}.png'), label, cmap='gray')

            count += 1
