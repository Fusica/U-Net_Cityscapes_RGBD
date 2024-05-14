import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CityscapesRGBDDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.images_dir = os.path.join(root, 'leftImg8bit', split)
        self.depth_dir = os.path.join(root, 'disparity', split)  # 假设深度图像存储在'disparity'目录下
        self.labels_dir = os.path.join(root, 'gtFine', split)

        self.image_paths = []
        self.depth_paths = []
        self.label_paths = []

        for city in os.listdir(self.images_dir):
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
        image = cv2.imread(self.image_paths[idx])
        depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
            depth = transforms.ToTensor()(depth).unsqueeze(0)  # 将深度图像转换为tensor并增加一个通道维度

        rgbd = torch.cat([image, depth], dim=0)  # 将RGB和深度图像在通道维度上拼接

        return rgbd, torch.from_numpy(label).long()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])
