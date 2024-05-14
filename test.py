import argparse
import torch
from torch.utils.data import DataLoader
from unet import UNet
from dataset import CityscapesRGBDDataset, transform
import numpy as np

from utils.choose_device import get_device


def calculate_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)

    return np.array(ious)


def test(args):
    device = get_device()

    model = UNet(in_channels=4, out_channels=34)  # Cityscapes有34类，输入通道数改为4
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path))  # 加载训练好的模型参数

    test_dataset = CityscapesRGBDDataset(root=args.data_path, split='val', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    ious = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            ious.append(calculate_iou(preds, labels, num_classes=34))

    miou = np.nanmean(ious)
    print(f'mIoU: {miou:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test U-Net on Cityscapes RGB-D dataset")
    parser.add_argument('--data_path', type=str, default="/Volumes/Data-1T/Datasets/Cityscapes",
                        help="Path to Cityscapes dataset")
    parser.add_argument('--model_path', type=str, required=True, help="Path to saved model")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for testing")

    args = parser.parse_args()
    test(args)
