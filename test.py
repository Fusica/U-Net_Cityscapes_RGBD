import torch
from torch.utils.data import DataLoader
import numpy as np

from models.unet import UNet
from datasets.cityscapes import CityscapesRGBDDataset, transform
from utils.utils import calculate_iou


def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(in_channels=4, out_channels=34).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

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
