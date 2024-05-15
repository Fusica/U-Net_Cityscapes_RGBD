import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import os
import datetime
from tqdm import tqdm


def calculate_iou(preds, labels, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        label_cls = labels == cls
        intersection = (pred_cls & label_cls).sum().float()
        union = (pred_cls | label_cls).sum().float()
        if union == 0:
            ious.append(float('nan'))  # 如果没有该类，跳过计算
        else:
            ious.append((intersection / union).item())
    return np.nanmean(ious)  # 计算去掉nan值的平均


# TODO 检查mIOU的计算是否正确
def evaluate(model, data_loader, device, criterion, num_classes):
    model.eval()
    running_loss = 0.0
    iou_total = 0.0

    with tqdm(total=len(data_loader), desc='Validating', leave=False, disable=False) as pbar:
        with torch.no_grad():
            for images, depth, labels in data_loader:
                images = images.to(device)
                depth = depth.to(device)
                labels = labels.to(device)
                outputs = model(torch.cat([images, depth.unsqueeze(1)], dim=1))
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                iou = calculate_iou(preds, labels, num_classes)
                iou_total += iou

                pbar.update(1)

    avg_loss = running_loss / len(data_loader)
    mIoU = iou_total / len(data_loader)

    return avg_loss, mIoU


def get_run_folder(base_path='runs'):
    run_folder = os.path.join(base_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(run_folder, exist_ok=True)
    return run_folder


def print_args(args):
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
