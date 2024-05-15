import torch
import numpy as np

import os


def calculate_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union != 0:
            ious.append(float(intersection) / union)

    return np.mean(ious)


def evaluate(model, data_loader, device, criterion, num_classes):
    model.eval()
    running_loss = 0.0
    iou_total = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate IoU
            _, preds = torch.max(outputs, 1)
            iou = calculate_iou(preds, labels, num_classes)
            iou_total += iou

    return running_loss / len(data_loader), iou_total / len(data_loader)


def get_run_folder(base_path='runs'):
    run_id = 1
    while os.path.exists(os.path.join(base_path, f'run{run_id}')):
        run_id += 1
    run_folder = os.path.join(base_path, f'run{run_id}')
    os.makedirs(run_folder)
    return run_folder


def print_args(args):
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
