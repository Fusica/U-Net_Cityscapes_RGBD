import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import os
from tqdm import tqdm


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
        else:
            ious.append(0.0)

    return np.mean(ious)


def evaluate(model, data_loader, device, criterion, num_classes):
    model.eval()
    running_loss = 0.0
    iou_total = 0.0
    total_correct_pixels = 0
    total_pixels = 0

    all_preds = []
    all_labels = []

    with tqdm(total=len(data_loader), desc='Validating', leave=False, disable=False) as pbar:
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                iou = calculate_iou(preds, labels, num_classes)
                iou_total += iou

                correct_pixels = (preds == labels).sum().item()
                total_correct_pixels += correct_pixels
                total_pixels += labels.numel()

                all_preds.append(preds.cpu().numpy().flatten())
                all_labels.append(labels.cpu().numpy().flatten())

                pbar.update(1)

    avg_loss = running_loss / len(data_loader)
    pixel_acc = total_correct_pixels / total_pixels
    mIoU = iou_total / len(data_loader)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, mIoU, pixel_acc, precision, recall, f1


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
