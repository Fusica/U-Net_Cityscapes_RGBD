import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import wandb
import argparse
from tqdm import tqdm
from datetime import datetime

from models.unet import UNet
from datasets.cityscapes import CityscapesRGBDDataset


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


def evaluate(model, data_loader, device, criterion, num_classes, scaler):
    model.eval()
    running_loss = 0.0
    iou_total = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            iou = calculate_iou(preds, labels, num_classes)
            iou_total += iou

            torch.cuda.empty_cache()  # Empty the cache to free up GPU memory

    return running_loss / len(data_loader), iou_total / len(data_loader)


def train(rank, args, run_dir):
    if args.use_cpu:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("gloo", rank=rank, world_size=args.world_size)
        device = torch.device('cpu')
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)

    print(f"Using device: {device}")

    model = UNet(in_channels=4, out_channels=34).to(device)
    if not args.use_cpu:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    start_epoch = 0
    best_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        if rank == 0:
            print(f"Resuming training from epoch {start_epoch}")

    train_dataset = CityscapesRGBDDataset(root=args.data_path, split='train')
    val_dataset = CityscapesRGBDDataset(root=args.data_path, split='val')

    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, sampler=val_sampler)

    if rank == 0:
        log_dir = os.path.join(run_dir, 'log')
        model_dir = os.path.join(run_dir, 'models')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        train_sampler.set_epoch(epoch)
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}', disable=(rank != 0)) as pbar:
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)  # Convert bytes to MB
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'memory(MB)': memory_allocated})
                pbar.update(1)

                torch.cuda.empty_cache()  # Empty the cache to free up GPU memory

        epoch_loss = running_loss / len(train_loader)
        if rank == 0:
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            if args.use_wandb:
                wandb.log({"Loss/train": epoch_loss})
            print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss:.4f}')

            # Run validation at the end of each epoch
            val_loss, val_miou = evaluate(model, val_loader, device, criterion, num_classes=34, scaler=scaler)
            if rank == 0:
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('mIoU/val', val_miou, epoch)
                if args.use_wandb:
                    wandb.log({"Loss/val": val_loss, "mIoU/val": val_miou})
                print(
                    f'Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {val_loss:.4f}, Validation mIoU: {val_miou:.4f}')

                # Save last checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                }, os.path.join(model_dir, 'last_model.pth'))

                # Save best checkpoint
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))

                # Remove old checkpoints
                for file in os.listdir(model_dir):
                    if file.startswith('model_checkpoint_epoch_'):
                        os.remove(os.path.join(model_dir, file))

        torch.cuda.empty_cache()  # Empty cache on all GPUs

    if rank == 0:
        writer.close()
        if args.use_wandb:
            wandb.finish()

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on Cityscapes RGB-D dataset")
    parser.add_argument('--data_path', type=str, default="/mnt/xwj/datasets/Cityscapes",
                        help="Path to Cityscapes dataset")
    parser.add_argument('--epochs', type=int, default=500, help="Number of epochs to train")
    parser.add_argument('--train_batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--val_batch_size', type=int, default=2, help='validation batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--resume', type=str, default=None, help="Path to resume training from a checkpoint")
    parser.add_argument('--use_wandb', action='store_true', help="Use Weights and Biases for logging")
    parser.add_argument('--world_size', type=int, default=4, help="Number of GPUs to use for DDP training")
    parser.add_argument('--use_cpu', action='store_true', help="Use CPU for training instead of GPU")

    args = parser.parse_args()

    # Create a unique run directory
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join('runs', current_time)
    os.makedirs(run_dir, exist_ok=True)

    mp.spawn(train, nprocs=args.world_size, args=(args, run_dir))


if __name__ == "__main__":
    main()
