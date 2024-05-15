import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from datasets.cityscapes import CityscapesRGBDDataset, train_transform, val_transform, save_augmented_images
from models.unet import UNet
from utils.utils import evaluate, get_run_folder, print_args


def train(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    run_folder = get_run_folder()
    if args.use_wandb and rank == 0:
        import wandb
        wandb.init(project="unet-cityscapes-rgbd", config=args)

    if args.world_size > 1:
        dist.init_process_group("gloo", rank=rank, world_size=args.world_size)

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if rank == 0:
        print(f"Using device: {device} (Rank: {rank})")
        print_args(args)
        print(f"Experiment folder: {run_folder}")

    model = UNet(in_channels=4, out_channels=34).to(device)
    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank] if torch.cuda.is_available() else None)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler() if torch.cuda.is_available() else None

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_loss = float('inf')
    best_model_path = None
    last_model_path = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        if rank == 0:
            print(f"Resuming training from epoch {start_epoch}")

    train_dataset = CityscapesRGBDDataset(
        root=args.data_path,
        split='train',
        transform=train_transform
    )
    val_dataset = CityscapesRGBDDataset(
        root=args.data_path,
        split='val',
        transform=val_transform
    )

    if args.world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=(val_sampler is None))

    if rank == 0:
        augmented_images_dir = os.path.join(run_folder, 'augmented_images')
        save_augmented_images(train_loader, save_dir=augmented_images_dir, num_images=5)

    if rank == 0:
        writer = SummaryWriter(log_dir=run_folder)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        if args.world_size > 1:
            train_sampler.set_epoch(epoch)
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}', disable=(rank != 0)) as pbar:
            for images, depths, labels in train_loader:
                images, depths, labels = images.to(device), depths.to(device), labels.to(device)

                optimizer.zero_grad()
                if scaler is not None:
                    with autocast():
                        outputs = model(torch.cat([images, depths.unsqueeze(1)], dim=1))
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(torch.cat([images, depths.unsqueeze(1)], dim=1))
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader)
        scheduler.step()

        if rank == 0:
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            if args.use_wandb:
                wandb.log({"Loss/train": epoch_loss})
            print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss:.4f}')

            val_loss, val_miou = evaluate(model, val_loader, device, criterion, num_classes=34)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('mIoU/val', val_miou, epoch)
            if args.use_wandb:
                wandb.log({"Loss/val": val_loss, "mIoU/val": val_miou})
            print(
                f'Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {val_loss:.4f}, Validation mIoU: {val_miou:.4f}')

            current_model_path = os.path.join(run_folder, f'model_checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, current_model_path)
            if last_model_path and os.path.exists(last_model_path):
                os.remove(last_model_path)
            last_model_path = current_model_path

            if val_loss < best_loss:
                best_loss = val_loss
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = os.path.join(run_folder, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)

    if rank == 0:
        writer.close()
        if args.use_wandb:
            wandb.finish()

    if args.world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--data_path', type=str, default="/Volumes/Data-1T/Datasets/Cityscapes", help='path to dataset')
    parser.add_argument('--resume', type=str, default=None, help='path to resume checkpoint')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging')
    parser.add_argument('--world_size', type=int, default=1, help='number of distributed processes')

    args = parser.parse_args()

    if args.world_size > 1:
        mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)
    else:
        train(0, args)
