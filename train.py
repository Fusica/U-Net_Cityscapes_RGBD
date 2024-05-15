import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from datasets.cityscapes import CityscapesRGBDDataset, transform
from models.unet import UNet
from utils.utils import calculate_iou, evaluate, get_run_folder, print_args

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from datasets.cityscapes import CityscapesRGBDDataset, transform
from models.unet import UNet
from utils.utils import calculate_iou, evaluate, get_run_folder, print_args


def train(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if args.use_wandb and rank == 0:
        import wandb
        wandb.init(project="unet-cityscapes-rgbd", config=args)

    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    if rank == 0:
        run_folder = get_run_folder()
    else:
        run_folder = None

    # Broadcast run_folder from rank 0 to all other processes
    run_folder = run_folder if run_folder is not None else 'None'
    run_folder = torch.tensor(list(run_folder.ljust(256)), dtype=torch.int8, device=device)
    dist.broadcast(run_folder, 0)
    run_folder = ''.join(chr(i) for i in run_folder.tolist()).strip()

    if rank == 0:
        print(f"Using GPU: {torch.cuda.get_device_name(device)} (CUDA ID: {rank})")
        print_args(args)
        print(f"Experiment folder: {run_folder}")

    model = UNet(in_channels=4, out_channels=34).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

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

    train_dataset = CityscapesRGBDDataset(root=args.data_path, split='train', transform=transform)
    val_dataset = CityscapesRGBDDataset(root=args.data_path, split='val', transform=transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    if rank == 0:
        writer = SummaryWriter(log_dir=run_folder)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        train_sampler.set_epoch(epoch)
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}', disable=(rank != 0)) as pbar:
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader)
        if rank == 0:
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            if args.use_wandb:
                wandb.log({"Loss/train": epoch_loss})
            print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss:.4f}')

            # Run validation at the end of each epoch
            val_loss, val_miou = evaluate(model, val_loader, device, criterion, num_classes=34)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('mIoU/val', val_miou, epoch)
            if args.use_wandb:
                wandb.log({"Loss/val": val_loss, "mIoU/val": val_miou})
            print(
                f'Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {val_loss:.4f}, Validation mIoU: {val_miou:.4f}')

            # Save the latest model and delete the previous one
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

            # Save the best model and delete the previous one
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

    dist.destroy_process_group()
