import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from unet import UNet
from dataset import CityscapesRGBDDataset, transform
import os

from utils.choose_device import get_device


def evaluate(model, data_loader, device, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(data_loader)


def train(args):
    device = get_device()

    model = UNet(in_channels=4, out_channels=34)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"Resuming training from epoch {start_epoch}")

    train_dataset = CityscapesRGBDDataset(root=args.data_path, split='train', transform=transform)
    val_dataset = CityscapesRGBDDataset(root=args.data_path, split='val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    writer = SummaryWriter()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss:.4f}')

        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            val_loss = evaluate(model, val_loader, device, criterion)
            writer.add_scalar('Loss/val', val_loss, epoch)
            print(f'Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {val_loss:.4f}')

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, f'model_checkpoint_epoch_{epoch + 1}.pth')

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on Cityscapes RGB-D dataset")
    parser.add_argument('--data_path', type=str, default="/Volumes/Data-1T/Datasets/Cityscapes",
                        help="Path to Cityscapes dataset")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--resume', type=str, default=None, help="Path to resume training from a checkpoint")

    args = parser.parse_args()
    train(args)
