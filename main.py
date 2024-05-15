import argparse
import torch.multiprocessing as mp
from train import train
from test import test

def main():
    parser = argparse.ArgumentParser(description="Train or Test U-Net on Cityscapes RGB-D dataset")
    parser.add_argument('mode', choices=['train', 'test'], help="Mode to run: train or test")
    parser.add_argument('--data_path', type=str, default="/home/server/xwj/datasets/Cityscapes",
                        help="Path to Cityscapes dataset")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=12, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--resume', type=str, default=None, help="Path to resume training from a checkpoint")
    parser.add_argument('--use_wandb', action='store_true', help="Use Weights and Biases for logging")
    parser.add_argument('--world_size', type=int, default=2, help="Number of GPUs to use for DDP training")
    parser.add_argument('--model_path', type=str, default='best_model.pth', help="Path to the model file for testing")

    args = parser.parse_args()

    if args.mode == 'train':
        mp.spawn(train, nprocs=args.world_size, args=(args,))
    elif args.mode == 'test':
        test(args)

if __name__ == "__main__":
    main()
