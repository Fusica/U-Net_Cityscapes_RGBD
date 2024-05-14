import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training and testing")
    parser.add_argument('--data_path', type=str, required=True, help="Path to Cityscapes dataset")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training and testing")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--test_model', type=str, default=None, help="Path to saved model for testing")

    args = parser.parse_args()

    if not args.test_model:
        print("Training the model...")
        os.system(f'python train.py --data_path {args.data_path} --epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr}')
    print("Testing the model...")
    os.system(f'python test.py --data_path {args.data_path} --model_path {args.test_model if args.test_model else "model_epoch_50.pth"} --batch_size {args.batch_size}')
