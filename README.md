# U-Net for Cityscapes RGB-D Segmentation

This repository contains the implementation of a U-Net model for semantic segmentation on the Cityscapes dataset, incorporating RGB-D data. The project includes training and testing scripts, dataset loading, and the use of `argparse` for parameter tuning, along with TensorBoard for monitoring training progress.

## Project Structure

project/
│
├── unet.py # Defines the U-Net model
├── dataset.py # Loads the Cityscapes RGB-D dataset
├── train.py # Trains the U-Net model, saves model, and logs training process
├── test.py # Tests the U-Net model and computes mIoU
└── main.py # Main script to run training and testing

## Prerequisites

- Python 3.7+
- PyTorch
- Torchvision
- OpenCV
- TQDM
- TensorBoard

Install the required packages using pip:
```bash
pip install torch torchvision tqdm opencv-python tensorboard


Usage
Training
To train the U-Net model, use the following command:
python main.py --data_path path_to_cityscapes --epochs 50 --batch_size 4 --lr 1e-4

Testing
To test the U-Net model, use the following command:
python main.py --data_path path_to_cityscapes --test_only
