# U-Net for Cityscapes RGB-D Segmentation (Currently v0.2 is executable)

This repository contains the implementation of a U-Net model for semantic segmentation on the Cityscapes dataset, incorporating RGB-D data. The project includes training and testing scripts, dataset loading, and the use of `argparse` for parameter tuning, along with TensorBoard and Wandb for monitoring training progress.

Referencing RFNet, the overall code structure was refactored to support the UNet network architecture.

## Project Structure

```
project/
│
├── unet.py # Defines the U-Net model
├── dataset.py # Loads the Cityscapes RGB-D dataset
├── train.py # Trains the U-Net model, saves model, and logs training process
├── test.py # Tests the U-Net model and computes mIoU
└── main.py # Main script to run training and testing
```

## Cityscapes Structure
Make sure your dataset structure same as:
```
Cityscapes/
│
├── leftImg8bit/               # RGB images
│   ├── train/
│   │   ├── aachen/
│   │   │   ├── aachen_000000_000019_leftImg8bit.png
│   │   │   ├── ...
│   │   ├── ...
│   ├── val/
│   │   ├── frankfurt/
│   │   │   ├── frankfurt_000000_000294_leftImg8bit.png
│   │   │   ├── ...
│   │   ├── ...
│   ├── test/
│       ├── ...
│
├── disparity/                 # Depth images (disparity maps)
│   ├── train/
│   │   ├── aachen/
│   │   │   ├── aachen_000000_000019_disparity.png
│   │   │   ├── ...
│   │   ├── ...
│   ├── val/
│   │   ├── frankfurt/
│   │   │   ├── frankfurt_000000_000294_disparity.png
│   │   │   ├── ...
│   │   ├── ...
│   ├── test/
│       ├── ...
│
├── gtFine/                    # Ground truth labels
│   ├── train/
│   │   ├── aachen/
│   │   │   ├── aachen_000000_000019_gtFine_labelIds.png
│   │   │   ├── ...
│   │   ├── ...
│   ├── val/
│   │   ├── frankfurt/
│   │   │   ├── frankfurt_000000_000294_gtFine_labelIds.png
│   │   │   ├── ...
│   │   ├── ...
│   ├── test/
│       ├── ...
```

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
```


## Getting Started
### Train
To train the U-Net model, use the following command:
```bash
python train.py --data_path path_to_cityscapes --epochs 50 --batch_size 4 --lr 1e-4
```

### Validation
To validate the U-Net model, use the following command:
```bash
python eval.py --data_path path_to_cityscapes
```
