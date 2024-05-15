import torch
from torch.utils.data import DataLoader

from models.unet import UNet
from datasets.cityscapes import CityscapesRGBDDataset
from utils.utils import evaluate


def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = UNet(in_channels=4, out_channels=34).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 加载测试数据集
    test_dataset = CityscapesRGBDDataset(root=args.data_path, split='val')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 评估模型
    criterion = torch.nn.CrossEntropyLoss()
    avg_loss, miou, pixel_acc, precision, recall, f1 = evaluate(model, test_loader, device, criterion, num_classes=34)

    print(f'mIoU: {miou:.4f}')
    print(f'Pixel Accuracy: {pixel_acc:.4f}')
    print(f'Mean Pixel Accuracy: {precision:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
