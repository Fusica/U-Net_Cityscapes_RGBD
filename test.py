import torch
from torch.utils.data import DataLoader
from unet import UNet
from dataset import CityscapesRGBDDataset, transform


def test_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # 这里可以添加计算准确率或可视化结果的代码


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(in_channels=4, out_channels=34)  # Cityscapes有34类，输入通道数改为4
model = model.to(device)
model.load_state_dict(torch.load('model.pth'))  # 加载训练好的模型参数

test_dataset = CityscapesRGBDDataset(root='path_to_cityscapes', split='val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

test_model(model, test_loader, device)
