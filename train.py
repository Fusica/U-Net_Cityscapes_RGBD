import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNet
from dataset import CityscapesRGBDDataset, transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(in_channels=4, out_channels=34)  # Cityscapes有34类，输入通道数改为4
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_dataset = CityscapesRGBDDataset(root='path_to_cityscapes', split='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

num_epochs = 50

for epoch in range(num_epochs):
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

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
