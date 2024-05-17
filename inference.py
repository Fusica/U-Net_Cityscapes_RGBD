import os

import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
import time

from dataloaders.utils import Colorize
from models.unet import UNet
from models.unet_rgbd.unet_rgbd import UNet_RGBD


def image_merge(image_root, label, save_name):
    image = Image.open(image_root)
    width, height = image.size

    # resize
    image = image.resize(label.size, Image.BILINEAR)

    image = image.convert('RGBA')
    label = label.convert('RGBA')
    image = Image.blend(image, label, 0.6)
    image.save(save_name)


def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print('{} not in model_state'.format(name))
            continue
        else:
            own_state[name].copy_(param)

    return model


unet = UNet_RGBD(use_bn=True)
model = UNet(unet, num_classes=19, use_bn=True)

new_state_dict = torch.load('last.pth', map_location=torch.device('cpu'))

print("Loading model...")
model = load_my_state_dict(model, new_state_dict['state_dict'])
print("Model loaded.")

# Load image
image_path = "/Volumes/Data-1T/Datasets/Cityscapes/leftImg8bit/test/bielefeld/bielefeld_000000_053779_leftImg8bit.png"
print("Loading image...")
image = Image.open(image_path)
image_tensor = ToTensor()(image).unsqueeze(0)
print("Image loaded.")

print("Running inference...")
start_time = time.time()
output = model(image_tensor)
end_time = time.time()
print("Inference time: {}".format(end_time - start_time))

# Colorize and save
pre_colors = Colorize()(torch.max(output, 1)[1].detach().cpu().byte())

for i in range(pre_colors.shape[0]):
    pre_color_image = ToPILImage()(pre_colors[i])

    output_path = os.path.join("result", os.path.basename(image_path))
    merge_output_path = os.path.join("result", 'merge_' + os.path.basename(image_path))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(merge_output_path), exist_ok=True)

    pre_color_image.save(output_path)
    image_merge(image_path, pre_color_image, merge_output_path)

    print('Saved merged image: {}'.format(merge_output_path))
