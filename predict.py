import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from models.unet import UNet  # 导入你的UNet模型


# 去掉state_dict中的“module”前缀
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# 过滤掉特定层的权重
def filter_state_dict(state_dict, filter_keys):
    new_state_dict = {k: v for k, v in state_dict.items() if k not in filter_keys}
    return new_state_dict


# 加载模型
def load_model(model_path, in_channels, out_channels):
    model = UNet(in_channels, out_channels)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = remove_module_prefix(state_dict)
    filter_keys = ["final.weight", "final.bias"]
    state_dict = filter_state_dict(state_dict, filter_keys)
    model.load_state_dict(state_dict, strict=False)  # 允许忽略未加载的层
    model.eval()
    return model


# 预处理RGB和深度图像
def preprocess_images(rgb_image_path, depth_image_path):
    rgb_image = Image.open(rgb_image_path).convert('RGB')
    depth_image = Image.open(depth_image_path).convert('L')  # 深度图像通常为灰度图

    preprocess_rgb = transforms.Compose([
        transforms.Resize((256, 256)),  # 替换为Cityscapes图像的大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preprocess_depth = transforms.Compose([
        transforms.Resize((256, 256)),  # 替换为Cityscapes图像的大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # 假设深度图像的归一化参数
    ])

    rgb_tensor = preprocess_rgb(rgb_image)
    depth_tensor = preprocess_depth(depth_image)

    input_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)  # 拼接RGB和深度图像
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    return input_batch


# 推理
def inference(model, input_batch):
    with torch.no_grad():
        output = model(input_batch)
    return output


# 定义Cityscapes数据集的颜色映射
def get_cityscapes_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]  # 例子颜色，替换为实际的Cityscapes颜色映射
    # 添加其他类别的颜色
    return colormap


# 后处理并显示结果
def postprocess_and_display(output):
    output = output.squeeze(0)  # 移除batch维度
    output = torch.argmax(output, dim=0).cpu().numpy()

    colormap = get_cityscapes_colormap()
    color_output = colormap[output]

    plt.imshow(color_output)
    plt.axis('off')  # 不显示坐标轴
    plt.show()


if __name__ == "__main__":
    model_path = 'path/to/your/model.pt'  # 替换为你的模型路径
    rgb_image_path = 'path/to/your/rgb_image.png'  # 替换为你的RGB图像路径
    depth_image_path = 'path/to/your/depth_image.png'  # 替换为你的深度图像路径

    # 定义模型输入和输出通道数
    in_channels = 4  # RGB (3) + Depth (1)
    out_channels = 19  # Cityscapes数据集的类别数

    model = load_model(model_path, in_channels, out_channels)
    input_batch = preprocess_images(rgb_image_path, depth_image_path)
    output = inference(model, input_batch)
    postprocess_and_display(output)
