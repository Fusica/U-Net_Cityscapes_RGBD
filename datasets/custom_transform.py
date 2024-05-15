import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        depth = np.array(depth).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        mean_depth = 0.12176
        std_depth = 0.09752
        depth /= 255.0
        depth -= mean_depth
        depth /= std_depth

        return {'image': img, 'depth': depth, 'label': mask}


class ToTensor(object):
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        depth = np.array(depth).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img = torch.from_numpy(img).float()
        depth = torch.from_numpy(depth).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img, 'depth': depth, 'label': mask}


class CropBlackArea(object):
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        if depth.mode != 'L':
            depth = depth.convert('L')
        width, height = img.size
        left = 140
        top = 30
        right = 2030
        bottom = 900
        img = img.crop((left, top, right, bottom))
        depth = depth.crop((left, top, right, bottom))
        mask = mask.crop((left, top, right, bottom))
        img = img.resize((width, height), Image.BILINEAR)
        depth = depth.resize((width, height), Image.BILINEAR)
        mask = mask.resize((width, height), Image.NEAREST)

        return {'image': img, 'depth': depth, 'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img, 'depth': depth, 'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        rotate_degree = random.uniform(-self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        depth = depth.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img, 'depth': depth, 'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return {'image': img, 'depth': depth, 'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        depth = depth.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        if short_size < min(self.crop_size):
            padw = self.crop_size[0] - ow if ow < self.crop_size[0] else 0
            padh = self.crop_size[1] - oh if oh < self.crop_size[1] else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            depth = ImageOps.expand(depth, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)

        w, h = img.size
        x1 = random.randint(0, w - self.crop_size[0])
        y1 = random.randint(0, h - self.crop_size[1])
        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        depth = depth.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        mask = mask.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        return {'image': img, 'depth': depth, 'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        depth = depth.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img, 'depth': depth, 'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        assert img.size == depth.size == mask.size
        img = img.resize(self.size, Image.BILINEAR)
        depth = depth.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img, 'depth': depth, 'label': mask}


class Relabel(object):
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        tensor[tensor == self.olabel] = self.nlabel
        return tensor
