import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize

import random
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter


class Resize(object):
    def __init__(self, crop_size_h, crop_size_w):
        # if len(crop_size) == 2:
        self.crop_size_w, self.crop_size_h = crop_size_w, crop_size_h
        # else:


    def __call__(self, sample):

        for key, val in sample.items():
            # if val is not None and len(val.shape) > 1:
            array1 = sample[key]
            array1 = cv2.resize(array1, (self.crop_size_h,self.crop_size_w), interpolation=cv2.INTER_LINEAR)
            sample[key] = array1
        return sample


class RandomCrop(object):
    def __init__(self, crop_size, key = 's'):

        self.crop_size_w, self.crop_size_h = crop_size, crop_size
        self.key  = key
    def __call__(self, sample):
        img = sample[self.key]

        w, h = img.shape[0:2]

        x1 = random.randint(0, w - self.crop_size_w)
        y1 = random.randint(0, h - self.crop_size_h)

        for key, val in sample.items():
            # if val is not None:
            sample[key] = val[x1:x1+self.crop_size_w,y1:y1+self.crop_size_h,...]

        return sample


class Crop(object):
    def __init__(self, crop_size):

        self.crop_size_w, self.crop_size_h = crop_size, crop_size

    def __call__(self, sample):
        img = sample['s']

        w, h = img.shape[0:2]

        x1 = int(round((w - self.crop_size_w) / 2.))
        y1 = int(round((h - self.crop_size_h) / 2.))
        for key, val in sample.items():
            # if val is not None:
            sample[key] = val[x1:x1+self.crop_size_w,y1:y1+self.crop_size_h,...]
        return sample

# class Normalize(object):
#     """Normalize a tensor image with mean and standard deviation.
#     Args:
#         mean (tuple): means for each channel.
#         std (tuple): standard deviations for each channel.
#     """
#     def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.),key='s'):
#         self.mean = mean
#         self.std = std
#         self.key = key
#
#     def __call__(self, sample):
#         img = sample[self.key]
#         # mask = sample['label']
#         img = np.array(img).astype(np.float32)
#         # mask = np.array(mask).astype(np.float32)
#         # img /= 255.0
#         img -= self.mean
#         img /= self.std
#         sample[self.key] = img
#
#         return sample
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.),key='s'):
        self.mean = mean
        self.std = std
        self.key = key

    def __call__(self, sample):
        img = sample[self.key]

        img = normalize(img,self.mean,self.std)

        sample[self.key] = img

        return sample
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample_out = {}
        for key, val in sample.items():
            if val is not None:
                if len(val.shape) > 1:
                    out = np.array(val).astype(np.float32)
                    if key == 's':
                        out = out / 255

                    if len(out.shape) == 2:
                        out = out[..., np.newaxis]
                    out = out.transpose((2, 0, 1))

                    sample_out[key] = torch.from_numpy(out).float()
                else:
                    sample_out[key] = torch.from_numpy(np.array(val))


        return sample_out

def flip_img(img, type = Image.FLIP_LEFT_RIGHT):
    if isinstance(img, np.ndarray):
        img = img[::-1, ...]
    elif img is not None:
        img = img.transpose(type)
    return img


class RandomHorizontalFlip(object):
    def __call__(self, sample):

        if random.random() < 0.5:
            for key, val in sample.items():
                if val is not None and len(val.shape) > 1:
                    sample[key] = flip_img(val)

        return sample


class RandomVertocalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            for key, val in sample.items():
                if val is not None:
                    sample[key] = flip_img(val, type=Image.FLIP_TOP_BOTTOM)


        return sample

def rot_img(img, rotate_degree):
    if isinstance(img, np.ndarray):
        img = np.rot90(img, k=rotate_degree)
    elif img is not None:
        rotate_degree = rotate_degree * 90
        img = img.rotate(rotate_degree)
    return img

class RandomRotate90(object):

    def __call__(self, sample):
        # img = sample['image']
        # mask = sample['label']
        # if 'depth' in sample.keys():
        #     depth = sample['depth']
        # else:
        #     depth = None

        rotate_degree = random.choice([0,1,2])

        # img = rot_img(img,rotate_degree)
        # mask = rot_img(mask,rotate_degree)
        # depth = rot_img(depth,rotate_degree)
        #
        # sample['image'] = img
        # sample['label'] = mask
        # if depth is not None:
        #     sample['depth'] = depth
        for key, val in sample.items():
            if val is not None:
                sample[key] = rot_img(val,rotate_degree)

        return sample

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}

class Lambda(object):
    def __call__(self, sample):
        return sample

class Masked_images(object):
    def __call__(self, sample):
        img = sample['s']
        if 't' in sample.keys():
            key = 't'
        elif 't_0' in sample.keys():
            key = 't_0'
        else:
            raise NotImplemented
        mask = sample[key]

        # if len(mask.shape) == 2:
        #     mask = mask[np.newaxis,...]
        assert mask.shape[0] == 1, 'tensors should already have the form CxHxW'
        # t_out = img * mask[i]
            # t_out.append(t_)
        # t_out = torch.cat(t_out,dim=-1)
        sample[key] = img * mask

        return sample

class RandomScaleCrop(object):
    def __init__(self, crop_size, fill=0,is_random=True):
        if len(crop_size) == 2:
            self.crop_size_w, self.crop_size_h = crop_size
        else:
            self.crop_size_w, self.crop_size_h = crop_size[0], crop_size[0]
        self.fill = fill
        self.is_random = is_random

    def __call__(self, sample):

        scale = random.uniform(0.75,1.75) if self.is_random else 1

        w,h = sample['s'].shape[0:2]
        ow, oh = int(w*scale), int(h*scale)

        if self.is_random:
            x1 = random.randint(0, max(0,ow - self.crop_size_w))
            y1 = random.randint(0, max(0,oh - self.crop_size_h))
        else:
            x1 = max(0,int(round((ow - self.crop_size_w) / 2.)))
            y1 = max(0,int(round((oh - self.crop_size_h) / 2.)))

        # pad crop if necessary
        padh = self.crop_size_h - oh if oh < self.crop_size_h  else 0
        padw = self.crop_size_w - ow if ow < self.crop_size_w else 0

        for key, val in sample.items():
            if val is not None and len(val.shape) > 1:
                array1 = sample[key]
                array1 = cv2.resize(array1, (oh, ow), interpolation=cv2.INTER_LINEAR)
                if len(array1.shape) == 3:
                    array1 = np.pad(array1,((padw,0),(padh,0),(0,0)), mode='constant',constant_values=self.fill)
                else:
                    array1 = np.pad(array1,((padw,0),(padh,0)), mode='constant', constant_values=self.fill)

                array1 = array1[x1:x1+self.crop_size_w,y1:y1+self.crop_size_h,...]
                sample[key] = array1
        return sample




# class RandomCrop(object):
#     def __init__(self, crop_size,is_random=True):
#         if len(crop_size) == 2:
#             self.crop_size_w, self.crop_size_h = crop_size
#         else:
#             self.crop_size_w, self.crop_size_h = crop_size[0], crop_size[0]
#         self.is_random = is_random
#     def __call__(self, sample):
#         img = sample['image']
#         # mask = sample['label']
#         # # random scale (short edge)
#         # short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
#         # w, h = img.size
#         # if h > w:
#         #     ow = short_size
#         #     oh = int(1.0 * h * ow / w)
#         # else:
#         #     oh = short_size
#         #     ow = int(1.0 * w * oh / h)
#         # img = img.resize((ow, oh), Image.BILINEAR)
#         # mask = mask.resize((ow, oh), Image.NEAREST)
#         # pad crop
#         # if short_size < self.crop_size:
#         #     padh = self.crop_size - oh if oh < self.crop_size else 0
#         #     padw = self.crop_size - ow if ow < self.crop_size else 0
#         #     img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
#         #     mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
#
#         # random crop crop_size
#         w, h = img.shape[0:2]
#         # assert  w > self.crop_size and h > self.crop_size, f'input of shape {img.shape} not valid'
#         if self.is_random:
#             x1 = random.randint(0, w - self.crop_size_w)
#             y1 = random.randint(0, h - self.crop_size_h)
#         else:
#             x1 = int(round((w - self.crop_size_w) / 2.))
#             y1 = int(round((h - self.crop_size_h) / 2.))
#         for key, val in sample.items():
#             if val is not None:
#                 sample[key] = val[x1:x1+self.crop_size_w,y1:y1+self.crop_size_h,...]
#                 # assert sample[key].shape[0:2] == (self.crop_size,self.crop_size)
#         return sample
#
#         # img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
#         # mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
#
#             # return {'image': img,
#         #         'label': mask}

class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        # mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}

class ColorJitter(object):
    def __call__(self, sample):

        if random.random() > 0.1:

            img = sample['s']

            # mask = sample['label']
            img = np.array(img).astype(np.float32)
            #
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] += np.random.rand() * 70 - 35
            hsv[:, :, 1] += np.random.rand() * 0.3 - 0.15
            hsv[:, :, 2] += np.random.rand() * 50 - 25
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 360.)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1.)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255.)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.uint8)

            sample['s'] = img

        return sample
