import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize
from torchvision.transforms import functional as F

import random
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter

class Resize(transforms.Resize):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """

        for key, val in sample.items():
            # if val is not None and len(val.shape) > 1:
            img = sample[key]

            array1 = F.resize(img, self.size, self.interpolation)

            sample[key] = array1
            return sample


class ResizeNoCrop(transforms.Resize):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """

        scale = self.size / max(sample['s'].size)
        for key, val in sample.items():
            # if val is not None and len(val.shape) > 1:
            img = sample[key]

            array0 = F.resize(img, (int(scale * img.size[1]), int(scale * img.size[0])), self.interpolation)

            if img.mode=="RGB":
                array1 = Image.new('RGB', (self.size, self.size), (0, 0, 0))
            else:
                array1 = Image.new('L', (self.size, self.size), (0,))

            array1.paste(array0, (int((self.size - array0.size[0]) / 2), int((self.size - array0.size[1]) / 2)))

            sample[key] = array1

        return sample



class RandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, sample):
        """
        Args:
            sample (dict of PIL Images): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """

        img = sample['s']
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        for key, val in sample.items():
            # if val is not None and len(val.shape) > 1:
            img = sample[key]
            array1 = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
            sample[key] = array1
        return sample

class CenterCrop(transforms.CenterCrop):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __call__(self, sample):

        for key, val in sample.items():
            # if val is not None and len(val.shape) > 1:
            img = sample[key]
            array1 = F.center_crop(img, self.size)
            sample[key] = array1
        return sample




class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self,*args, ** kwargs):
        self.is_birds_flipping = kwargs.pop('is_birds_flipping') if 'is_birds_flipping' in kwargs.keys() else False
        self.is_cct_flipping = kwargs.pop('is_cct_flipping') if 'is_cct_flipping' in kwargs.keys() else False
        super().__init__(*args, ** kwargs)

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            for key, val in sample.items():
                # if val is not None and len(val.shape) > 1:
                img = sample[key]
                array1 = F.hflip(img)
                sample[key] = array1

            if self.is_birds_flipping:
                sample = self.swapLeftRightBirds(sample)
            elif self.is_cct_flipping:
                sample = self.swapLeftRightCCT(sample)

        return sample

    def swapLeftRightBirds(self, sample):

        # eye
        if ('t_6' in sample) and ('t_10' in sample):
            l, r = sample['t_6'], sample['t_10']
            sample['t_6'], sample['t_10'] = r, l

        # legs
        if ('t_7' in sample) and ('t_11' in sample):
            l, r = sample['t_7'], sample['t_11']
            sample['t_7'], sample['t_11'] = r, l

        # wings
        if ('t_8' in sample) and ('t_12' in sample):
            l, r = sample['t_8'], sample['t_12']
            sample['t_8'], sample['t_12'] = r, l

        return sample

    def swapLeftRightCCT(self, sample):

        # front-leg
        if ('t_1' in sample) and ('t_2' in sample):
            l, r = sample['t_1'], sample['t_2']
            sample['t_1'], sample['t_2'] = r, l

        # back-leg
        if ('t_3' in sample) and ('t_4' in sample):
            l, r = sample['t_3'], sample['t_4']
            sample['t_3'], sample['t_4'] = r, l

        return sample

class ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        for key, val in sample.items():
            # if val is not None and len(val.shape) > 1:
            img = sample[key]
            array1 = F.to_tensor(img)
            sample[key] = array1
        return sample



class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, key):
        self.mean = mean
        self.std = std
        self.key = key

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        sample[self.key] = F.normalize(sample[self.key], self.mean, self.std)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



def transform_tr_pil(sample):
    composed_transforms = transforms.Compose([
        RandomResizedCrop(448, scale=[0.5, 1]),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), key='s'),
    ])

    return composed_transforms(sample)


def transform_val_pil(sample):
    composed_transforms = transforms.Compose([
        Resize(448),
        CenterCrop(448),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), key='s'),
    ])

    return composed_transforms(sample)

