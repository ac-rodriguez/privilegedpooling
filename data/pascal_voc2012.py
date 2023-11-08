# from __future__ import print_function

from PIL import Image, ImageDraw
from os.path import join
import os
import scipy.io
import numpy as np
from itertools import compress
import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url, list_dir, list_files
import torchvision.transforms as transforms
import data.custom_transforms as tr
import csv
import collections
import xml.etree.ElementTree as ET
import glob

class VOCDetection(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    folder = 'VOC2012'
    download_url_prefix = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.ta'

    def __init__(self,
                 root,
                 classes_train='all',
                 datasplit='train',
                 args=None,
                 trainvalindex=None,
                 x_star='keypoints',
                 download=False,
                 crop_size=(300,300)):
        super(VOCDetection, self).__init__()

        self.root = join(os.path.expanduser(root), self.folder)
        self.args = args
        self.x_star = x_star
        self.classes_train = classes_train

        if args is None:
            self.data_fraction = 1
            self.every_n_is_val = 3

        else:
            self.data_fraction = self.args.train_data_fraction
            self.every_n_is_val = self.args.every_n_is_val
        self.is_masked_images = 'masked' in self.x_star
        self.datasplit=datasplit
        self.crop_size = crop_size[0]
        assert crop_size[0] == crop_size[1]

        self.is_debug = False


        split = self.load_split()

        if self.datasplit in ['train','val']:
            if trainvalindex is not None:
                assert len(split) == len(trainvalindex)
                split = list(compress(split,trainvalindex))
            # take every n image for validation
            valindex = range(len(split))
            valindex = np.mod(valindex,self.every_n_is_val) == 0

            if self.datasplit == 'val':
                split = list(compress(split, valindex))
            # else:
            #     split = list(compress(split, ~valindex))

        image_dir = os.path.join(self.root, 'JPEGImages')
        annotation_dir = os.path.join(self.root, 'Annotations')
        mask_dir = os.path.join(self.root, 'SegmentationClass')


        self.images_folder = join(self.root, 'JPEGImages')
        self.annotations_folder = join(self.root, 'Annotations')
        if self.classes_train == 'all':
            self.images = [os.path.join(image_dir, x + ".jpg") for x in split]
            self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in split]
            self.masks = [os.path.join(mask_dir, x + ".png") for x in split]

        else:
            split = [x.split(' ',1) for x in split]

            # remove unsure
            split = [x for x in split if int(x[1]) != 0]
            self.images = [os.path.join(image_dir, x[0] + ".jpg") for x in split]
            self.annotations = [os.path.join(annotation_dir, x[0] + ".xml") for x in split]
            self.masks = [os.path.join(mask_dir, x[0] + ".png") for x in split]

            self.labels = [max(0,int(x[1]))  for x in split] # classes to 0 and 1

            # print(np.unique(self.labels))

        self.classes = ['aeroplane',
                        'bicycle',
                        'bird',
                        'boat',
                        'bottle',
                        'bus',
                        'car',
                        'cat',
                        'chair',
                        'cow',
                        'diningtable',
                        'dog',
                        'horse',
                        'motorbike',
                        'person',
                        'pottedplant',
                        'sheep',
                        'sofa',
                        'train',
                        'tvmonitor']

    def __len__(self):
        if self.is_debug:
            return 1000
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        dict_out = {}

        image = Image.open(self.images[index]).convert('RGB')

        tree = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        object_ = tree['annotation']['object']
        if isinstance(object_, list):
            names_ = [x['name'] for x in object_]
            # assert len(set(names_)) == 1
            object_ = object_[0]
        if self.classes_train == 'all':
            target_class = self.classes.index(object_['name'])
        else:
            target_class = self.labels[index]

        def get_bbox(bbox):
            bbox = list(map(int, [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]))

            # creating new Image object
            bbox_mask = Image.new("L", image.size)

            # create rectangle image
            img1 = ImageDraw.Draw(bbox_mask)
            img1.rectangle(bbox, fill=1)
            return np.array(bbox_mask)

        if self.x_star == 'bbox':
            dict_out['t'] = get_bbox(object_['bndbox'])

        elif 'keypoints' in self.x_star:

            if os.path.isfile(self.masks[index]):
                mask_ = Image.open(self.masks[index]).convert('RGB')
                mask_ = np.array(mask_)

                enc_mask = self.encode_segmap(mask_) - 1

                if self.classes_train == 'all':
                    mask1 = enc_mask == target_class
                else:
                    mask1 = enc_mask == 3

                if mask1.any():

                    cols = np.sum(mask1, axis=0)
                    y1 = next((id for id, val in enumerate(cols) if val > 0), 0)
                    x1 = np.argmax(mask1[:, y1])

                    y2 = next((id for id, val in reversed(list(enumerate(cols))) if val > 0), 0)
                    x2 = np.argmax(mask1[:, y2])

                    rows = np.sum(mask1, axis=1)

                    x3 = next((id for id, val in enumerate(rows) if val > 0), 0)
                    y3 = np.argmax(mask1[x3,:])

                    x4 = next((id for id, val in reversed(list(enumerate(rows))) if val > 0), 0)
                    y4 = np.argmax(mask1[x4,:])

                    x5 = mask1 * np.arange(0, mask1.shape[0])[..., np.newaxis]
                    x5 = x5[x5 > 0].mean()
                    y5 = mask1 * np.arange(0, mask1.shape[1])[np.newaxis,...]
                    y5 = y5[y5 > 0].mean()

                    parts = {1:(y1,x1),
                             2:(y2,x2),
                             3: (y3, x3),
                             4: (y4, x4),
                             5: (y5, x5)}
                else:
                    parts = {}
            else:
                parts = {}
            # parts = self.part_locs_dict[id_]

            size_circle = 10 #10 # pixels
            mask_out = []
            for i in range(1,6): # loop over all the parts
                # creating new Image object
                bbox_mask = Image.new("L", image.size)
                if i in parts.keys():
                    x, y = parts[i]
                    x1,y1 = x-size_circle,y-size_circle
                    x2,y2 = x+size_circle,y+size_circle
                    circle = (x1,y1,x2,y2)

                    # create rectangle image
                    img1 = ImageDraw.Draw(bbox_mask)
                    img1.ellipse(circle, fill=1)
                mask_out.append(np.array(bbox_mask))
            if 'bbox' in self.x_star:
                mask_out.append(get_bbox(object_['bndbox']))

            dict_out['t'] = np.stack(mask_out, axis=-1)

            if self.is_masked_images:
                dict_out['t'] = np.array(dict_out['t'].sum(axis=-1) > 0,dtype=np.float32)[...,np.newaxis]


        dict_out['s'] = np.array(image)

        if self.datasplit == 'train':
            dict_out = self.transform_tr(dict_out)
        else:
            dict_out = self.transform_val(dict_out)

        dict_out['label'] = torch.from_numpy(np.array(target_class))

        return dict_out


    @staticmethod
    def get_boxes(path):

        e = ET.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def load_split(self):

        if self.classes_train == 'all':
            prefix_ = ''
        elif self.classes_train == 'boats':
            prefix_ = 'boat_'
        else:
            raise NotImplementedError

        splits_dir = os.path.join(self.root, 'ImageSets/Main')

        if self.datasplit in ['train','val']:
            # we use the trainset as trainval
            split_f = os.path.join(splits_dir, prefix_+'train.txt')
        else:
            split_f = os.path.join(splits_dir, prefix_+ 'val.txt')


        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        return file_names

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.Resize(600, 600),
            tr.Resize(512, 512),
            tr.RandomHorizontalFlip(),
            tr.RandomCrop(crop_size=self.crop_size),
            tr.ColorJitter(),
            tr.ToTensor(),
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
            tr.Masked_images() if self.is_masked_images else tr.Lambda(),
        ])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.Resize(600, 600),
            tr.Resize(512, 512),
            tr.Crop(crop_size=self.crop_size),
            tr.ToTensor(),
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
        ])
        return composed_transforms(sample)


    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        # if plot:
        #     plt.imshow(rgb)
        #     plt.show()
        # else:
        return rgb

    def stats(self):
        counts = {}
        for index in range(len(self.images)):
            tree = self.parse_voc_xml(
                ET.parse(self.annotations[index]).getroot())
            object_ = tree['annotation']['object']
            if isinstance(object_,list):
                names_ = [x['name'] for x in object_]
                if len(set(names_)) > 1:
                    assert not ['boat'] in names_
                object_ = object_[0]
            target_class = self.classes.index(object_['name'])

            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print(f"{self.datasplit}: {len(self.images)} samples spanning {len(counts.keys())} classes "
              f"(avg {float(len(self.images))/float(len(counts.keys())):.2f} per class)")

        return counts


class BiasedBoats(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    folder = 'boat_and_sea'

    def __init__(self,
                 root,
                 datasplit='test',
                 args=None,
                 nclass=20,
                 trainvalindex=None,
                 x_star='keypoints',
                 download=False,
                 crop_size=(300,300)):
        super(BiasedBoats, self).__init__()

        self.root = join(os.path.expanduser(root), self.folder)
        self.args = args
        self.x_star = x_star

        # if args is None:
        #     self.data_fraction = 1
        #     self.every_n_is_val = 3
        #
        # else:
        #     self.data_fraction = self.args.train_data_fraction
        #     self.every_n_is_val = self.args.every_n_is_val
        # self.is_masked_images = 'masked' in self.x_star
        self.datasplit=datasplit
        self.crop_size = crop_size[0]
        assert crop_size[0] == crop_size[1]

        self.is_debug = False

        boats_ = glob.glob(self.root + '/boat/*.jp*g')
        sea_ = glob.glob(self.root + '/sea/*.jp*g')

        self.images = boats_ + sea_
        if nclass == 20:
            self.annotations = [3] * len(boats_) + [20] * len(sea_)
        else:
            self.annotations = [1] * len(boats_) + [0] * len(sea_)

        self.classes = ['aeroplane',
                        'bicycle',
                        'bird',
                        'boat',
                        'bottle',
                        'bus',
                        'car',
                        'cat',
                        'chair',
                        'cow',
                        'diningtable',
                        'dog',
                        'horse',
                        'motorbike',
                        'person',
                        'pottedplant',
                        'sheep',
                        'sofa',
                        'train',
                        'tvmonitor']

    def __len__(self):
        if self.is_debug:
            return 1000
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        dict_out = {}

        image = Image.open(self.images[index]).convert('RGB')

        # tree = self.parse_voc_xml(
        #     ET.parse(self.annotations[index]).getroot())
        #
        # object_ = tree['annotation']['object']
        # if isinstance(object_, list):
        #     names_ = [x['name'] for x in object_]
        #     # assert len(set(names_)) == 1
        #     object_ = object_[0]

        target_class = self.annotations[index]

        # if self.x_star == 'bbox':
        #
        #     bbox = object_['bndbox']
        #     bbox = list(map(int, [bbox['xmin'],bbox['ymin'], bbox['xmax'],bbox['ymax']]))
        #
        #     # creating new Image object
        #     bbox_mask = Image.new("L", image.size)
        #
        #     # create rectangle image
        #     img1 = ImageDraw.Draw(bbox_mask)
        #     img1.rectangle(bbox, fill=255)
        #     dict_out['t'] = np.array(bbox_mask)
        #
        dict_out['s'] = np.array(image)

        # if self.datasplit == 'train':
        #     dict_out = self.transform_tr(dict_out)
        # else:
        dict_out = self.transform_val(dict_out)
        dict_out['label'] = torch.from_numpy(np.array(target_class))

        return dict_out



    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.Resize(600, 600),
            tr.Resize(512, 512),
            tr.Crop(crop_size=self.crop_size),
            tr.ToTensor(),
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
        ])
        return composed_transforms(sample)

    # def stats(self):
    #     counts = {}
    #     for index in range(len(self.images)):
    #         tree = self.parse_voc_xml(
    #             ET.parse(self.annotations[index]).getroot())
    #         object_ = tree['annotation']['object']
    #         if isinstance(object_,list):
    #             names_ = [x['name'] for x in object_]
    #             if len(set(names_)) > 1:
    #                 assert not ['boat'] in names_
    #             object_ = object_[0]
    #         target_class = self.classes.index(object_['name'])
    #
    #         if target_class not in counts.keys():
    #             counts[target_class] = 1
    #         else:
    #             counts[target_class] += 1
    #
    #     print(f"{self.datasplit}: {len(self.images)} samples spanning {len(counts.keys())} classes "
    #           f"(avg {float(len(self.images))/float(len(counts.keys())):.2f} per class)")
    #
    #     return counts


if __name__ == "__main__":

    voc = VOCDetection(root='/scratch/data',x_star='keypoints', datasplit='train', classes_train='boats')

    for i in range(100):
        a = voc.__getitem__(i)

    a1 = voc.__getitem__(5)

    boats = BiasedBoats(root='/scratch/data')

    b = boats.__getitem__(5)
    b1 = boats.__getitem__(10)

    print('done!')
