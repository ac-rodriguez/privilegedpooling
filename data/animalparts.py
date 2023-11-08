import glob, os, sys
import tarfile


# from __future__ import print_function

from PIL import Image, ImageDraw
from os.path import join
import os

import csv
import numpy as np
from itertools import compress
import torch.utils.data as data
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms
import torch
import data.custom_transforms as tr

class AnimalParts(data.Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'ImageNet'

    def __init__(self,
                 root,
                 datasplit='train',
                 args=None,
                 # cropped=False,
                 trainvalindex=None,
                 x_star='keypoints',
                 transform=None,
                 target_transform=None,
                 crop_size=(300, 300)):
        super(AnimalParts, self).__init__()

        self.root = join(os.path.expanduser(root), self.folder)
        self.args = args
        self.x_star = x_star

        if args is None:
            self.data_fraction = 1
            self.every_n_is_val = 3
        else:
            self.data_fraction = self.args.train_data_fraction
            self.every_n_is_val = self.args.every_n_is_val

        self.is_masked_images = 'masked' in self.x_star
        self.datasplit=datasplit
        self.crop_size = crop_size
        # self.cropped = cropped
        self.is_debug = False
        self.transform = transform
        self.target_transform = target_transform

        imgnet_clases = {}

        with open(self.root + '/map_clsloc.txt') as f:
            for line in f:
                line = line.split(' ')
                imgnet_clases[line[0]] = (int(line[1]), line[2].strip())
        self.imgnet_classes = imgnet_clases

        class_file = self.root + '/animalPartsFinal/classMapWNID.txt'

        classes = []
        with open(class_file) as f:
            index = 0
            for line in f:
                if index > 0:  # skip header
                    line_ = line.split(';')
                    classes.append(line_[1].strip())
                index += 1
        self.classes = classes
        use_intersection= True
        use_val_for_train = True


        if self.datasplit in ['train','val']:
            xml_foot = glob.glob(self.root + '/animalPartsFinal/xml/foot/n*.xml')
            xml_eyes = glob.glob(self.root + '/animalPartsFinal/xml/eye/n*.xml')

            xml_foot = [os.path.basename(x).strip('.xml') for x in xml_foot]
            xml_eyes = [os.path.basename(x).strip('.xml') for x in xml_eyes]

            if use_intersection:
                image_list = list(sorted(set(xml_eyes).intersection(xml_foot)))
            else:
                image_list = list(sorted(set(xml_eyes).union(xml_foot)))


            if trainvalindex is not None:
                assert len(image_list) == len(trainvalindex)
                image_list = list(compress(image_list, trainvalindex))
            # take every n image for validation
            valindex = range(len(image_list))
            valindex = np.mod(valindex, self.every_n_is_val) == 0
            if self.datasplit == 'val':
                image_list = list(compress(image_list, valindex))
            elif not use_val_for_train:
                image_list = list(compress(image_list, ~valindex))

        else:
            xml_foot = glob.glob(self.root + '/animalPartsFinal/xml/foot/ILSVRC2012_val*.xml')
            xml_eyes = glob.glob(self.root + '/animalPartsFinal/xml/eye/ILSVRC2012_val*.xml')

            xml_foot = [os.path.basename(x).strip('.xml') for x in xml_foot]
            xml_eyes = [os.path.basename(x).strip('.xml') for x in xml_eyes]

            if use_intersection:
                image_list = list(sorted(set(xml_eyes).intersection(xml_foot)))
            else:
                image_list = list(sorted(set(xml_eyes).union(xml_foot)))

            self.val_labels = get_val_labels(self.root)

        images = {}
        for name in image_list:

            eyes = self.root + '/animalPartsFinal/xml/eye/'+name+'.xml'
            foot = self.root + '/animalPartsFinal/xml/foot/'+name+'.xml'

            images[name] = (eyes,foot)

        print(len(images.keys()))


        self.images_parts = images
        self.flat_images = image_list



    def __len__(self):
        if self.is_debug:
            return 1000
        return len(self.flat_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
#        image_name, target_class = self._flat_breed_images[index]

        dict_out = {}

        id_ = self.flat_images[index]

        class_, target_class = self.get_class(id_)

        if self.datasplit in ['train','val']:
            image_path = join(self.root,'train',class_, id_+'.JPEG')
        else:
            image_path = join(self.root,'val','val_images',id_+'.JPEG')
        image = Image.open(image_path).convert('RGB')

        size_circle = 10 # pixels

        if 'bbox' in self.x_star:
            filename_ = self.images_parts[id_][0]
            if not os.path.isfile(filename_):
                filename_ = self.images_parts[id_][1]
                if not os.path.isfile(filename_):
                    raise AssertionError(filename_)

            bbox_, _ = self.get_boxes(filename_)

            # creating new Image object
            bbox_mask = Image.new("L", image.size)

            # create rectangle image
            img1 = ImageDraw.Draw(bbox_mask)
            img1.rectangle(bbox_[0], fill=1)
            dict_out['t'] = np.array(bbox_mask)

        elif 'keypoints' in self.x_star:
            mask_out = []
            for filename_ in self.images_parts[id_]:
                bbox_mask = Image.new("L", image.size)
                if os.path.isfile(filename_):
                    bbox_, part_loc = self.get_boxes(filename_)

                    if not np.isnan(part_loc).any():
                        x,y = part_loc[0]

                        x1,y1 = x-size_circle,y-size_circle
                        x2,y2 = x+size_circle,y+size_circle
                        circle = (int(x1),int(y1),int(x2),int(y2))
                        # create rectangle image
                        img1 = ImageDraw.Draw(bbox_mask)
                        img1.ellipse(circle, fill=1)
                mask_out.append(np.array(bbox_mask))

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
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        points = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
            points.append([float(objs.find('point').find('x').text),
                           float(objs.find('point').find('y').text)])
        return (boxes,points)

    def get_class(self, id_):
        if self.datasplit in ['train','val']:
            name_ =  id_.split('_')[0]
        elif self.datasplit == 'test':
            name_ = self.val_labels[id_]
        class_id = self.classes.index(name_)
        return name_, class_id



    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(600, 600),
            tr.RandomHorizontalFlip(),
            tr.RandomCrop(crop_size=448),
            tr.ColorJitter(),
            tr.ToTensor(),
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
            tr.Masked_images() if self.is_masked_images else tr.Lambda(),
            ])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.Resize(600, 600),
            tr.Crop(crop_size=448),
            tr.ToTensor(),
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        ])
        return composed_transforms(sample)

    def stats(self):
        counts = {}
        for id_ in self.flat_images:
            target_class, _ = self.get_class(id_)
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print(f"{self.datasplit}: {len(self.flat_images)} samples spanning {len(counts.keys())} classes "
              f"(avg {float(len(self.flat_images))/float(len(counts.keys())):.2f} per class)")
        return counts

def get_val_labels(root):

    val_file = root+'/caffe_ilsvrc12/val.txt'
    val_id = np.loadtxt(val_file, str)

    mapping_file = root+'/caffe_ilsvrc12/synset_words.txt'
    with open(mapping_file) as f:
        mapping_list = [line.split(' ')[0] for line in f]

    val_dict = {}
    for file_,class_ in val_id:
        file1 = file_.replace('.JPEG','')
        val_dict[file1] = mapping_list[int(class_)]

    return val_dict



def extract_data():
    class_file = '/scratch/data/imageNetmain/animalPartsFinal/animalPartsFinal/classMapWNID.txt'

    org_class = []
    # Build dictionary of indices to classes
    with open(class_file) as f:
        index = 0
        for line in f:
            if index > 0:  # skip header
                line_ = line.split(';')
                org_class.append(line_[1].strip())
                # class_name = line.split(';')[1].strip()

                # class_to_index[class_name] = index
            index += 1
    # class_to_index = class_to_index

    print(org_class)


    imgnet_folder = '/scratch/data/imageNet_train'

    dest_folder = '/scratch/data/imageNetmain/train'
    # extract tars

    is_extract = True
    missing_class = []
    for class_ in org_class:
        tar_ = os.path.join(imgnet_folder,class_+'.tar')
        dest_tar = os.path.join(dest_folder,class_)
        if os.path.isfile(tar_):
            if not os.path.isdir(dest_tar) and is_extract:
                tar = tarfile.open(tar_)
                tar.extractall(path=dest_tar)
                tar.close()
            print(class_,'ok')
        else:
            missing_class.append(class_)

    print('missing:',missing_class)



if __name__ == "__main__":

    animals = AnimalParts(root='/scratch/data/', x_star='keypoints', datasplit='test')

    a = animals.__getitem__(4)

    a1 = animals.__getitem__(100)

    animals.stats()
    print('done!')


