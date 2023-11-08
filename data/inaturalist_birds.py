# from __future__ import print_function

from PIL import Image, ImageDraw
from os.path import join
import os
import pickle
import csv
import numpy as np
from itertools import compress
import torch.utils.data as data
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms
import torch
import data.custom_transforms as tr
import data.custom_transforms_pil as trpil

from collections import OrderedDict
import json

class Birds(data.Dataset):

    folder = '2017'

    def __init__(self,
                 root,
                 datasplit='val',
                 args=None,
                 crop_size=(300, 300)):
        super(Birds, self).__init__()
        self.as_pil = True
        self.root = join(os.path.expanduser(root), self.folder)
        self.args = args
        self.fill = 255 if self.as_pil else 1

        if args is None:
            self.data_fraction = 1
            self.every_n_is_val = 3
            self.is_crop = False
        else:
            self.data_fraction = self.args.train_data_fraction
            self.every_n_is_val = self.args.every_n_is_val
            self.is_crop = self.args.is_crop
        # self.is_masked_images = 'masked' in self.x_star
        self.datasplit=datasplit
        self.crop_size = crop_size[0]
        assert crop_size[0] == crop_size[1]
        # self.cropped = cropped
        self.is_debug = False
        # self.transform = transform
        # self.target_transform = target_transform

        data = self.load_split()

        aves_2017 = [x for x in data['categories'] if x['supercategory'] == 'Aves']

        with open(os.path.join(self.root, 'classes_CUB200_taxonomy.csv')) as f:
            spamreader = csv.reader(f, delimiter=' ')
            classes = [(int(x[1]), x[2], x[3]) for x in spamreader]

        names2017 = [x['name'] for x in aves_2017]
        ids2017 = [x['id'] for x in aves_2017]

        classes = [x for x in classes if x[2] in names2017]

        ids_inat = []
        for i, cl in enumerate(classes):
            index_ = names2017.index(cl[2])
            ids_inat.append(ids2017[index_])

        self.images = data['images']

        annotations = data['annotations']
        annotations_dict = {}
        for i in annotations:
            if isinstance(i, dict):
                annotations_dict[i['image_id']] = i['category_id']

        # add category_id
        for image in self.images:
            image_id = image['id']
            image['category_id'] = annotations_dict[image_id]

        # filter classes in CUB
        self.images = [x for x in self.images if x['category_id'] in ids_inat]

        # filter exisiting
        self.images = [x for x in self.images if os.path.isfile(join(self.root,x['file_name']))]

        # add class_id from CUB
        for image in self.images:
            index_ = ids_inat.index(image['category_id'])
            label_ = classes[index_][0]
            image['label'] = label_ - 1

        self.classes = classes

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
#        image_name, target_class = self._flat_breed_images[index]

        dict_out = {}

        image_dict = self.images[index]

        # id_, image_path = self.flat_images[index]
        target_class = image_dict['label']

        image_path = join(self.root, image_dict['file_name'])
        image = Image.open(image_path).convert('RGB')

        dict_out['s'] = image if self.as_pil else np.array(image)
        if self.as_pil:
            # if self.datasplit == 'train':
                # dict_out = self.transform_tr_pil(dict_out)
            # else:
            dict_out = self.transform_val_pil(dict_out)

            dict_out = {'s': dict_out['s']}
        else:
            raise NotImplementedError

        dict_out['label'] = torch.from_numpy(np.array(target_class))

        return dict_out

    def load_split(self):

        with open(os.path.join(self.root, 'train_val2017',f'{self.datasplit}2017.json')) as json_data:
            data = json.load(json_data)

        return data


    # def transform_tr_pil(self, sample):
    #
    #     composed_transforms = transforms.Compose([
    #         trpil.RandomResizedCrop(448, scale=[0.5,1]),
    #         trpil.RandomHorizontalFlip(),
    #         trpil.ToTensor(),
    #         trpil.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),key='s'),
    #     ])
    #
    #     return composed_transforms(sample)

    def transform_val_pil(self, sample):

        if self.is_crop or (self.datasplit in ['train','val']):
            composed_transforms = transforms.Compose([
                trpil.Resize(512),
                trpil.CenterCrop(448),
                trpil.ToTensor(),
                trpil.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), key='s'),
            ])
        else:
            composed_transforms = transforms.Compose([
                    trpil.ResizeNoCrop(800),
                    trpil.ToTensor(),
                    trpil.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), key='s'),
                ])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.Resize(600, 600),
            tr.Resize(512, 512),
            tr.Crop(crop_size=self.crop_size),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        ])
        return composed_transforms(sample)

    def stats(self):
        counts = {}
        # for index in range(len(self._flat_breed_images)):
        # for id_, _,_  in self.flat_images:
        for image in self.images:
        # for key, value in self.labels_dict.items():
            target_class = image['label']
            # target_class = self.labels_dict[id_]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1
        count_vector = np.array([val for key, val in counts.items()])

        print(f"{self.datasplit}: {len(self)} samples spanning {len(counts.keys())} classes "
              f"(avg {count_vector.mean():.2f} min {count_vector.min()} max {count_vector.max()} samples per class)")

        return counts

    def untar(self):
        import tarfile
        import os
        tar_ = '/scratch/data/iNaturalist/2017/train_val_images.tar.gz'
        i = 0
        missing = []
        for image in self.images:
            path_ = join(self.root, image['file_name'])

            if not os.path.isfile(path_):
                i +=1
                folder = os.path.dirname(image['file_name'])
                missing.append(os.path.basename(folder))
                # print(folder)
                cmd = f'tar -k -zxvf {tar_} {folder} -C {self.root}/{folder}'
                # print(cmd)
                # os.system(cmd)
                # with tarfile.open(tar_) as tar:
                    # subdir_and_files = [
                    #     tarinfo for tarinfo in tar.getmembers()
                    #     if tarinfo.name.startswith(f"{folder}/")
                    # ]
                    # tar.extractall(members=subdir_and_files, path=self.root)
                    # tar.extractall(members=image['file_name'], path=self.root)

        print(f'missing {i}/{len(self.images)}')
        print(set(missing))
        print(len(set(missing)))

if __name__ == "__main__":

    birds = Birds(root='/scratch/data/iNaturalist/', datasplit='val')

    birds.untar()

    a = birds.__getitem__(4)

    a1 = birds.__getitem__(100)

    birds.stats()
    print('done!')
