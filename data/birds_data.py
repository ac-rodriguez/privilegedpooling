# from __future__ import print_function

from PIL import Image, ImageDraw
from os.path import join
import os
import pickle
import csv
import json
import numpy as np
from itertools import compress
import torch.utils.data as data
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms
import torch
try:
    import data.custom_transforms as tr
    import data.custom_transforms_pil as trpil
except ModuleNotFoundError:
    import custom_transforms as tr
    import custom_transforms_pil as trpil


from collections import OrderedDict

import cv2

def create_bbox_from_keypoint(x, y, size):
    x1, y1 = x - size, y - size
    x2, y2 = x + size, y + size
    return x1, y1, x2, y2
keypoints_CUB = ["back", "beak", "belly", "breast","crown","forehead","left eye","left leg","left wing","nape","right eye","right leg","right wing","tail","throat"]

keypoints_text = ''
for i, key in enumerate(keypoints_CUB):
    keypoints_text += f'{i+1}-{key}\n'

class Birds(data.Dataset):
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
    folder = 'CUB_200_2011'

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
        super(Birds, self).__init__()
        self.as_pil = True
        self.root = join(os.path.expanduser(root), self.folder)
        self.args = args
        self.x_star = x_star
        if args is None:
            self.data_fraction = 1
            self.every_n_is_val = 3
            self.only_novel=False
            self.n_shot = None
            self.experiment_id = 1
        else:
            self.data_fraction = self.args.train_data_fraction
            self.every_n_is_val = self.args.every_n_is_val
            self.only_novel = self.args.n_shot_only_novel
            self.n_shot = self.args.n_shot
            self.experiment_id = self.args.n_shot_experiment_id

        self.is_masked_images = 'masked' in self.x_star
        self.datasplit=datasplit
        self.crop_size = crop_size[0]
        assert crop_size[0] == crop_size[1]
        # self.cropped = cropped
        self.is_debug = False
        self.balance_train_dataset = (self.n_shot is not None) and self.only_novel
        self.transform = transform
        self.target_transform = target_transform

        split = self.load_split()
        use_val_for_train = True

        if self.datasplit in ['train','val']:
            if trainvalindex is not None:
                assert len(split) == len(trainvalindex)
                split = list(compress(split,trainvalindex))
            if self.n_shot is not None:
                assert self.args.train_data_fraction == 1
                split = self.filter_few_shot(split)
            # take every n image for validation
            valindex = range(len(split))
            valindex = np.mod(valindex, self.every_n_is_val) == 0

            if self.datasplit == 'val':
                split = list(compress(split, valindex))
            elif not use_val_for_train:
                split = list(compress(split, ~valindex))


        self.images_paths = []
        with open(os.path.join(self.root, 'images.txt')) as f:
            spamreader = csv.reader(f, delimiter=' ')
            for row in spamreader:
                id_ = int(row[0])
                if id_ in split:
                    self.images_paths.append((id_,join('images',row[1])))


        self.bb_bird_dict = {}
        with open(os.path.join(self.root, 'bounding_boxes.txt')) as f:
            spamreader = csv.reader(f, delimiter=' ')
            for row in spamreader:
                id_ = int(row[0])
                if id_ in split:
                    self.bb_bird_dict[id_] = [int(float(x)) for x in row[1:5]]


        self.part_locs_dict = dict((item,{}) for item in split)
        with open(os.path.join(self.root,'parts', 'part_locs.txt')) as f:
            spamreader = csv.reader(f, delimiter=' ')
            for row in spamreader:
                id_ = int(row[0])
                part_ = int(row[1])
                if id_ in split:
                    if row[4] == "1":  #only add visible parts
                        self.part_locs_dict[id_][part_] = [int(float(x)) for x in row[2:4]]

        if 'attributes' in  self.x_star:
            self.load_attributes_as_dict()

        self.flat_images = self.images_paths

        with open(os.path.join(self.root,'classes.txt')) as f:
            spamreader = csv.reader(f, delimiter=' ')
            self.classes = list((y.split('.')[1] for x,y in spamreader))

        # list of images per class for balanced sampling
        self.n_classes = len(set(self.labels_dict.values()))
        self.per_class_img = [[] for _ in range(0,self.n_classes)]
        for en,x in enumerate(self.flat_images):
            id_, image_path = x
            lab = self.labels_dict[id_]
            self.per_class_img[lab].append(en)

            
    def get_size_attributes(self):
        
        return [len(val) for key,val in self.cat_attributes_dict.items()]
    
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

        if self.balance_train_dataset and self.datasplit=="train":
            idx1 = torch.randint(high=self.n_classes,size=(1,))
            idx2 = torch.randint(high=len(self.per_class_img[idx1]),size=(1,))
            index = self.per_class_img[idx1][idx2]

        dict_out = {}
        keypoints_bbox = None

        id_, image_path = self.flat_images[index]
        target_class = self.labels_dict[id_]

        image_path = join(self.root, image_path)
        image = Image.open(image_path).convert('RGB')
        mask_out = []

        if 'bbox' in self.x_star:

            bbox = self.bb_bird_dict[id_]

            # creating new Image object
            bbox_mask = Image.new("L", image.size)

            # create rectangle image
            img1 = ImageDraw.Draw(bbox_mask)
            img1.rectangle(bbox, fill=255)
            if not 'keypoints' in self.x_star:

                teacher = bbox_mask if self.as_pil else np.array(bbox_mask)
                teacher = [teacher]
            else:
                mask_out.append(bbox_mask)
        if 'keypoints-viz' in self.x_star:
            parts = self.part_locs_dict[id_]

            draw = ImageDraw.Draw(image)
            for i in range(1,16): # loop over all the parts

                if i in parts.keys():
                    x, y = parts[i]

                    rect_ = create_bbox_from_keypoint(x, y, size=5)

                    draw.rectangle(rect_, outline=255)
                    draw.text((x,y),str(i))
            draw.text((0,0),keypoints_text)

            image.resize((1024, 1024))
            while True:
                cv2.imshow('img', np.array(image))
                # Wait, and allow the user to quit with the 'esc' key
                k = cv2.waitKey(1)
                if k == 27:
                    break
            cv2.destroyAllWindows()
            return -1

        if 'keypoints' in self.x_star:

            parts = self.part_locs_dict[id_]
            parts_list = [(0,0)] * 15
            size_circle = 10 #10 # pixels
            min_y,min_x,max_y,max_x = image.size[0],image.size[1],0,0
            for i in range(1,16): # loop over all the parts
                # creating new Image object
                bbox_mask = Image.new("L", image.size)
                if i in parts.keys():
                    x, y = parts[i]
                    parts_list[i-1] = parts[i]
                    min_y = min(min_y,y-50)
                    min_x = min(min_x,x-50)
                    max_y = max(max_y,y+50)
                    max_x = max(max_x,x+50)

                    x1,y1 = x-size_circle,y-size_circle
                    x2,y2 = x+size_circle,y+size_circle
                    circle = (x1,y1,x2,y2)

                    # create rectangle image
                    img1 = ImageDraw.Draw(bbox_mask)
                    img1.ellipse(circle, fill=255)
                if self.as_pil:
                    mask_out.append(bbox_mask)
                else:
                    mask_out.append(np.array(bbox_mask))

            min_y = max(min_y,0)
            min_x = max(min_x,0)
            max_y = min(max_y,image.size[1])
            max_x = min(max_x,image.size[0])

            keypoints_bbox = [min_x,min_y,max_x,max_y]

            #for i in range(0,15): # loop over all the parts
            #    mask_out[i] = mask_out[i].crop(keypoints_bbox)
            #image = image.crop(keypoints_bbox)

            teacher = mask_out if self.as_pil else np.stack(mask_out, axis=-1)

            if self.is_masked_images:
                teacher_array = np.stack([np.array(x) for x in teacher]).max(axis=0)
                teacher = [Image.fromarray(teacher_array)]

        dict_out['s'] = image if self.as_pil else np.array(image)
        if self.as_pil:
            for i, t in enumerate(teacher):
                dict_out[f't_{i}'] = t
            if self.datasplit == 'train':
                dict_out = self.transform_tr_pil(dict_out)
            else:
                dict_out = self.transform_val_pil(dict_out)

            list_cat = [dict_out[f't_{i}'] for i in range(len(teacher))]
            dict_out = {'s': dict_out['s']}
            dict_out['t'] = torch.cat(list_cat, dim=0)

        else:
            dict_out['t'] = teacher

            if self.datasplit == 'train':
                dict_out = self.transform_tr(dict_out)
            else:
                dict_out = self.transform_val(dict_out,keypoints_bbox)

        dict_out['label'] = torch.from_numpy(np.array(target_class))
        if 'keypoints' in self.x_star:
            dict_out['t_loc'] = torch.from_numpy(np.array(parts_list))
        if 'catattributes' in self.x_star:
            attributes = self.attributes_dict_categorical[id_]

            dict_out['t_attr'] = [torch.from_numpy(np.array(x)) for x in attributes]
        elif 'attributes' in self.x_star:
            attributes = self.attributes_dict[id_]

            if self.args.model == 'outer_class':
                class_matrix = np.zeros((200,312))
                class_matrix[target_class] = np.array(attributes)
                dict_out['t_attr'] = torch.from_numpy(class_matrix).float().flatten()
            else:
                dict_out['t_attr'] = torch.from_numpy(np.array(attributes)).float()

        return dict_out


    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
            if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    def load_attributes_as_dict(self):
        dict_file = os.path.join(self.root, 'attributes','image_attribute_labels.pkl')

        if not os.path.isfile(dict_file):
            self.attributes_dict = {}
            with open(os.path.join(self.root, 'attributes', 'image_attribute_labels.txt')) as f:
                spamreader = csv.reader(f, delimiter=' ')
                for row in spamreader:
                    id_ = int(row[0])
                    part_ = int(row[1]) - 1
                    if not id_ in self.attributes_dict.keys():
                        self.attributes_dict[id_] = [None] * 312
                    self.attributes_dict[id_][part_] = int(row[2])
            with open(dict_file, 'wb') as f:
                pickle.dump(self.attributes_dict, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(dict_file, 'rb') as f:
                self.attributes_dict = pickle.load(f)

        # load attributes as categories

        if 'catattributes' in self.x_star:
            cat_attributes_dict = OrderedDict()
            flat_cat_list = []
            with open(os.path.join(self.root, 'attributes', 'attributes.txt')) as f:
                spamreader = csv.reader(f,delimiter=' ')
                for row in spamreader:
                    cat_, value = row[1].split('::')
                    if cat_ in cat_attributes_dict.keys():
                        cat_attributes_dict[cat_] += [value]
                    else:
                        cat_attributes_dict[cat_] = [value]
                    flat_cat_list.append([list(cat_attributes_dict.keys()).index(cat_), cat_attributes_dict[cat_].index(value)])

            self.flat_cat_attributes = flat_cat_list
            self.cat_attributes_dict = cat_attributes_dict

            dict_file = os.path.join(self.root, 'attributes', 'image_catattribute_labels.pkl')

            if not os.path.isfile(dict_file):

                dict_out = {}
                for key, val in self.attributes_dict.items():
                    for cat_ in range(len(cat_attributes_dict.keys())):
                        values = [ind_ for ind_, (a, b) in enumerate(flat_cat_list) if a == cat_]
                        val_ = np.take(val,values)

                        if key in dict_out.keys():
                            dict_out[key] += [val_]
                        else:
                            dict_out[key] = [val_]
                with open(dict_file, 'wb') as f:
                    pickle.dump(dict_out, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(dict_file, 'rb') as f:
                    dict_out = pickle.load(f)

            self.attributes_dict_categorical = dict_out



    @staticmethod
    def get_boxes(path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes

    def load_split(self):

        with open(os.path.join(self.root, 'train_test_split.txt')) as f:
            spamreader = csv.reader(f, delimiter=' ')
            split = list(spamreader)
                #[row for row in spamreader]


        if self.datasplit in ['train','val']:
            split = [x for x,data_type in split if data_type == "1"]
        else:
            split = [x for x,data_type in split if data_type == "0"]

        split = [int(x) for x in split]


        self.labels_dict = {}
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            spamreader = csv.reader(f, delimiter=' ')
            for row in spamreader:
                id_ = int(row[0])
                if id_ in split:
                    self.labels_dict[id_] = int(row[1]) - 1  # offset by 0

        return split

    def filter_few_shot(self, split):

        experiment_path = '/fewshot/splitfile_cub_{:d}.json'.format(self.experiment_id)
        print(f'loading {experiment_path}')
        experiment_path = self.root + experiment_path

        with open(experiment_path, 'r') as f:
            exp = json.load(f)

        if self.only_novel:
            base_novel_path = self.root + '/fewshot/base_novel_split.json'

            with open(base_novel_path, 'r') as f:
                lowshotmeta = json.load(f)

            novel_classes = lowshotmeta['novel_classes']
            novel_index = np.array(exp)[:, :self.n_shot]
            novel_index = novel_index[novel_classes, :].reshape(-1)
            novel_ids = [split[x] for x in novel_index]

            base_classes = lowshotmeta['base_classes']
            base_ids = [x for x in split if self.labels_dict[x] in base_classes]

            ids = novel_ids + base_ids

            self.lowshotmeta = lowshotmeta

        else:
            idx = np.array(exp)[:, :self.n_shot]
            idx = idx.reshape(-1)
            ids = [split[x] for x in idx]

        return [x for x in split if x in ids]




    def transform_tr_pil(self, sample):


        is_birds_flipping = 'keypoints' in self.x_star
        is_croping = self.args.is_crop_train and self.datasplit == 'train'

        composed_transforms = transforms.Compose([
            trpil.RandomResizedCrop(448, scale=[0.5,1]) if is_croping else trpil.ResizeNoCrop(800),
            trpil.RandomHorizontalFlip(is_birds_flipping=is_birds_flipping),
            trpil.ToTensor(),
            trpil.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),key='s'),
            tr.Masked_images() if self.is_masked_images else tr.Lambda(),

        ])

        return composed_transforms(sample)

    def transform_val_pil(self, sample):

        if self.args.is_crop or (self.datasplit in ['train','val']):
            composed_transforms = transforms.Compose([
                trpil.Resize(448),
                trpil.CenterCrop(448),
                trpil.ToTensor(),
                trpil.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), key='s'),
                tr.Masked_images() if self.is_masked_images else tr.Lambda(),
            ])
        else:
            composed_transforms = transforms.Compose([
                    trpil.ResizeNoCrop(800),
                    trpil.ToTensor(),
                    trpil.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), key='s'),
                    tr.Masked_images() if self.is_masked_images else tr.Lambda(),
                ])


        return composed_transforms(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.Resize(600, 600),
            # tr.Resize(512, 512),
            tr.RandomHorizontalFlip(),
            tr.RandomCrop(crop_size=self.crop_size),
            # tr.ColorJitter(),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
            tr.Masked_images() if self.is_masked_images else tr.Lambda(),
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
                tr.Masked_images() if self.is_masked_images else tr.Lambda(),
            ])

        return composed_transforms(sample)

    def stats(self):
        counts = {}
        for s, _ in self.flat_images:
            target_class = self.labels_dict[s]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print(f"{self.datasplit}: {sum(counts.values())} samples spanning {len(counts.keys())} classes "
              f"(avg {float(sum(counts.values()))/float(len(counts.keys())):.2f} per class)")

        if self.only_novel and self.datasplit == 'train':

            counts_base = {}
            counts_novel = {}
            for s, _ in self.flat_images:
                target_class = self.labels_dict[s]
                if target_class in self.lowshotmeta['base_classes']:
                    if target_class not in counts_base.keys():
                        counts_base[target_class] = 1
                    else:
                        counts_base[target_class] += 1
                elif target_class in self.lowshotmeta['novel_classes']:
                    if target_class not in counts_novel.keys():
                        counts_novel[target_class] = 1
                    else:
                        counts_novel[target_class] += 1
                else:
                    raise ValueError(f'class {target_class}')

            print(f"BASE: {sum(counts_base.values())} samples spanning {len(counts_base.keys())} classes "
                  f"(avg {float(sum(counts_base.values()))/float(len(counts_base.keys())):.2f} per class)")


            print(f"NOVEL: {sum(counts_novel.values())} samples spanning {len(counts_novel.keys())} classes "
                  f"(avg {float(sum(counts_novel.values()))/float(len(counts_novel.keys())):.2f} per class)")

        return counts

def generate_attributes_classes(root):
    # coding=utf-8
    import re
    import numpy as np

    # get attribute cluster idx
    attribute_name_file = root+'/attributes.txt'
    f1 = open(attribute_name_file, 'rb')
    start_idxs = []
    last_attr = ''
    for line in csv.reader(f1,delimiter=' '):
        strs = line.split('::')
        if (strs[1] != last_attr):
            start_idxs.append(int(strs[0]))
        last_attr = strs[1]
    start_idxs.append(int(strs[0]) + 1)
    print(start_idxs)
    a = np.array(start_idxs)
    nums = a[1:] - a[:-1] + 1
    print(np.sum(nums))
    print(nums.tolist())

    # transform binary attribute to clustered attribute
    nb_attr = len(start_idxs) - 1
    A_all = np.zeros((11788, nb_attr))
    image_attribute_file = root+'/image_attribute_labels.txt'
    f2 = open(image_attribute_file, 'rb')
    for line in f2.readlines():
        strs = re.split(' ', line)
        img_id = int(strs[0]) - 1
        attr_id = int(strs[1])
        is_present = int(strs[2])
        if (is_present > 0):
            for i in range(len(start_idxs)):
                if (attr_id < start_idxs[i]):
                    break
            A_all[img_id][i - 1] = attr_id - start_idxs[i - 1] + 1  # 0 mean no attr
    print(A_all[1])

    new_attr_file = root+'/processed_attributes.txt'
    np.savetxt(new_attr_file, A_all, fmt='%d')

if __name__ == "__main__":

    # generate_attributes_classes(root='/scratch/data/CUB_200_2011/CUB_200_2011/attributes')

    birds = Birds(root='/scratch/data/CUB_200_2011', x_star='keypoints')


    birds = Birds(root='/scratch/data/CUB_200_2011', x_star='keypoints-viz')
    for i in range(10):
        _ = birds[i]

    # a = birds.__getitem__(4)

    # a1 = birds.__getitem__(100)

    birds.stats()
    print('done!')
