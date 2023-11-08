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
import cv2
import ast

class fgvc(data.Dataset):

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
        super(fgvc, self).__init__()
        self.as_pil = True
        self.root = root  # join(os.path.expanduser(root), self.folder)
        self.args = args
        self.x_star = x_star

        if args is None:
            self.data_fraction = 1
            self.every_n_is_val = 3
        else:
            self.data_fraction = self.args.train_data_fraction
            self.every_n_is_val = self.args.every_n_is_val

        self.datasplit = datasplit
        self.crop_size = crop_size[0]
        assert crop_size[0] == crop_size[1]
        # self.cropped = cropped
        self.is_debug = False
        self.transform = transform
        self.target_transform = target_transform

        with open(os.path.join(self.root, 'variants.txt')) as f:
            self.cat_list = [line.rstrip('\n') for line in f]
        self.labels_dict = {self.cat_list[i]: i for i in range(0, len(self.cat_list))}

        files_list = {"train": "images_variant_train.txt",
                      "val": "images_variant_val.txt",
                      "test": "images_variant_test.txt"}

        with open(os.path.join(self.root, files_list[datasplit])) as f:
            spamreader = csv.reader(f, delimiter=' ')
            data_list = list(spamreader)

        for idx in range(len(data_list)):
            if len(data_list[idx]) > 2:
                data_list[idx] = [data_list[idx][0], " ".join(data_list[idx][1:])]
        self.data_list = data_list

        if args.merge_train_validation and datasplit == "train":
            with open(os.path.join(self.root, files_list["val"])) as f:
                spamreader = csv.reader(f, delimiter=' ')
                data_list = list(spamreader)

            for idx in range(len(data_list)):
                if len(data_list[idx]) > 2:
                    data_list[idx] = [data_list[idx][0], " ".join(data_list[idx][1:])]

            self.data_list.extend(data_list)

        if datasplit == "train" and args.train_data_fraction < 1:
            s0 = None
            if args.np_seed is not None:
                s0 = np.random.get_state()
                np.random.seed(args.np_seed)

            # list of images per class for balanced sampling
            self.per_class_img = [[] for i in range(0, 100)]
            for en, x in enumerate(self.data_list):
                lab = self.labels_dict[x[1]]
                self.per_class_img[lab].append(x.copy())

            new_list = []
            for lab in range(0, 100):
                n_img = len(self.per_class_img[lab])
                get_n = int(n_img * args.train_data_fraction)
                idxs = np.random.choice(n_img, get_n, replace=False)
                for idx in idxs:
                    new_list.append(self.per_class_img[lab][idx].copy())

            if s0 is not None:
                np.random.set_state(s0)

            self.data_list = new_list

    def __len__(self):

        return len(self.data_list)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        #        image_name, target_class = self._flat_breed_images[index]

        dict_out = {}

        id, label_name = self.data_list[index]
        target_class = self.labels_dict[label_name]

        image_path = join(self.root, "images/{}.jpg".format(id))
        image = Image.open(image_path).convert('RGB')

        # remove copyright bottom frame
        bottom = image.size[1] - 20
        image = image.crop((0, 0, image.size[0], bottom))

        if 'keypoints' in self.x_star:
            # parts = self.part_locs_dict[id_]
            parts_list = [(0, 0)] * 1
            size_circle = 20  # 10 # pixels
            mask_out = []
            min_y, min_x, max_y, max_x = image.size[0], image.size[1], 0, 0
            for i in range(1, 2):  # loop over all the parts
                # creating new Image object
                bbox_mask = Image.new("L", image.size)
                if True:  # i in parts.keys():
                    x, y = 0, 0
                    # parts_list[i-1] = parts[i]
                    min_y = min(min_y, y - 50)
                    min_x = min(min_x, x - 50)
                    max_y = max(max_y, y + 50)
                    max_x = max(max_x, x + 50)

                    x1, y1 = x - size_circle, y - size_circle
                    x2, y2 = x + size_circle, y + size_circle
                    circle = (x1, y1, x2, y2)

                    # create rectangle image
                    img1 = ImageDraw.Draw(bbox_mask)
                    img1.ellipse(circle, fill=255)
                if self.as_pil:
                    mask_out.append(bbox_mask)
                else:
                    mask_out.append(np.array(bbox_mask))

            min_y = max(min_y, 0)
            min_x = max(min_x, 0)
            max_y = min(max_y, image.size[1])
            max_x = min(max_x, image.size[0])

            keypoints_bbox = [min_x, min_y, max_x, max_y]

            # for i in range(0,15): # loop over all the parts
            #    mask_out[i] = mask_out[i].crop(keypoints_bbox)
            # image = image.crop(keypoints_bbox)
            teacher = mask_out if self.as_pil else np.stack(mask_out, axis=-1)

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
                dict_out = self.transform_val(dict_out)

        dict_out['label'] = torch.from_numpy(np.array(target_class))

        return dict_out

    def transform_tr_pil(self, sample):

        is_birds_flipping = False
        composed_transforms = transforms.Compose([
            trpil.RandomResizedCrop(448, scale=[0.8, 1]),
            trpil.RandomHorizontalFlip(is_birds_flipping=is_birds_flipping),
            trpil.ToTensor(),
            trpil.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), key='s'),
        ])

        return composed_transforms(sample)

    def transform_val_pil(self, sample):

        if self.args.is_crop:
            composed_transforms = transforms.Compose([
                trpil.Resize(448),
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
        counts = np.zeros(len(self.cat_list))
        # for index in range(len(self._flat_breed_images)):
        # for id_, _,_  in self.flat_images:

        for el in self.data_list:
            counts[self.labels_dict[el[1]]] += 1

        print(f"{self.datasplit}: {len(self.data_list)} samples spanning {len(counts)} classes "
              f"(avg {float(len(self.data_list)) / float(len(counts)):.2f} per class)")
        return counts

def annotate_keypoints(img_path, output_path, bbox, dict_info):
    # Path to source video:
    # output_path = '/home/stephen/Desktop/clicks.csv'

    # Mouse callback function
    global keypoints_input_list
    global keypoints

    positions, keypoints_input_list = [], []
    keypoints = ['tip','tail','wing','engine_wing','engine_tail','wheels']

    def callback(event, x, y, flags, param):
        global keypoints_input_list

        if len(keypoints_input_list) < len(keypoints):
            if event == 1 or event == 2:
                if event == 2:
                    x, y = -1, -1
                print(x,y)
                keypoints_input_list.append((x, y))
                if len(keypoints_input_list) < len(keypoints):
                    if dict_info['category_id'] != 11 and len(keypoints_input_list) == 7:
                        keypoints_input_list.extend([(-1,-1), (-1,-1)])
                    else:
                        print('keypoint', keypoints[len(keypoints_input_list)])

            elif event == 3:
                keypoints_input_list.pop(-1)
                if len(keypoints_input_list) < len(keypoints):
                    print('retype keypoint', keypoints[len(keypoints_input_list)])
        elif event != 0:
                print('done!, press esc to continue to next image')

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1200, 1200)
    cv2.setMouseCallback('img', callback)

    img = cv2.imread(img_path)

    if bbox is not None:
        img = cv2.rectangle(img.copy(), (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    print('keypoint', keypoints[0])

    # Mainloop - show the image and collect the data
    while True:
        cv2.imshow('img', img)
        # Wait, and allow the user to quit with the 'esc' key
        k = cv2.waitKey(1)
        # If user presses 'esc' break
        # if k == 27 or len(keypoints_input_list) == len(keypoints):
        if k == 27:
            break
    cv2.destroyAllWindows()

    dict_out = dict_info
    for i, position in enumerate(keypoints_input_list):
        # x, y = position[0], position[1]
        dict_out[keypoints[i]] = position
    print(dict_out)


    # # Write data to a spreadsheet
    # with open(output_path, 'w') as csvfile:
    #     fieldnames = keypoints + list(dict_info.keys())
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerow(dict_out)

    # # Adding hyperparameters to results.csv
    # flags = trainer.args.__dict__
    dset = pd.DataFrame(columns=dict_out.keys())
    dset = dset.append(dict_out,ignore_index=True)
    # results = pd.concat((flags_dset,results),axis=1)

    # save_file = os.path.join(args.save_dir, trainer.saver.data_name, 'results.csv')
    if os.path.isfile(output_path):
        dset_old = pd.read_csv(output_path)
        dset = pd.concat([dset_old,dset])
    dset.to_csv(output_path, index=False)


if __name__ == "__main__":
    # generate_attributes_classes(root='/scratch/data/CUB_200_2011/CUB_200_2011/attributes')

    fgvc = fgvc(root="/scratch/tmp/fgvc-aircraft-2013b/data/", x_star='keypoints')

    a = fgvc.__getitem__(4)

    a1 = fgvc.__getitem__(100)

    fgvc.stats()
    print('done!')
