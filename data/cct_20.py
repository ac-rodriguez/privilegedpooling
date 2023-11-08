

from PIL import Image, ImageDraw
from os.path import join
import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
import pandas as pd
import cv2
import ast

import data.custom_transforms_pil as trpil
import data.custom_transforms as tr


def create_bbox_from_keypoint(x, y, size):
    x1, y1 = x - size, y - size
    x2, y2 = x + size, y + size
    return x1, y1, x2, y2


class CCT20(data.Dataset):
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
    folder = 'CaltechCameraTraps/CCT20'
    # download_url_prefix = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.ta'

    def __init__(self,
                 root,
                 classes_train='all',
                 datasplit='train',
                 args=None,
                 trainvalindex=None,
                 x_star='keypoints',
                 download=False,
                 crop_size=(300,300)):
        super(CCT20, self).__init__()
        self.as_pil = True

        self.root = join(os.path.expanduser(root), self.folder)
        self.args = args
        self.x_star = x_star
        self.classes_train = classes_train
        self.is_remove_multiple_instances = True
        self.balance_train_dataset = True

        if args is None:
            self.data_fraction = 1
            self.every_n_is_val = 3
            self.crop_train_around_bbox = False
            self.only_annotated_images = False
            self.is_crop = False
        else:
            self.data_fraction = self.args.train_data_fraction
            self.every_n_is_val = self.args.every_n_is_val
            self.crop_train_around_bbox = 'cropbbox' in self.args.tag
            self.only_annotated_images = not 'fulldataset' in self.args.tag
            self.is_crop = self.args.is_crop

        self.is_masked_images = 'masked' in self.x_star
        self.datasplit=datasplit
        self.crop_size = crop_size[0]
        assert crop_size[0] == crop_size[1]

        self.is_debug = False

        data_dict = self.load_split()

        self.categories_dict = {}
        for c in data_dict['categories']:
            self.categories_dict.update({c['id']: c['name']})

        self.images_list = data_dict['images']
        self.cat_list = [val for key, val in self.categories_dict.items()]

        dict_ = {}
        id_multiple_instances = []
        for a in data_dict['annotations']:
            id_ = a['image_id']
            if id_ in dict_.keys():
                id_multiple_instances.append(id_)
                if self.is_remove_multiple_instances:
                    dict_.pop(id_)
                # raise ValueError
            else:
                dict_[id_] = a
        self.annotations_dict = dict_
        if self.is_remove_multiple_instances:
            print('removing samples with multiple instances')
            self.images_list = [x for x in self.images_list if not x['id'] in id_multiple_instances]
        self.id_multiple_labels = id_multiple_instances

        if True :#self.datasplit == 'train':
            # print('removing samples without bbox annotation')
            # self.images_list = [x for x in self.images_list if 'bbox' in self.annotations_dict[x['id']].keys()]

            print('samples without bbox annotation and cars as empty class')
            empty_cat_id = 30

            for x in self.images_list:
                dict_ = self.annotations_dict[x['id']]
                cat_id_ = self.categories_dict[dict_['category_id']]
                if not 'bbox' in dict_.keys() or cat_id_ == "car":
                    self.annotations_dict[x['id']]['category_id'] = empty_cat_id

        self.__info__ = data_dict['info']

        self.images_folder = os.path.join(self.root, 'eccv_18_all_images_sm')
        # self.images_folder = os.path.join(self.root, 'eccv_18_cropped')

        self.loc_list = list(set([x['location'] for x in self.images_list]))

        # keypoints
        self.keypoints_file = os.path.join(self.root, 'keypoints/keypoints_andres.csv')
        self.read_keypoints_dict()
        self.keypoints_name = ['head', 'left-front-leg', 'right-front-leg', 'left-back-leg', 'right-back-leg', 'tail', 'body'] #   'left-wing', 'right-wing']

        print(len(self.keypoints_dict))
        ann_list = list(self.keypoints_dict.values())
        image_id_list_key = []
        for el in ann_list:
            image_id_list_key.append(el['image_id'])

        if self.only_annotated_images and self.datasplit == 'train':
            print('removing samples without keypoints annotation')
            self.images_list = [x for x in self.images_list if ( (x['id'] in image_id_list_key) or (self.categories_dict[self.annotations_dict[x['id']]['category_id']] ==  "empty") )]
        print(len(self.images_list))

        # list of images per class for balanced sampling
        self.per_class_img = [ [] for i in range(0,16)]
        for en,x in enumerate(self.images_list):
            lab = self.cat_list.index(self.categories_dict[self.annotations_dict[x['id']]['category_id']])
            self.per_class_img[lab].append(en)


        self.stats(verbose=False)

        self.is_viz = True

    def read_keypoints_dict(self):
        file = os.path.join(self.root, 'keypoints/keypoints_andres.csv')
        if os.path.isfile(file):
            ds = pd.read_csv(file, dtype={'id': str}).set_index('id')
        else:
            ds = None
        file = os.path.join(self.root, 'keypoints/keypoints_stefano.csv')
        if os.path.isfile(file):
            ds_ = pd.read_csv(file, dtype={'id': str}).set_index('id')
            if ds is not None:
                ds = pd.concat([ds,ds_])
            else:
                ds = ds_

        if ds is not None:
            ds.dropna(axis=0,inplace=True)
            self.keypoints_dict = ds.to_dict('index')
        else:
            self.keypoints_dict = {}

    def __len__(self):
        if self.is_debug:
            return 1000
        return len(self.images_list)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """

        if self.balance_train_dataset and self.datasplit=="train":
            idx1 = 14
            while idx1 == 14:
                idx1 = torch.randint(high=16,size=(1,))
            idx2 = torch.randint(high=len(self.per_class_img[idx1]),size=(1,))
            index = self.per_class_img[idx1][idx2]

        dict_out = {}
        image_dict = self.images_list[index]
        image_path = os.path.join(self.images_folder,image_dict['file_name'])
        image = Image.open(image_path).convert('RGB')

        target_class = self.annotations_dict[image_dict['id']]['category_id']
        target_class_name = self.categories_dict[target_class]
        label = self.cat_list.index(target_class_name)
        loc_ = image_dict['location']

        dict_ = self.annotations_dict[image_dict['id']]

        r = image_dict['width'] / 1024 if self.images_folder.endswith('_sm') else 1

        if 'bbox' in dict_.keys():
            bbox = dict_['bbox'] # x , y , w, h
            x, y, w, h = bbox
            # resize bbox coordinates to new size


            # bbox = list(map(int, [x/r, y/r, (x + w)/r, (y + h)/r])) # xmin, ymin, xmax , ymax
            bbox = [x/r, y/r, (x + w)/r, (y + h)/r] # xmin, ymin, xmax , ymax
            bbox = list(map(int, bbox))

            dict_info = {k:v for k,v in self.annotations_dict[image_dict['id']].items() if k in ['id', 'image_id', 'category_id']}

            # check if already annotated
            # if os.path.isfile(self.keypoints_file):
            if 'keypoints-annotate' in self.x_star:
                current_count = self.keypoint_table[self.loc_list.index(loc_), self.cat_list.index(target_class_name)]

                # if current_count >= 5 or target_class in [33, 30]:
                #     return -1

                self.read_keypoints_dict()
                exists = dict_info['id'] in self.keypoints_dict.keys()

                if not exists:
                    ## annotate image:
                    print('category: ',  self.categories_dict[target_class], 'loc:', loc_ , 'current count: ', current_count)
                    annotate_keypoints(image_path, output_path=self.keypoints_file,
                                           bbox=bbox, dict_info=dict_info)

                elif self.is_viz:
                    draw = ImageDraw.Draw(image)

                    draw.rectangle(bbox,outline=255)
                    for key, val in self.keypoints_dict[dict_info['id']].items():
                        if key in self.keypoints_name:

                            val = ast.literal_eval(val)

                            if val[0] != -1:  # not missing
                                val = (val[0] / r, val[1] / r)

                                rect_ = create_bbox_from_keypoint(val[0], val[1], size=10)

                                draw.rectangle(rect_, outline=255)
                    image.resize((1024, 1024))
                    image.show()
                    image.close()
                return -1

        else:
            # print('bbox missing')
            bbox = None

        if self.x_star == 'bbox':

            bbox_mask = Image.new("L", image.size)
            if bbox is not None:
                draw = ImageDraw.Draw(bbox_mask)
                draw.rectangle(bbox,fill=255)

            teacher = bbox_mask if self.as_pil else np.array(bbox_mask)
            teacher = [teacher]


        elif 'keypoints' in self.x_star:

            mask_out = []
            annotation_id = self.annotations_dict[image_dict['id']]['id']
            if not annotation_id in self.keypoints_dict.keys():
                mask_out = [Image.new("L", image.size) for _ in self.keypoints_name]
            else:
                for k in self.keypoints_name:
                    bbox_mask = Image.new("L", image.size)
                    draw = ImageDraw.Draw(bbox_mask)

                    if k in self.keypoints_dict[annotation_id].keys():
                        val = self.keypoints_dict[annotation_id][k]
                        if val is None:
                            val = (-1,-1)
                        else:
                            val = ast.literal_eval(val)
                            val = (val[0]/r, val[1]/r)

                        if val[0] != -1:  # not missing
                            rect_ = create_bbox_from_keypoint(val[0], val[1], size=10)
                            draw.ellipse(rect_, fill=255)
                    mask_out.append(bbox_mask)

            if 'bbox' in self.x_star:
                bbox_mask = Image.new("L", image.size)
                if bbox is not None:
                    draw = ImageDraw.Draw(bbox_mask)
                    draw.rectangle(bbox,fill=255)
                mask_out.append(bbox_mask)

            if self.as_pil:
                teacher = mask_out
            else:
                teacher = np.stack([np.array(x) for x in mask_out], axis=-1)
        else:
            teacher = None
            # if self.is_masked_images:
            #     dict_out['t'] = np.array(dict_out['t'].sum(axis=-1) > 0,dtype=np.float32)[...,np.newaxis]
        if self.datasplit == 'train' and self.crop_train_around_bbox and bbox is not None:
            covered_bbox = 1 - 0.65 # 65% of the image should be covered by the bbox
            left,upper,right,lower = bbox
            w_, h_ = (right - left)//2, (lower - upper)//2
            bbox_ext = [max(0,left - w_//covered_bbox),
                        max(0,upper - h_//covered_bbox),
                        min(image.size[0]-1, right + w_//covered_bbox),
                        min(image.size[1]-1, lower + h_//covered_bbox)]
            image = image.crop(bbox_ext)
            if self.as_pil:
                teacher = [t.crop(bbox_ext) for t in teacher]
            else:
                raise NotImplementedError

        dict_out['s'] = image if self.as_pil else np.array(image)
        if self.as_pil:
            if teacher is not None:
                for i, t in enumerate(teacher):
                    dict_out[f't_{i}'] = t

            dict_out = self.transform_tr_pil(dict_out) if self.datasplit == 'train' else self.transform_val_pil(dict_out)

            if teacher is not None:
                list_cat = [dict_out[f't_{i}'] for i in range(len(teacher))]
                dict_out['t'] = torch.cat(list_cat, dim=0)

        else:
            if self.datasplit == 'train':
                dict_out = self.transform_tr(dict_out)
            else:
                dict_out = self.transform_val(dict_out)

        dict_out['label'] = torch.tensor(label)

        return dict_out

    def load_split(self):

        splits_dir = os.path.join(self.root, 'eccv_18_annotation_files')

        assert self.datasplit in ['train', 'cis_val', 'trans_val', 'cis_test', 'trans_test']

        split_f = os.path.join(splits_dir, self.datasplit+'_annotations.json')

        with open(split_f) as f:
            data = f.read()
            data_dict = json.loads(data)
        return data_dict


    def transform_tr_pil(self, sample):

        is_cct_flipping = 'keypoints' in self.x_star

        composed_transforms = transforms.Compose([
            trpil.RandomResizedCrop(448, scale=[0.8,1]),
            trpil.RandomHorizontalFlip(is_cct_flipping=is_cct_flipping),
            trpil.ToTensor(),
            trpil.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),key='s'),
        ])

        return composed_transforms(sample)

    def transform_val_pil(self, sample):

        if self.is_crop or (self.datasplit in ['train','val']):
            composed_transforms = transforms.Compose([
                trpil.Resize(600),
                trpil.CenterCrop(600),
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

    def stats(self, verbose=True, return_keypoint_dict=False):
        counts = {}
        for index in range(len(self)):

            image_dict = self.images_list[index]
            target_class = self.annotations_dict[image_dict['id']]['category_id']
            target_class = self.categories_dict[target_class]

            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1
        if verbose:
            print('samples with multiple labels', len(set(self.id_multiple_labels)))
            print(f"\t {self.datasplit}: {len(self)} samples spanning {len(counts.keys())} classes "
                  f"(avg {float(len(self))/float(len(counts.keys())):.2f} per class)")

            print(counts)
            print('\n')

        keypoints_table = np.zeros((10, 16), dtype=np.int32)

        if self.datasplit == 'train':

            image_id_list = [x['id'] for x in self.images_list]

            for key, val in self.keypoints_dict.items():

                # image_dict = self.images_list[index]
                target_class = val['category_id']
                target_class = self.categories_dict[target_class]

                loc_ = self.images_list[image_id_list.index(val['image_id'])]['location']

                keypoints_table[self.loc_list.index(loc_),self.cat_list.index(target_class)] +=1

            if verbose:
                print('')
                print(keypoints_table.sum(), 'samples with keypoints')
                print('samples with keypoints / class')
                print(self.cat_list)
                print(keypoints_table.sum(axis=0))
                print('samples per class and location')
                print('locations',self.loc_list)
                print(keypoints_table.sum(axis=1))
                print(keypoints_table)
                print('percentage complete', (keypoints_table > 5).mean())

            if return_keypoint_dict:
                counts_keypoints = {k:n for k,n in zip(self.cat_list,keypoints_table.sum(axis=0))}
                return counts, counts_keypoints

        self.keypoint_table = keypoints_table

        return counts

def annotate_keypoints(img_path, output_path, bbox=None, dict_info=None):
    # Path to source video:
    # output_path = '/home/stephen/Desktop/clicks.csv'

    # Mouse callback function
    global keypoints_input_list
    global keypoints

    positions, keypoints_input_list = [], []
    keypoints = ['head', 'left-front-leg','right-front-leg', 'left-back-leg','right-back-leg', 'tail', 'body', 'left-wing', 'right-wing']

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

    dset = pd.DataFrame(columns=dict_out.keys())
    dset = dset.append(dict_out,ignore_index=True)
    # results = pd.concat((flags_dset,results),axis=1)

    # save or update the csv
    if os.path.isfile(output_path):
        dset_old = pd.read_csv(output_path)
        dset = pd.concat([dset_old,dset])
    dset.to_csv(output_path, index=False)

def save_hdf5(images_folder, save_path):

    import h5py
    import tqdm

    hf = h5py.File(save_path,'w')

    file_list = os.listdir(images_folder)
    for file in tqdm.tqdm(file_list):
        if file.endswith('.jpg') or file.endswith('.JPEG') or file.endswith('png'):
            val = Image.open(images_folder + '/' + file)
            hf.create_dataset(file, data=np.array(val))

    hf.close()
    print('saved!', save_path)



if __name__ == "__main__":

    # save_hdf5(images_folder='/scratch/data/CaltechCameraTraps/CCT20/eccv_18_all_images_sm',
    #  save_path='/scratch/data/CaltechCameraTraps/CCT20/eccv_18_all_images_sm.h5')

    cct = CCT20(root='/scratch/data',x_star='keypoints', datasplit='train', classes_train='boats')
    # cct = CCT20(root='/scratch/data',x_star='keypoints', datasplit='train', classes_train='boats')

    stats, stats_plus = cct.stats(return_keypoint_dict=True)
    df = pd.DataFrame(stats, index=['train'])
    df = pd.concat((df, pd.DataFrame(stats_plus, index=['train+'])))

    for i in range(10):
        a = cct.__getitem__(i)
    #
    cct = CCT20(root='/scratch/data',x_star='', datasplit='cis_val', classes_train='boats')

    stats = cct.stats()
    df = pd.concat((df, pd.DataFrame(stats, index=['cis_val'])))


    cct = CCT20(root='/scratch/data',x_star='bbox', datasplit='trans_val', classes_train='boats')

    stats = cct.stats()
    df = pd.concat((df, pd.DataFrame(stats, index=['trans_val'])))

    cct = CCT20(root='/scratch/data',x_star='bbox', datasplit='cis_test', classes_train='boats')

    stats = cct.stats()
    df = pd.concat((df, pd.DataFrame(stats, index=['cis_test'])))

    cct = CCT20(root='/scratch/data',x_star='bbox', datasplit='trans_test', classes_train='boats')

    stats = cct.stats()
    df = pd.concat((df, pd.DataFrame(stats, index=['trans_test'])))

    # df.to_csv('/scratch/data/summary_cct.csv')

    print(df)

    print('done!')
