

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
from tqdm import tqdm

try:
    import data.custom_transforms_pil as trpil
except ModuleNotFoundError:
    import custom_transforms_pil as trpil


def create_bbox_from_keypoint(x, y, size):
    x1, y1 = x - size, y - size
    x2, y2 = x + size, y + size
    return x1, y1, x2, y2

# (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
# IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
# IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
# DEFAULT_CROP_RATIO = 224/256


### CUB keypoints
# 1 back
# 2 beak
# 3 belly
# 4 breast
# 5 crown
# 6 forehead
# 7 left eye
# 8 left leg
# 9 left wing
# 10 nape
# 11 right eye
# 12 right leg
# 13 right wing
# 14 tail
# 15 throat

keypoints_CUB = ["back", "beak", "belly", "breast","crown","forehead","left eye","left leg","left wing","nape","right eye","right leg","right wing","tail","throat"]

keypoints_text = ''
for i, key in enumerate(keypoints_CUB):
    keypoints_text += f'{i+1}-{key}\n'



class Birds18(data.Dataset):
    folder = '2018'

    def __init__(self,
                 root,
                #  classes_train='all',
                 datasplit='train',
                 args=None,
                 trainvalindex=None,
                 x_star='keypoints',
                 download=False,
                 crop_size=(300,300)):
        super(Birds18, self).__init__()
        # self.as_pil = True

        self.root = join(os.path.expanduser(root), self.folder)
        self.args = args
        self.x_star = x_star
        self.year = '2018'
        self.is_debug = False

        # self.gpu = 0
        # self.classes_train = classes_train
        # self.is_remove_multiple_instances = True
        # self.balance_train_dataset = True

        if args is None:
            self.data_fraction = 1
            # self.every_n_is_val = 3
            # self.crop_train_around_bbox = False
            self.only_annotated_images = False
            self.class_balanced = False
        else:
            self.data_fraction = self.args.train_data_fraction
            # self.every_n_is_val = self.args.every_n_is_val
            # self.crop_train_around_bbox = 'cropbbox' in self.args.tag
            self.only_annotated_images = not 'fulldataset' in self.args.tag
            self.class_balanced = 'classbalanced' in self.args.tag

        # self.is_masked_images = 'masked' in self.x_star
        self.datasplit=datasplit
        self.crop_size = crop_size[0]
        assert crop_size[0] == crop_size[1]

        json_file_path = self._get_split_files(split = self.datasplit)

        with open(json_file_path) as json_file:
            annotations = json.load(json_file)

        names_aves = [x['name'] for x in annotations['categories'] if x['supercategory'] == 'Aves']        


        with open(f'{self.root}/categories.json') as json_file:
            self.categories = json.load(json_file)
        
        self.sci_names_inat = []
        self.category_info = dict()

        for category in self.categories:
            if category['supercategory'] == 'Aves':
                self.sci_names_inat.append(category['name'])
            self.category_info[category['id']] = category

        self.n_classes = len(self.sci_names_inat)

        self.split_image_file = [x for x  in  annotations['images'] if 'Aves' in x['file_name'].split('/')[1]]

        self.transform = self.transform_tr_pil if self.datasplit == 'train' else self.transform_val_pil

        # keypoints
        self.keypoints_file = os.path.join('./data/keypoints_inat2018.csv')
        self.read_keypoints_dict()
        self.keypoints_name = keypoints_CUB


        # print(len(self.keypoints_dict))

        if self.datasplit == 'train':
            y_train = np.array([self.__getlabel__(i) for i in range(len(self.split_image_file))])
            self.class_counts = np.array([np.sum(y_train==t) for t in range(self.n_classes)])

        if self.only_annotated_images and self.datasplit == 'train':
            image_id_list_key = list(self.keypoints_dict.keys())

            print('removing samples without keypoints annotation')
            self.split_image_file = [x for x in self.split_image_file if x['id'] in image_id_list_key]
        print(f'total samples in f{self.datasplit}',len(self.split_image_file))

        self.label_list = [self.__getlabel__(i) for i in range(len(self.split_image_file))]


        # list of images per class for balanced sampling
        self.per_class_img = [[] for _ in range(self.n_classes)]
        for i, lab in enumerate(self.label_list):
            self.per_class_img[lab].append(i)

        if self.datasplit == 'train':
            if self.class_balanced:
                weight = 1. / self.class_counts
                samples_weight = np.array([weight[t] for t in self.label_list])
                self.samples_weight = torch.from_numpy(samples_weight)

            self.compute_class_groups()

            self.classes_indices_sorted = np.argsort(self.class_counts)
            # self.per_class_img = [self.per_class_img[i] for i in indices]
            ## choose samples to be labelled with keypoints

            np.random.seed(1)
            self.per_class_img_shuffled  = [[] for _ in range(self.n_classes)]

            # n = 5
            for lab, samples in enumerate(self.per_class_img):
                samples_rand = np.array(samples)
                np.random.shuffle(samples_rand)
                # samples = [val[i] for i in rand_choice[:n]]
                self.per_class_img_shuffled[lab] = samples_rand

        self.stats(verbose=False)

        self.is_viz = False
        self.is_viz_after_labelling = True

    def compute_class_groups(self):        
        

        ## define class groups index
        self.class_groups = {
            'many': self.class_counts > 100,
            'mid': np.logical_and(self.class_counts <= 100, self.class_counts >= 20),
            'low': self.class_counts < 20,
            'low15': self.class_counts < 15
        }
        print(f'class groups {len(self.class_counts)}:')
        for k, v in self.class_groups.items():
            print(k, np.sum(v))


    def _get_split_files(self, split):

        if self.year == '2017':
            annotations_file = f'{self.root}/train_val2017/{split}2017.json'
        elif self.year == '2018':            
            annotations_file = f'{self.root}/{split}2018.json'
        else:
            raise NotImplemented
        return annotations_file


    def read_keypoints_dict(self):
        file = os.path.join(self.root, 'keypoints/keypoints_andres.csv')
        if os.path.isfile(file):
            ds = pd.read_csv(file, dtype={'id': int}).set_index('id')
        else:
            ds = None
        file = os.path.join(self.root, 'keypoints/keypoints_stefano.csv')
        if os.path.isfile(file):
            ds_ = pd.read_csv(file, dtype={'id': int}).set_index('id')
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
            return min(1000,len(self.split_image_file))
        return len(self.split_image_file)

    def __getlabel__(self, index):
        image_dict = self.split_image_file[index]

        image_path = image_dict['file_name']
        if self.year == '2018':
            id_class = image_path.split('/')[2]
            sci_name = self.category_info[int(id_class)]['name']
        else:
            sci_name = image_path.split('/')[2]
        # id_sample = image_dict['id']

        label = self.sci_names_inat.index(sci_name)
        return label

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_dict = self.split_image_file[index]

        image_path = image_dict['file_name']

        label = self.label_list[index]

        image_path_full = os.path.join(self.root, image_path)

        image = Image.open(image_path_full).convert('RGB')

        # if self.balance_train_dataset and self.datasplit=="train":
        #     idx1 = 14
        #     while idx1 == 14:
        #         idx1 = torch.randint(high=16,size=(1,))
        #     idx2 = torch.randint(high=len(self.per_class_img[idx1]),size=(1,))
        #     index = self.per_class_img[idx1][idx2]

        dict_out = {}

        ## TODO
        teacher = None
        bbox = None

        r = 1
        if 'keypoints-annotate' in self.x_star:
            # current_count = self.keypoint_table[self.loc_list.index(loc_), self.cat_list.index(target_class_name)]

            # if current_count >= 5 or target_class in [33, 30]:
            #     return -1

            self.read_keypoints_dict()
            exists = image_dict['id'] in self.keypoints_dict.keys()

            if not exists:
                ## annotate image:
                # print('category: ',  self.categories_dict[target_class], 'loc:', loc_ , 'current count: ', current_count)
                annotate_keypoints(image_path_full, output_path=self.keypoints_file,
                                    bbox=bbox, dict_info=image_dict)
                if self.is_viz_after_labelling:
                    self.read_keypoints_dict()
                    self.visualize_sample(image, image_dict, bbox, verbose=False)
            elif self.is_viz:
                self.visualize_sample(image, image_dict, bbox)

            ## check if all -1s
            keypoints_none = [ast.literal_eval(val) == (-1,-1)  for key, val in self.keypoints_dict[image_dict['id']].items() if key in self.keypoints_name]
            if np.all(keypoints_none):
                return -1
            
            return 0

        if self.x_star == 'bbox':
            raise NotImplemented
            bbox_mask = Image.new("L", image.size)
            if bbox is not None:
                draw = ImageDraw.Draw(bbox_mask)
                draw.rectangle(bbox,fill=255)

            teacher = bbox_mask if self.as_pil else np.array(bbox_mask)
            teacher = [teacher]

        elif 'keypoints' in self.x_star:

            mask_out = []
            img_id = image_dict['id']
            # annotation_id = self.annotations_dict[image_dict['id']]['id']
            if not img_id in self.keypoints_dict.keys():
                mask_out = [Image.new("L", image.size) for _ in self.keypoints_name]
            else:
                for k in self.keypoints_name:
                    bbox_mask = Image.new("L", image.size)
                    draw = ImageDraw.Draw(bbox_mask)

                    if k in self.keypoints_dict[img_id].keys():
                        val = self.keypoints_dict[img_id][k]
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

            # if self.as_pil:

            teacher = mask_out
            # else:
                # teacher = np.stack([np.array(x) for x in mask_out], axis=-1)
        else:
            teacher = None


        dict_out['s'] = image
        # if self.as_pil:
        if True:
            if teacher is not None:
                for i, t in enumerate(teacher):
                    dict_out[f't_{i}'] = t

            dict_out = self.transform(dict_out)

            if teacher is not None:
                list_cat = [dict_out[f't_{i}'] for i in range(len(teacher))]
                dict_out['t'] = torch.cat(list_cat, dim=0)

        dict_out['label'] = torch.tensor(label)

        return dict_out


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

        if self.args.is_crop or (self.datasplit in ['train','val']):
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

    def visualize_sample(self,image, image_dict, bbox, verbose = True):
        
        if verbose: print(f"visualizing sample id {image_dict['id']}, press esc to continue...")
        r = 1
        draw = ImageDraw.Draw(image)
        if bbox is not None:
            draw.rectangle(bbox,outline=255)
        for key, val in self.keypoints_dict[image_dict['id']].items():
            if key in self.keypoints_name:

                val = ast.literal_eval(val)

                if val[0] != -1:  # not missing
                    val = (val[0] / r, val[1] / r)

                    rect_ = create_bbox_from_keypoint(val[0], val[1], size=10)

                    draw.rectangle(rect_, outline=255)

                    draw.text((val[0],val[1]),str(self.keypoints_name.index(key)+1))
            draw.text((0,0),keypoints_text + f"\nsample id {image_dict['id']}")
        # image.resize((1024, 1024))
        while True:
            cv2.imshow('img', np.array(image)[:,:,(2,1,0)])
            
            # Wait, and allow the user to quit with the 'esc' key
            k = cv2.waitKey(1)
            # If user presses 'esc' break
            # if k == 27 or len(keypoints_input_list) == len(keypoints):
            if k == 27:
                break
        cv2.destroyAllWindows()


    def stats(self, verbose=True, return_keypoint_dict=False):
        counts = {}
        for label, val in  enumerate(self.per_class_img):
            counts[label] = len(val)

        if verbose:
            print(f"\t {self.datasplit}: {len(self)} samples spanning {len(counts.keys())} classes "
                  f"(avg {float(len(self))/float(len(counts.keys())):.2f} per class)")
            # print(counts)
            print('\n')

        keypoints_table = np.zeros((self.n_classes), dtype=np.int32)

        if self.datasplit == 'train':

            image_id_list = [x['id'] for x in self.split_image_file]

            for key, val in self.keypoints_dict.items():

                index = image_id_list.index(int(key))
                target_class = self.__getlabel__(index)

                keypoints_table[target_class] +=1

            if verbose:
                print('')
                print(keypoints_table.sum(), 'samples with keypoints')
                # print('samples with keypoints / class')
                # print(self.cat_list)
                # print(keypoints_table.sum(axis=0))
                # print('samples per class and location')
                # print('locations',self.loc_list)
                # print(keypoints_table.sum(axis=1))
                # print(keypoints_table)
                # print('percentage complete', (keypoints_table > 5).mean())

            if return_keypoint_dict:
                counts_keypoints = keypoints_table
                # counts_keypoints = {k:n for k,n in zip(self.cat_list,keypoints_table.sum(axis=0))}
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

    keypoints = ["back", "beak", "belly", "breast","crown","forehead","left eye","left leg","left wing","nape","right eye","right leg","right wing","tail","throat"]
    pbar = tqdm(total=len(keypoints))

    def callback(event, x, y, flags, param):
        global keypoints_input_list
        if len(keypoints_input_list) < len(keypoints):
            if event == 1 or event == 3:
                if event == 3:
                    x, y = -1, -1
                # print("\t", x,y)
                pbar.update() 
                description = f" previous: {keypoints[len(keypoints_input_list)]} ({x},{y})"
                keypoints_input_list.append((x, y))
                if len(keypoints_input_list) < len(keypoints):
                    # print(keypoints[len(keypoints_input_list)])
                    description = f"{keypoints[len(keypoints_input_list)]}" + description
                pbar.set_description(description)

            elif event == 2:
                keypoints_input_list.pop(-1)
                pbar.update(-1)
                if len(keypoints_input_list) < len(keypoints):
                    print('\tretype keypoint', keypoints[len(keypoints_input_list)])
        elif event != 0:
                pbar.set_description('done!, press esc to continue to next image')

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1200, 1200)
    cv2.setMouseCallback('img', callback)

    img = cv2.imread(img_path)

    if bbox is not None:
        img = cv2.rectangle(img.copy(), (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    # print(keypoints[0])
    pbar.update()
    pbar.set_description(keypoints[0])

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
    pbar.close()
    dict_out = dict_info
    for i, position in enumerate(keypoints_input_list):
        # x, y = position[0], position[1]
        dict_out[keypoints[i]] = [position]

    dset = pd.DataFrame.from_dict(dict_out)

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

    user = 'andres'
    assert user in ['andres','stefano']

    birds = Birds18(root='/scratch/data/iNaturalist',x_star='keypoints-annotate', datasplit='train', user=user)
    birds.is_viz = False
    birds.is_viz_after_labelling = True

    stats, stats_plus = birds.stats(return_keypoint_dict=True)
    df = pd.DataFrame(stats, index=['train'])

    if user == 'andres':
        classes_to_label = birds.classes_indices_sorted # [0::2]
    elif user == 'stefano':
        classes_to_label = birds.classes_indices_sorted[1::2]

    bar_classes = tqdm(enumerate(classes_to_label), total=len(classes_to_label))
    counter = 0
    for j, y in bar_classes:
        N_y = len(birds.per_class_img[y])
        description = f'current class: {birds.sci_names_inat[y]} ({N_y})'
        bar_classes.set_description(description)
        
        # print(f'current class: {birds.sci_names_inat[y]} \t total train samples: {N_y}  \t{j}/{len(classes_to_label)}')
        n_label = min(5,N_y)
        i = 0
        while i < n_label:
            index = birds.per_class_img_shuffled[y][i]
            counter +=1
            id = birds.split_image_file[index]['id']
            if not id in birds.keypoints_dict.keys():
                # print(f'\tcurrent sample {counter} (id={id})')
                bar_classes.set_description(description+ f' {i} - current sample {counter} (id={id})')
            result = birds.__getitem__(index)
            if result == -1 and (N_y > n_label +1):
                n_label +=1
                # print('all -1s found, adding one sample more')
                bar_classes.set_description(description+ ' - all -1s found, adding one sample more')

            i += 1
        # print('labeled samples: ', n_label)
        bar_classes.set_description(description+ f'labeled samples: {n_label}')
        
    # df.to_csv('/scratch/data/summary_cct.csv')

    print(df)

    print('done!')
