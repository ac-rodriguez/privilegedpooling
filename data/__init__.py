from random import Random
import socket
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
import torch
from data import stanford_dogs_data, birds_data, animalparts, pascal_voc2012, inaturalist_birds, cct_20, FGVC_data
from data import inaturalist18_birds

def get_basepath():

    '''
    Update this to where you want to store your data
    '''
    PATH='/scratch'
    
    assert os.path.exists(PATH), f'{PATH} does not exist'

    return PATH


def get_trainvalindex(seed, N, fraction):
    trainvalindex = None
    s0 = None
    if fraction < 1:
        if seed is not None:
            s0 = np.random.get_state()
            np.random.seed(seed)

        ids = np.random.choice(N, int(N * fraction), replace=False)
        trainvalindex = [x in ids for x in range(N)]
    return trainvalindex, s0

IS_TEST_AS_VAL = True

def make_data_loader(args):

    batch_size = args.batch_size
    if args.model == 'MINE':
        batch_size *= 2
    is_x_star_val = args.model in ['upper','teacher']
    is_attributes = args.model == 'outer_class' or args.model == 'multi_class'

    kwargs = {'num_workers': args.workers, 'pin_memory': True}

    cropsize = (args.crop_size, args.crop_size)

    if args.dataset == 'dogs':
        n_class = 120
        N_trainval = 12000
        trainvalindex = None
        if args.train_data_fraction < 1:
            if args.np_seed is not None:
                s0 = np.random.get_state()
                np.random.seed(args.np_seed)
            ids = np.random.choice(N_trainval,int(N_trainval*args.train_data_fraction),replace=False)
            trainvalindex = [x in ids for x in range(N_trainval)]

        root = get_basepath()+'/data'
        size_s = 3
        size_t = 1

        traindataset = stanford_dogs_data.Dogs(root=root,args=args, bbox_as_mask=True, datasplit='train',
                                               trainvalindex=trainvalindex, crop_size=cropsize)
        traindataset.stats()
        valdataset = stanford_dogs_data.Dogs(root=root,args=args, bbox_as_mask=is_x_star_val, datasplit='val',
                                             trainvalindex=trainvalindex, crop_size=cropsize)
        valdataset.stats()
        testdataset = stanford_dogs_data.Dogs(root=root,args=args, bbox_as_mask=is_x_star_val, datasplit='test',
                                              crop_size=cropsize)
        testdataset.stats()

        if args.train_data_fraction < 1 and args.np_seed is not None:
            np.random.set_state(s0)

        train_dataloader = DataLoader(traindataset, batch_size=batch_size,
                                           shuffle=True, **kwargs)
        val_dataloader = DataLoader(valdataset, batch_size=batch_size, **kwargs)
        test_dataloader = DataLoader(testdataset, batch_size=batch_size, **kwargs)

        return train_dataloader,val_dataloader, test_dataloader, n_class, (size_s, size_t)


    elif args.dataset == 'birds':
        n_class = 200
        N_trainval = 5994

        trainvalindex, s0 = get_trainvalindex(args.np_seed,N=N_trainval,fraction=args.train_data_fraction)

        root = get_basepath()+'/data/CUB_200_2011/'
        size_s = 3

        if args.x_star in ['masked-keypoints','masked-bbox']:
            size_t = 3
        elif args.x_star == 'attributes' or is_attributes:
            size_t = 312
        elif args.x_star in ['keypoints', 'keypoints-catattributes']:
            size_t = 15
        elif args.x_star in ['keypoints-bbox', 'bbox-keypoints']:
            size_t = 16
        elif args.x_star == 'bbox':
            size_t = 1
        else:
            raise NotImplementedError

        traindataset = birds_data.Birds(root=root,args=args, datasplit='train', trainvalindex=trainvalindex,
                                        crop_size=cropsize, x_star=args.x_star)
        traindataset.stats()
        train_dataloader = DataLoader(traindataset, batch_size=batch_size,
                                      shuffle=True, **kwargs)

        x_star_val = args.x_star if is_x_star_val else ''

        testdataset = birds_data.Birds(root=root,args=args, datasplit='test', trainvalindex=trainvalindex,
                                      crop_size=cropsize, x_star=args.x_star)
        testdataset.stats()
        test_dataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, **kwargs)


        if not IS_TEST_AS_VAL:
            valdataset = birds_data.Birds(root=root,args=args,datasplit='val', trainvalindex=trainvalindex,
                                          crop_size=cropsize,  x_star=args.x_star) #x_star=x_star_val)
            valdataset.stats()
            val_dataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, **kwargs)
        else:
            val_dataloader = test_dataloader
        IS_INAT_AS_TEST = True
        if IS_INAT_AS_TEST:
            print('Using iNaturalist as test set')
            testdataset = inaturalist_birds.Birds(root=get_basepath() + '/data/iNaturalist/', args=args, datasplit='val',
                                                  crop_size=cropsize)
            testdataset.stats()
            test_dataloader = DataLoader(testdataset, batch_size=batch_size,
                                         shuffle=False, **kwargs)

        if s0 is not None:
            np.random.set_state(s0)


        if 'catattributes' in args.x_star:
            return train_dataloader,val_dataloader, test_dataloader, n_class, (size_s, size_t, traindataset.get_size_attributes())
        else:
            return train_dataloader,val_dataloader, test_dataloader, n_class, (size_s, size_t)

    elif args.dataset == 'cct20':
        n_class = 16
        N_trainval = 11685 # if removing samples w/o bbox
        if args.train_data_fraction < 1.0:
            raise NotImplementedError

        trainvalindex, s0 = get_trainvalindex(args.np_seed, N=N_trainval, fraction=args.train_data_fraction)

        root = get_basepath()+'/data/'
        size_s = 3

        if args.x_star in ['keypoints']:
            size_t = 7
        elif args.x_star == 'keypoints-bbox':
            size_t = 8
        elif args.x_star == 'bbox':
            size_t = 1
        else:
            raise NotImplementedError

        traindataset = cct_20.CCT20(root=root,args=args, datasplit='train', trainvalindex=trainvalindex,
                                        crop_size=cropsize, x_star=args.x_star)
        traindataset.stats()
        train_dataloader = DataLoader(traindataset, batch_size=batch_size,
                                      shuffle=True, **kwargs)

        valdataset = cct_20.CCT20(root=root,args=args,datasplit='cis_val', trainvalindex=trainvalindex,
                                      crop_size=cropsize,  x_star='')
        valdataset.stats()
        val_dataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, **kwargs)

        testdataset = cct_20.CCT20(root=root,args=args, datasplit='cis_test', trainvalindex=trainvalindex,
                                      crop_size=cropsize, x_star=args.x_star)
        testdataset.stats()
        test_dataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, **kwargs)

        if s0 is not None:
            np.random.set_state(s0)

        return train_dataloader,val_dataloader, test_dataloader, n_class, (size_s, size_t)

    elif 'cct20' in args.dataset:
        n_class = 16
        N_trainval = 11685 # if removing samples w/o bbox
        if args.train_data_fraction < 1.0:
            raise NotImplementedError

        trainvalindex, s0 = get_trainvalindex(args.np_seed, N=N_trainval, fraction=args.train_data_fraction)

        root = get_basepath()+'/data/'
        size_s = 3

        if args.x_star in ['keypoints']:
            size_t = 7
        elif args.x_star == 'keypoints-bbox':
            size_t = 8
        elif args.x_star == 'bbox':
            size_t = 1
        else:
            raise NotImplementedError

        if args.dataset == 'cct20_trans':
            dtype = 'trans_test'
        elif args.dataset == 'cct20_cis':
            dtype = 'cis_test'
        else:
            raise NotImplemented

        testdataset = cct_20.CCT20(root=root,args=args, datasplit=dtype, trainvalindex=trainvalindex,
                                      crop_size=cropsize, x_star=args.x_star)
        testdataset.stats()
        test_dataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, **kwargs)

        if s0 is not None:
            np.random.set_state(s0)

        return test_dataloader, None, test_dataloader, n_class, (size_s, size_t)
    elif args.dataset == 'inat2017':
        n_class = 200

        root = get_basepath() + '/data/iNaturalist/'
        size_s = 3

        if args.x_star in ['masked-keypoints','masked-bbox']:
            size_t = 3
        elif args.x_star in ['keypoints', 'keypoints-catattributes']:
            size_t = 15
        elif args.x_star in ['keypoints-bbox', 'bbox-keypoints']:
            size_t = 16
        elif args.x_star == 'bbox':
            size_t = 1
        else:
            raise NotImplementedError

        testdataset = inaturalist_birds.Birds(root=root, args=args, datasplit='val',
                                        crop_size=cropsize)
        testdataset.stats()
        test_dataloader = DataLoader(testdataset, batch_size=batch_size,
                                      shuffle=False, **kwargs)

        return test_dataloader, None, test_dataloader, n_class, (size_s, size_t)
    elif args.dataset == 'birdstest':
        n_class = 200

        size_s = 3
        if args.x_star in ['masked-keypoints','masked-bbox']:
            size_t = 3
        elif args.x_star in ['keypoints', 'keypoints-catattributes']:
            size_t = 15
        elif args.x_star in ['keypoints-bbox', 'bbox-keypoints']:
            size_t = 16
        elif args.x_star == 'bbox':
            size_t = 1
        else:
            raise NotImplementedError

        root = get_basepath()+'/data/CUB_200_2011/'
        testdataset = birds_data.Birds(root=root,args=args, datasplit='test', trainvalindex=None,
                                      crop_size=cropsize, x_star=args.x_star)
        testdataset.stats()
        test_dataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, **kwargs)

        return test_dataloader, None, test_dataloader, n_class, (size_s, size_t)

    elif args.dataset == 'animalparts':
        assert not args.pretrained
        n_class = 158
        N_trainval = 7123
        # N_trainval = 13473
        trainvalindex, s0 = get_trainvalindex(args.np_seed,N=N_trainval,fraction=args.train_data_fraction)

        root = get_basepath() + '/data/'
        size_s = 3
        if args.x_star in ['masked-keypoints','masked-bbox']:
            size_t = 3
        elif args.x_star == 'keypoints':
            size_t = 2
        elif args.x_star == 'keypoints-bbox':
            size_t = 3
        elif args.x_star == 'bbox':
            size_t = 1
        else:
            raise NotImplementedError

        traindataset = animalparts.AnimalParts(root=root, args=args, datasplit='train',
                                               trainvalindex=trainvalindex, crop_size=cropsize, x_star=args.x_star)
        traindataset.stats()
        train_dataloader = DataLoader(traindataset, batch_size=batch_size,
                                      shuffle=True, **kwargs)

        x_star_val = args.x_star if is_x_star_val else ''

        testdataset = animalparts.AnimalParts(root=root, args=args,datasplit='test',
                                              trainvalindex=trainvalindex, crop_size=cropsize, x_star=args.x_star)
        testdataset.stats()
        test_dataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, **kwargs)

        if not IS_TEST_AS_VAL:
            valdataset = animalparts.AnimalParts(root=root, args=args, datasplit='val',
                                                 trainvalindex=trainvalindex, crop_size=cropsize, x_star=args.x_star)
            valdataset.stats()
            val_dataloader = DataLoader(valdataset, batch_size=batch_size,shuffle=True, **kwargs)
        else:
            val_dataloader = test_dataloader

        if s0 is not None:
            np.random.set_state(s0)


        return train_dataloader, val_dataloader, test_dataloader, n_class, (size_s, size_t)

    elif args.dataset == 'voc2012':
        n_class = 20
        N_trainval = 5717
        # N_trainval = 13473
        trainvalindex, s0 = get_trainvalindex(args.np_seed,N=N_trainval,fraction=args.train_data_fraction)

        root = get_basepath() + '/data/'
        size_s = 3
        if args.x_star in ['masked-bbox']:
            size_t = 3
        elif args.x_star == 'bbox':
            size_t = 1
        elif args.x_star == 'keypoints':
            size_t = 5 # created from semantic masks
        elif args.x_star == 'keypoints-bbox':
            size_t = 6 # created from semantic masks
        else:
            raise NotImplementedError

        traindataset = pascal_voc2012.VOCDetection(root=root, args=args, datasplit='train',
                                               trainvalindex=trainvalindex, crop_size=cropsize, x_star=args.x_star)
        traindataset.stats()
        train_dataloader = DataLoader(traindataset, batch_size=batch_size,
                                      shuffle=True, **kwargs)

        x_star_val = args.x_star if is_x_star_val else ''

        testdataset = pascal_voc2012.VOCDetection(root=root, args=args,datasplit='test',
                                              trainvalindex=trainvalindex, crop_size=cropsize, x_star=args.x_star)
        testdataset.stats()
        test_dataloader = DataLoader(testdataset, batch_size=batch_size,shuffle=True, **kwargs)

        if not IS_TEST_AS_VAL:
            valdataset = pascal_voc2012.VOCDetection(root=root, args=args, datasplit='val',
                                                 trainvalindex=trainvalindex, crop_size=cropsize, x_star=args.x_star)
            valdataset.stats()
            val_dataloader = DataLoader(valdataset, batch_size=batch_size,shuffle=True, **kwargs)
        else:
            val_dataloader = test_dataloader

        if s0 is not None:
            np.random.set_state(s0)

        return train_dataloader, val_dataloader, test_dataloader, n_class, (size_s, size_t)

    elif args.dataset == 'vocboats':
        n_class = 2
        N_trainval = 5717
        # N_trainval = 13473
        trainvalindex, s0 = get_trainvalindex(args.np_seed,N=N_trainval,fraction=args.train_data_fraction)

        root = get_basepath() + '/data/'
        size_s = 3
        if args.x_star in ['masked-bbox']:
            size_t = 3
        elif args.x_star == 'bbox':
            size_t = 1
        elif args.x_star == 'keypoints':
            size_t = 5 # created from semantic masks
        elif args.x_star == 'keypoints-bbox':
            size_t = 6 # created from semantic masks
        else:
            raise NotImplementedError

        traindataset = pascal_voc2012.VOCDetection(root=root, args=args, datasplit='train',classes_train='boats',
                                               trainvalindex=trainvalindex, crop_size=cropsize, x_star=args.x_star)
        traindataset.stats()
        train_dataloader = DataLoader(traindataset, batch_size=batch_size,
                                      shuffle=True, **kwargs)

        x_star_val = args.x_star if is_x_star_val else ''

        testdataset = pascal_voc2012.VOCDetection(root=root, args=args,datasplit='test',classes_train='boats',
                                              trainvalindex=trainvalindex, crop_size=cropsize, x_star=args.x_star)
        testdataset.stats()
        test_dataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, **kwargs)

        if not IS_TEST_AS_VAL:
            valdataset = pascal_voc2012.VOCDetection(root=root, args=args, datasplit='val',
                                                 trainvalindex=trainvalindex, crop_size=cropsize, x_star=args.x_star)
            valdataset.stats()
            val_dataloader = DataLoader(valdataset, batch_size=batch_size,shuffle=True, **kwargs)
        else:
            val_dataloader = test_dataloader

        if s0 is not None:
            np.random.set_state(s0)

        return train_dataloader, val_dataloader, test_dataloader, n_class, (size_s, size_t)

    elif args.dataset == 'boat_test' or args.dataset == 'boat_test_binary':
        n_class = 20
        if args.dataset == 'boat_test_binary':
            n_class = 2

        root = get_basepath() + '/data/'
        size_s = 3
        if args.x_star in ['masked-bbox']:
            size_t = 3
        elif args.x_star == 'bbox':
            size_t = 1
        elif args.x_star == 'keypoints':
            size_t = 5 # created from semantic masks
        else:
            raise NotImplementedError

        testdataset = pascal_voc2012.BiasedBoats(root=root, args=args,datasplit='test',nclass=n_class,
                                              trainvalindex=None, crop_size=cropsize, x_star=args.x_star)

        test_dataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False,**kwargs)

        return test_dataloader, None, test_dataloader, n_class, (size_s, size_t)
    elif args.dataset == 'inat2018':
        n_class = 1258
        N_trainval = -1 # if removing samples w/o bbox
        if args.train_data_fraction < 1.0:
            raise NotImplementedError

        # trainvalindex, s0 = get_trainvalindex(args.np_seed, N=N_trainval, fraction=args.train_data_fraction)

        root = get_basepath() + '/data/iNaturalist/'

        size_s = 3

        if args.x_star in ['keypoints']:
            size_t = 15
        else:
            raise NotImplementedError

        traindataset = inaturalist18_birds.Birds18(root=root,args=args, datasplit='train',
                                        crop_size=cropsize, x_star=args.x_star)

        if traindataset.class_balanced:
            print('using class balanced training')
            sampler = WeightedRandomSampler(traindataset.samples_weight.double(), len(traindataset.samples_weight))
        else:
            sampler = RandomSampler(traindataset)
        # train_dataloader = traindataset.loader
        # traindataset.stats()
        train_dataloader = DataLoader(traindataset, batch_size=batch_size, sampler=sampler, **kwargs)

        if IS_TEST_AS_VAL:
            valdataset = inaturalist18_birds.Birds18(root=root,args=args, datasplit='val',
                                            crop_size=cropsize, x_star=args.x_star)
            
            val_dataloader = DataLoader(valdataset, batch_size=batch_size,
                                            shuffle=False, **kwargs)

            print('using val as test [!]')
            test_dataloader = val_dataloader
        else:
            raise NotImplemented
        # if s0 is not None:
            # np.random.set_state(s0)

        return train_dataloader,val_dataloader, test_dataloader, n_class, (size_s, size_t)
    else:
        raise NotImplementedError(args.dataset)
