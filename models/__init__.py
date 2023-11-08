import numpy as np
from models import model_base, model_teacher_student, dann, lupi_dropout, model_reconstruct, model_upper_student,\
    attention, model_attention_map, model_WS_DAN, model_s3n # , model_rnnattention_map,model_attention_map_adaptgrad
from models import model_transformer # ,model_spatial_attention
from models.iSQRT import model_init
from models.iSQRT.src.representation.MPNCOV import MPNCOV
from models import model_attentionalpooling_action

def get_model_cnn(model_name, n_class, size_s, size_t=None, size_attributes=None, args=None):

    if args.bilinear:  ### this is not converted
        assert args.backbone == 'resnet50'
        if model_name == 'student' or model_name == 'dist-vanilla':
            model = model_base.CNNBilinear(output_size=n_class, input_size=size_s,
                               input_keys=['s'], sync_bn=args.sync_bn, d_lowrank=args.dim_lowrank, pretrained=args.pretrained)
        elif model_name in ['L2_s', 'L2_st', 'DANN', 'clusterIB']:
            if args.use_upper:
                raise NotImplementedError
            else:
                model = model_teacher_student.CNNBilinear(output_size=n_class, size_s=size_s, size_t=size_t,
                                                   backbone='resnet50',
                                                   sync_bn=args.sync_bn, pretrained=args.pretrained)
        elif model_name == 'multi_class':
            model = model_teacher_student.CNNBilinear_multi(output_size=n_class, size_s=size_s, size_t=size_t,
                                                   backbone='resnet50',
                                                   sync_bn=args.sync_bn, pretrained=args.pretrained)
        elif model_name == 'outer_class':
            model = model_teacher_student.CNNBilinear_outer(output_size=n_class, size_s=size_s, size_t=size_t,
                                                   backbone='resnet50',
                                                   sync_bn=args.sync_bn, pretrained=args.pretrained)
        else:
            raise NotImplementedError
        return model
    # BASELINES
    ## Upper bound of a model with all the information
    if model_name == 'upper':
        model = model_upper_student.CNN1(output_size=n_class,size_s=size_s, size_t=size_t, args=args)
    elif model_name == 'student' or model_name == 'dist-vanilla':
        model = model_base.CNN(output_size=n_class,input_size=size_s, args=args)

    elif model_name == 'teacher':
        model = model_base.CNN(output_size=n_class,input_size=size_t, args=args,input_keys=['t'])

    elif model_name in ['L2_s', 'L2_st', 'DANN', 'clusterIB', 'attention', 'attention_sample']:
        model = model_teacher_student.CNN1(output_size=n_class,size_s=size_s,size_t=size_t, args=args)
    elif model_name == 'dropout':
        model = lupi_dropout.CNN1(output_size=n_class,size_s=size_s,size_t=size_t, args=args)
    elif model_name == 'multi_task_mask':
        model = model_reconstruct.CNN_multitask(output_size=n_class,size_s=size_s,size_t=size_t,args=args)
    elif model_name == 'attention_map':
        model = model_attention_map.CNN1(output_size=n_class, size_s=size_s, size_t=size_t,size_attributes=size_attributes,args=args)
    elif model_name == 'attention_action':
        model = model_attentionalpooling_action.CNN1(output_size=n_class, size_s=size_s, size_t=size_t,size_attributes=size_attributes,args=args)
    elif model_name == 'transformer':
        model = model_transformer.CNN1(output_size=n_class, input_size=size_s, args=args)
    elif model_name == 'WS_DAN':
        model = model_WS_DAN.CNN1(output_size=n_class, size_s=size_s,args=args)
    elif model_name == 'iSQRT':
        representation = {'function': MPNCOV,
                          'iterNum': 5,
                          'is_sqrt': True,
                          'is_vec': True,
                          'input_dim': 2048,
                          'dimension_reduction': None if args.pretrained else 256}
        assert ((args.backbone == "mpncovresnet50") or (args.backbone == "mpncovresnet101") )
        assert (args.pretrained == True)
        model = model_init.get_model(args.backbone, representation, n_class, False, pretrained=True)
    elif model_name == 'iSQRT_attention':
        model = model_attention_map.CNN1(output_size=n_class, size_s=size_s, size_t=size_t,size_attributes=size_attributes,args=args)
    elif model_name == 'S3N':
        model = model_s3n.S3N(num_classes=n_class, args=args)
    else:
        raise NotImplementedError
    return model


def get_phi_model(model_name, n_class=None, args=None):

    n_feat = max(32,args.latent_size//10)
    if model_name in ['DANN', 'DANN_drop']:
        phi_model = dann.DANN(args.latent_size, n_feat=n_feat)
    elif model_name in ['attention_sample']:
        phi_model = attention.attentionNetsample(args.latent_size, fullattn=args.is_fullattention)
    else:
        phi_model = None

    return phi_model



def get_default_model_version(args):

    if args.tag in ['default','']:
        tag = ''
        if args.model == 'attention_map':
            defaults_dict = {
                'inat2018':'_maxpool_dropout_classbalanced_fulldataset_regularizedvariance',
                'birds':'_maxpool_dropout_regularizedvariance',
                'cct':'maxpool_dropout_cropbbox_fulldataset'
                }
        elif args.model == 'iSQRT_attention':
            defaults_dict = {
                'inat2018':'_dropout_classbalanced_fulldataset_regularizedvariance',
                'birds':'_dropout_regularizedvariance',
                'cct':'_dropout_fulldataset'
                }    
        tag = defaults_dict[args.dataset]

        if tag !='':
            print('model with default version:', tag)

        return tag
    else:
        return args.tag
