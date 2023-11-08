import os, sys, argparse
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data import make_data_loader
from models import get_model_cnn, get_phi_model, get_default_model_version
from models.sync_batchnorm.replicate import patch_replication_callback

from utils.metrics import EvaluatorImgClass
from utils import losses as losses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import save_parameters, get_path, Saver
from utils import plots
import signal

parser = argparse.ArgumentParser(description="Information based data fusion - Infusion",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

### general and logging parameters ############################################
parser.add_argument("--no-cuda", action="store_true", default=False,
                    help="disables CUDA training")
parser.add_argument("--gpu-ids", type=str, default="0",
                    help="use which gpu to train, must be a comma-separated list of integers only (default=0)")
parser.add_argument("--sync-bn", type=bool, default=None,
                    help="whether to use sync bn (default: auto)")
parser.add_argument("--np-seed", default=None,type=int,
                    help="random seed for numpy")
parser.add_argument("--is-overwrite", default=False, action="store_true",
                    help="Delete model_dir before starting training from iter 0. Overrides --is-restore flag")
parser.add_argument("--save-dir", default='./',
                    help="Path to directory where models should be saved")
parser.add_argument('--is-save-model', action='store_true', default=False,
                    help='whether to save the model weights')
parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
parser.add_argument("--testing-model", default="last",choices=['last','best'],
                    help="which model to use for testing and saving")
parser.add_argument('--test-only', action='store_true', default=False,
                    help='skip training and predict on test dataset only')
parser.add_argument('--message', default=None, type=str,
                    help='personalized message')
parser.add_argument('--tag', default="",type=str)


#### general model parameters #################################################
parser.add_argument("--model", default="attention_map",
                    help="Model Architecture to be used")
parser.add_argument("--use-upper", action='store_true', default=False,
                    help="uses upper model (x and x*) instead of teacher model (x*)")
parser.add_argument("--backbone", default="resnet101_original",
                    help="backbone network to be used")
parser.add_argument("--final-stride", default=1, type=int, choices=[1,2],
                    help="final-stride used to construct the feature map")
parser.add_argument("--is-not-pretrained",'--not-pretrained', default=True, action="store_false", dest='pretrained',
                    help="load backbone pretrained model on ImageNet")
parser.add_argument('--latent-size', type=int, default=-1,
                    help="dimension of the latent size (-1: auto)")


#### data parameters ##########################################################
parser.add_argument("--dataset", default="birds",help="Dataset to be used")
parser.add_argument("--x-star", default="keypoints",choices=['keypoints','masked-keypoints','bbox','masked-bbox',
                                                             'attributes', 'catattributes','keypoints-attributes',
                                                             'keypoints-catattributes', 'keypoints-bbox'],
                    help="type of x-star to be used")
parser.add_argument("--workers", type=int, default=4,metavar="N",
                    help="dataloader threads")
parser.add_argument("--batch-size", type=int, default=5)
parser.add_argument("--accum-grad-iters", type=int, default=2)
parser.add_argument("--crop-size", type=int, default=448)
parser.add_argument("--crop-size-transformer", type=int, default=448)

parser.add_argument("--not-random-crop", action="store_false", default=True,dest="random_crop",
                    help="center-crop instdead of random crop at train time")
parser.add_argument("--every-n-is-val", type=int, default=5,
                    help="take every n sample as val sample from trainval.txt list")
parser.add_argument("--train-data-fraction", default=1, type=float,
                    help="Percent of train-data to be used")
parser.add_argument("--n-shot", type=int, default=None,
                    help="take only n samples from each class.")
parser.add_argument('--n-shot-experiment-id', type=int, default=1,
                    help="experiment randomization to use")
parser.add_argument("--n-shot-only-novel", action='store_true', default=False,
                    help="do n-shot only for novel classes")
parser.add_argument("--do-not-crop", action='store_false', default=True,dest="is_crop",
                    help="applies cropping for images of the test set")
parser.add_argument("--do-not-crop-train", action='store_false', default=True,dest="is_crop_train",
                    help="applies cropping for images for training")


#### optimizer parameters #####################################################
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--optimizer',default='sgd', choices=['sgd','adam'])
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr-backbone-multiplier', type=float, default=0.1,
                    help='learning rate multiplier of the backbone layer')
parser.add_argument('--max-lr-backbone', type=float, default=1.,
                    help='maximum learning rate applied to the backbone')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--w-decay', type=float, default=1e-4)
parser.add_argument('--lr-scheduler', type=str, default='exp-iter',choices=['poly', 'step', 'cos', 'exp', 'exp-iter','step-iter'],
                    help='lr scheduler mode')
parser.add_argument('--lr-step', type=int, default=1000,
                    help=' number of epochs between decreasing steps applies to lr-scheduler in [step, exp], negative number indicates lr-step in iters')
parser.add_argument('--patience', type=int, default=10000,help='Number of epochs without improving in val set and converge')


parser.add_argument('--latent-size-upper', type=int, default=-1,
                    help="dimension of the latent size for upperbound")
parser.add_argument('--is-alpha-loss', action='store_true', default=False,
                    help='apply alpha coef to second loss term')
parser.add_argument("--inverted-alpha", default=False,type=bool,
                    help="alpha becomes 1 - alpha")
parser.add_argument('--temperature', type=float, default=1.0,help='Temperature for the KL Div loss in dist-vanilla')
parser.add_argument('--dist-center', type=float, default=0.2,help='Distance to penalize centers on suppl. maps')

# dropout settings
parser.add_argument('--lognormal', action='store_true', default=False,
                    help='use lognormal distribution for noise. (default: False)')
parser.add_argument('--lambda-noise', default=0.0, type=float,
                    help='lambda to regulaze dropout noise')
parser.add_argument('--add-fc', action='store_true', default=False,
                    help='add a second fc layer (like original setup from lupi-dropout paper)')

parser.add_argument("--teacher-weights", default=None,
                    help="Path to trained teacher model")
parser.add_argument('--ft', action='store_true', default=False,
                    help='finetuning on a different dataset')

# attn params
parser.add_argument('--freeze-optimizer-attention', default=0, type=int,
                    help='epochs mine optimizer is frozen')
parser.add_argument('--is-fullattention', action='store_true', default=False)
parser.add_argument('--atn-non-linearity', default="sigmoid",type=str)
parser.add_argument('--attention-kernel-size', type=str, default="3x3",help='size of the attention kernel')
parser.add_argument('--supervision-type', default='direct',choices=['direct','positive','no','ce','keyloc', 'l2'])
parser.add_argument('--nr-attention-maps', default=32, type=int)
parser.add_argument('--bilinear', action='store_true', default=False, help='bilinear features')
parser.add_argument('--dim-lowrank', type=int, default=8192,help='dim of low rank approximiation for bilinear features')
parser.add_argument('--attention-kernel-size-attributes', type=str, default="3x3",help='size of the attention kernel for attributes maps')


parser.add_argument('--WS_DAN-regularization', type=str, default="paper",choices=['paper','code'],help='feature centering code version or paper version')


class Trainer(object):

    def __init__(self, args):

        # if no config is given set default
        args.tag = get_default_model_version(args)

        self.args = args
        self.saver = Saver(self.args)
        self.model_name = self.saver.model_name
        self.writer = SummaryWriter(log_dir=self.saver.experiment_dir)
        self.writer_val = SummaryWriter(log_dir=os.path.join(self.saver.experiment_dir,'val'))

        # Load Data
        train, val, test, self.n_class, others = make_data_loader(args)
        self.train_dataloader = train
        self.val_dataloader = val
        self.test_dataloader = test
        if 'attributes' in args.x_star:
            self.size_s, self.size_t, self.size_attributes = others
        else:
            self.size_s, self.size_t = others
            self.size_attributes = None

        self.model = get_model_cnn(args.model, n_class=self.n_class,
                               size_s=self.size_s,
                               size_t=self.size_t,
                               size_attributes=self.size_attributes,args=self.args)

        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        if self.args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
        self.model.to(self.device)

        if self.model_name == 'dist-vanilla':
            assert self.args.teacher_weights is not None
            self.phi_model = get_model_cnn('upper', n_class=self.n_class,
                                         size_s=self.size_s,
                                         size_t=self.size_t,
                                        size_attributes=self.size_attributes, args=self.args)

            self.phi_model = torch.nn.DataParallel(self.phi_model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.phi_model)
            self.phi_model.to(self.device)
            self.resume(path=self.args.teacher_weights, model=self.phi_model, optimizer=None)
        else:
            self.phi_model = get_phi_model(self.model_name, n_class=self.n_class, args=args)
            if self.phi_model is not None:
                self.phi_model = torch.nn.DataParallel(self.phi_model, device_ids=self.args.gpu_ids)
                patch_replication_callback(self.phi_model)
                self.phi_model.to(self.device)

        self.lr = self.args.lr
        if self.args.lr_backbone_multiplier == 0:
            self.model.module.backbone.requires_grad = False
            self.params_opt = [{'params': self.model.module.get_lr_params(is_backbone=False), 'lr': self.lr}]
        else:
            self.params_opt = [{'params': self.model.module.get_lr_params(keyword='backbone', inclusive=True), 'lr': self.lr*self.args.lr_backbone_multiplier},
                               {'params': self.model.module.get_lr_params(keyword='backbone', inclusive=False), 'lr': self.lr}]


        self.phi_optimizer = None
        if self.phi_model:
            self.params_opt += [{'params': self.phi_model.parameters(), 'lr': self.lr}]

        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.params_opt, momentum=self.args.momentum,
                                             weight_decay=self.args.w_decay)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.params_opt, weight_decay=1e-4)
        else:
            raise NotImplementedError

        self.scheduler = LR_Scheduler(self.args, len(self.train_dataloader))
        self.phi_scheduler = LR_Scheduler(self.args,len(self.train_dataloader))

        self.nll_loss = torch.nn.NLLLoss()
        self.CE_loss = torch.nn.CrossEntropyLoss()

        self.best_pred = 0
        self.start_epoch = 0
        if self.args.dataset == 'boat_test':
            self.evaluator = EvaluatorImgClass(self.n_class+1)
            self.evaluator_test = EvaluatorImgClass(self.n_class+1)
        elif self.args.dataset == 'inat2018':
            self.evaluator = EvaluatorImgClass(self.n_class, class_groups=self.train_dataloader.dataset.class_groups)
            self.evaluator_test = EvaluatorImgClass(self.n_class, class_groups=self.train_dataloader.dataset.class_groups)
        else:
            self.evaluator = EvaluatorImgClass(self.n_class)
            self.evaluator_test = EvaluatorImgClass(self.n_class)

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        print('N param in model ', sum([np.prod(p.size()) for p in model_parameters]))

        # Resuming checkpoint
        if args.resume is not None:
            if args.resume == 'best':
                args.resume = os.path.dirname(self.saver.experiment_dir) + '/model_best.pth.tar'
            self.best_pred, self.start_epoch = self.resume(path=args.resume,model=self.model,optimizer=self.optimizer)


        self.temp_min = 0.5
        self.ANNEAL_RATE = 0.00003


    def train_and_eval(self):
        is_tbar = True
        if is_tbar:
            tbar = trange(self.start_epoch,self.args.epochs, desc='\r')
            set_descr = tbar.set_description
        else:
            tbar = range(self.args.epochs)
            set_descr = print

        bad_epochs = 0
        self.iter = 0
        self.current_discrepancy = 0
        self.temp = 1.0
        self.max_iters_train = len(self.train_dataloader)*self.args.epochs
        self.eval_every = 2

        for e in tbar:
            self.p = 0 if e <= 20 else 1 # model S3N

            train_loss = self.training()
            self.writer.add_scalar('total_loss_epoch', train_loss, e)

            if np.mod(e,self.eval_every) == 0:
                self.p = 1 if e <= 20 else 2  # for model S3N
                test_loss = self.validation(epoch=e)
                self.writer_val.add_scalar('total_loss_epoch', test_loss, e)

                metrics = self.evaluator.get_metrics()
                # print(metrics)
                for key, val in metrics.items():
                    self.writer_val.add_scalar('metrics/' + key, val, e)
                new_pred = metrics['Acc']

                # log lr
                lr_ = np.log10(self.scheduler.lr_current)
                self.writer_val.add_scalar('log_lr',lr_, e)

                if self.args.optimizer == 'sgd':
                    set_descr(
                        f'Epoch {e}  (prev: {self.best_pred*100:.2f}) {new_pred*100:.2f} '
                        f'lr {lr_:.2f} mi {self.current_discrepancy:.2f}')
                else:
                    set_descr(
                        f'Epoch {e}  (prev: {self.best_pred*100:.2f}) {new_pred*100:.2f} mi {self.current_discrepancy:.2f}')
                if self.best_pred < new_pred:
                    self.best_pred = new_pred
                    bad_epochs = 0
                    is_best = True
                    if self.args.testing_model == 'best':
                        self.test_model_state_dict = self.model.state_dict()
                        self.saver.save_checkpoint({
                            'epoch': e + 1,
                            'state_dict': self.model.module.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'best_pred': self.best_pred,
                            'model_name': self.saver.model_name
                        }, is_best)
                else:
                    bad_epochs += self.eval_every
                
            if self.args.testing_model == 'last':
                self.test_model_state_dict = self.model.state_dict()
            
            if bad_epochs > args.patience:
                break
        if self.args.testing_model == 'last':
            self.test_model_state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': e + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'model_name': self.saver.model_name
            }, False)
        return e

    def training(self):
        tbar_train = tqdm(self.train_dataloader)
        train_loss = 0
        self.model.train()

        accum_grad_iters = self.args.accum_grad_iters

        for sample in tbar_train:
            sample = self.to_device(sample)
            target = sample['label']
            sample.pop('label')
            alpha = self.get_alpha()

            if self.args.optimizer == 'sgd':
                self.scheduler(self.optimizer,self.iter,self.best_pred)
            if self.iter % accum_grad_iters == 0:
                self.optimizer.zero_grad()
            if self.phi_optimizer:
                self.phi_optimizer.zero_grad()
            
            ### TODO still needed?
            if self.model_name == 'attention_map':
                if self.iter % 500 == 1:
                    self.temp = np.maximum(self.temp * np.exp(-self.ANNEAL_RATE * self.iter), self.temp_min)
                sample['temperature'] = self.temp
            if self.model_name == 'S3N':
                sample['p'] = self.p

            assert self.model_name in ['dist-vanilla', 'L2_s', 'L2_st', 'DANN', 'dropout', 'student', 'attention_map', 'rnnattention_map',
                                       'multi_task_mask','similarity_grad','similarity_grad_v2','WS_DAN',
                                        'S3N', 'transformer','attention_map_adaptgrad','iSQRT', 'iSQRT_attention','upper',
                                        'attention_action'] #'spatial_attention'

            if self.model_name == 'similarity_grad':
                output = self.model(sample)
                loss,loss_reg = self.model.module.get_loss(sample,output,target,self.phi_model,alpha)
                loss_main = loss.detach()
                loss += loss_reg
            elif self.model_name == 'iSQRT':
                output = self.model(sample)
                loss = self.CE_loss(output['s'], target)
                loss_main = loss.detach()
                loss_reg = torch.zeros((1,)).float().to(self.device)
                loss.backward()
            elif self.model_name == 'WS_DAN':
                output = self.model(sample)
                loss = self.nll_loss(output['s'], target) / 3.
                loss_main = 3*loss.detach()
                loss_reg = self.model.module.get_loss(sample,output,target,self.phi_model,self.scheduler.lr_current)
                loss = loss_main + loss_reg
                loss.backward()

                ##################################
                # Attention Cropping
                ##################################
                with torch.no_grad():
                    crop_images = self.model.module.batch_augment(sample['s'], output['crop_drop'][:, :1, :, :], mode='crop', theta=(0.4, 0.6),padding_ratio=0.1)
                sample_crop = {'s': crop_images}
                # crop images forward
                y_pred_crop = self.model(sample_crop)
                loss = self.nll_loss(y_pred_crop['s'], target) / 3.
                loss.backward()

                ##################################
                # Attention Dropping
                ##################################
                with torch.no_grad():
                    drop_images = self.model.module.batch_augment(sample['s'], output['crop_drop'][:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
                sample_drop = {'s': drop_images}
                # drop images forward
                y_pred_drop = self.model(sample_drop)
                loss = self.nll_loss(y_pred_drop['s'], target) / 3.
                loss.backward()
                # loss
            else:
                output = self.model(sample)
                if self.model_name == 'S3N':
                    loss = 0
                else:
                    loss = self.nll_loss(output['s'], target)
                    loss_main = loss.detach()

                loss_reg = self.model.module.get_loss(sample,output,target,self.phi_model,self.get_alpha())
                if self.model_name == 'S3N':
                    loss_main = loss_reg.detach()

                if self.model_name == "attention_map_adaptgrad":
                    self.model.module.run_grad_merging(loss,loss_reg,self.writer,self.iter,sample)
                else:
                    loss += loss_reg
                    loss /= accum_grad_iters
                    loss.backward()

            self.iter += 1

            if self.iter % accum_grad_iters == 0:
                self.optimizer.step()

            train_loss += loss_main.item()

            if self.iter % 100 == 1:
                self.writer.add_scalar('total_loss_iter/total', loss.item(), self.iter)
                self.writer.add_scalar('total_loss_iter/main', loss_main.item(), self.iter)
                if torch.is_tensor(loss_reg): self.writer.add_scalar('total_loss_iter/reg', loss_reg.item(), self.iter)
                self.writer.add_scalar('total_loss_iter/alpha', alpha, self.iter)
                tbar_train.set_description(f'Train main loss: {train_loss/(self.iter):.2f} alpha: {alpha:.2f}')
        return train_loss
    
    def to_device(self, sample,device=None):
        if device is None:
            device = self.device
        sampleout = {}
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                sampleout[key] = val.to(device)
            elif isinstance(val, list):
                new_val = []
                for e in val:
                    assert( isinstance(e, torch.Tensor))
                    new_val.append(e.to(device))
                sampleout[key] = new_val
            else:
                sampleout[key] = val
        return sampleout

    def validation(self,epoch=None):
        
        self.evaluator.reset()
        self.model.eval()
        tbar = tqdm(self.val_dataloader, desc='Validation')
        test_loss = 0.0
        first = True
        is_plot_attention = 'attention' in self.model_name and not 'action' in self.model_name
        for sample in tbar:
            
            sample = self.to_device(sample)
            ## TODO still needed?
            if self.model_name == 'attention_map': sample['temperature'] = self.temp
            if self.model_name == 'S3N':           sample['p'] = self.p
            with torch.no_grad():
                output = self.model(sample)
            loss = self.nll_loss(output['s'], sample['label'])
            test_loss += loss.item()
            ps = torch.exp(output['s'])

            probab = ps.cpu().numpy()
            true_label = sample['label'].cpu().numpy()
            self.evaluator.add_batch(np.array(true_label), np.array(probab))
            
            if first:
                output = self.to_device(output, 'cpu')
                sample = self.to_device(sample, 'cpu')

                if is_plot_attention:
                    plots.plot_attention(self.writer_val, sample, output, global_step=epoch)
                if 'x_trans' in output.keys():
                    plots.plot_images_theta_transformer(self.writer_val,
                                                        sample['s'],
                                                        output['theta'],
                                                        theta_gt=output['theta_gt'] if 'theta_gt' in output.keys() else None,
                                                        global_step=epoch,
                                                        name='x_trans1')
                if 'x_trans_attn' in output.keys():
                    x_trans = output['x_trans_attn']
                    x_trans = torch.cat((x_trans, sample['s']), dim=2)
                    plots.plot_images(self.writer_val, x_trans, global_step=epoch, name='x_trans_attn')

                first = False
        plots.add_confusion_image(self.writer_val,self.evaluator.confusion_matrix,epoch)

        return test_loss

    def test(self):
        self.model.eval()
        if not self.args.test_only:
            # model was already loaded before
            self.model.load_state_dict(self.test_model_state_dict)
                    
        self.evaluator_test.reset()
        i = 0
        n_samples = 200
        id_batch = np.random.choice(len(self.test_dataloader), n_samples//self.args.batch_size, replace=False)
        is_save_hdf5 = 'savehdf5' in self.args.tag
        is_tensorboard_plot_test = False
        is_save_input_hdf5 = False
        is_plot_attention = 'attention' in self.model_name and not 'action' in self.model_name
        if is_save_hdf5:
            hf = h5py.File(self.writer_val.log_dir+'/output.h5','w')
            hfout = hf.create_group('output')
            if is_save_input_hdf5:
                hfin = hf.create_group('sample')
            id_counter = 0
        for id_,sample in enumerate(tqdm(self.test_dataloader)):
            sample = self.to_device(sample)
            if self.model_name == 'S3N': sample['p'] = 2
            with torch.no_grad():
                output = self.model(sample)

            ps = torch.exp(output['s'])
            probab = ps.cpu().numpy()
            true_label = sample['label'].cpu().numpy()
            self.evaluator_test.add_batch(np.array(true_label), np.array(probab))

            if is_tensorboard_plot_test and is_plot_attention and id_ in id_batch:
                output = self.to_device(output,'cpu')
                sample = self.to_device(sample,'cpu')
                if 'x_trans' in output.keys():
                    plots.plot_images_theta_transformer(self.writer_val,
                                                        sample['s'],
                                                        theta=output['theta'] if 'theta' in output.keys() else None,
                                                        theta_gt=output['theta_gt'] if 'theta_gt' in output.keys() else None,
                                                        bbox=output['bboxes'] if 'bboxes' in output.keys() else None,
                                                        global_step=i,
                                                        name='x_trans1')

                plots.plot_attention(self.writer_val,sample,output,global_step=i)
                i+=1

            if is_save_hdf5 and id_ in id_batch:
                output = self.to_device(output,'cpu')
                sample = self.to_device(sample,'cpu')
                B = output['s'].shape[0]
                for i in range(B):
                    for key, val in output.items():
                        if val is not None:
                            hfout.create_dataset(name=f'{key}/{id_counter}', data=val[i])
                    if is_save_input_hdf5:
                        for key, val in sample.items():
                            if val is not None:
                                hfin.create_dataset(name=f'{key}/{id_counter}', data=val[i])
                    id_counter += 1
        if is_save_hdf5:
            hf.close()

    def get_alpha(self):
        return losses.calc_coeff(self.iter, max_iter=self.max_iters_train)

    def resume(self,path,model,optimizer):
        if not os.path.isfile(path):
            raise RuntimeError("=> no checkpoint found at '{}'".format(path))
        checkpoint = torch.load(path)
        start_epoch = checkpoint['epoch']

        # if model is the same, if not only matching keys will be loaded.
        if checkpoint['model_name'] != self.model_name:
            self._load_pretrained_model(checkpoint['state_dict'],model)
        else:
            if self.args.cuda:
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
        if not self.args.ft and (optimizer is not None) and not args.test_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
        best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if self.args.ft:
            start_epoch = 0

        return best_pred,start_epoch

    def _load_pretrained_model(self, pretrain_dict, model):

        model_dict = {}
        if self.args.cuda:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        for k, v in pretrain_dict.items():
            if self.args.cuda:
                k = 'module.'+k
            if k in state_dict:
                model_dict[k] = v

        pretrained_modules = set([x.split('.')[0] for x in pretrain_dict])
        if self.args.cuda:
            modules = set([x.replace('module.','').split('.')[0] for x in state_dict])
        else:
            modules = set([x.split('.')[0] for x in state_dict])

        print('not loaded modules:', modules.difference(pretrained_modules))
        state_dict.update(model_dict)

        if self.args.cuda:
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

def main(args):

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    trainer = Trainer(args)

    if not args.test_only:
        e = trainer.train_and_eval()
        print(f'Converged Epoch {e} {trainer.evaluator.get_metrics()}')
        trainer.writer.close()
        trainer.writer_val.close()
    trainer.test()

    metrics = trainer.evaluator_test.get_metrics()
    if trainer.args.dataset == 'boat_test':
        m = trainer.evaluator_test.get_boat_metrics()
        metrics['Acc_boat'] = m[0]
        metrics['Acc_water'] = m[1]
        metrics['Acc_averall'] = m[2]
    elif trainer.args.dataset == 'boat_test_binary':
        m = trainer.evaluator_test.get_boat_only_metrics()
        metrics['Acc_boat'] = m[0]
        metrics['Acc_water'] = m[1]
        metrics['Acc_averall'] = m[2]
    elif 'cct20' in trainer.args.dataset:
        cat_list = trainer.train_dataloader.dataset.cat_list
        acc = trainer.evaluator_test.Acc_Per_Class()
        m = {'Acc_'+c: a for (c, a) in zip(cat_list,acc)}
        metrics = {**metrics,**m}
    
    if 'inat2018' in trainer.args.dataset:
        save_file = os.path.join(trainer.saver.experiment_dir,'Evaluator.pkl')
        trainer.evaluator_test.save_object(save_file)

    metrics['Model'] = trainer.model_name + trainer.saver.model_version
    print(metrics)

    results = pd.DataFrame(columns=metrics.keys())
    results = results.append(metrics,ignore_index=True)
    results['experiment'] = os.path.basename(trainer.saver.experiment_dir)

    # Adding hyperparameters to results.csv
    flags = trainer.args.__dict__
    flags_dset = pd.DataFrame(columns=flags.keys())
    flags_dset = flags_dset.append(flags,ignore_index=True)
    results = pd.concat((flags_dset,results),axis=1)

    save_file = os.path.join(args.save_dir, trainer.saver.data_name, 'results.csv')
    if os.path.isfile(save_file):
        results_old = pd.read_csv(save_file)
        results = pd.concat([results_old,results])
    results.to_csv(save_file,index=False)

    # def keyboardInterruptHandler(signal, frame):
    #     print(f"Timelimit has reached... (ID: {signal}) stopping training and evaluating on the test data")
    #     # last_part()

    # signal.signal(signal.SIGUSR2, keyboardInterruptHandler)

    print(save_file,'saved!')


if __name__ == '__main__':

    args = parser.parse_args()
    main(args)

