import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.backbone import build_backbone
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from models import model_base


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP', non_lin='sigmoid', convmaxpool=None):
        super(BAP, self).__init__()
        self.non_lin = non_lin
        # self.grad_damp = grad_damp(alpha=0.1)
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            if convmaxpool is not None:
                self.pool = nn.Sequential(convmaxpool, nn.AdaptiveMaxPool2d(1))
            else:
                self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, raw_attentions, temperature=None):
        B = features.size(0)
        M = raw_attentions.size(1)

        if self.non_lin == 'sigmoid':
            attentions = torch.sigmoid(raw_attentions)
        elif self.non_lin == 'none':
            attentions = raw_attentions

        feature_matrix = []
        for i in range(M):
            AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, 1, -1)

            # if self.non_lin == 'sqrt':
                # AiF = torch.sign(AiF) * torch.sqrt(torch.abs(AiF) + 1e-12)
            # AiF = F.normalize(AiF, dim=2, p=2)

            feature_matrix.append(AiF)

        feature_matrix = torch.cat(feature_matrix, dim=1)

        if self.non_lin == 'sqrt':
            attentions = torch.sigmoid(attentions)
        if self.non_lin == 'gumbel':
            return feature_matrix, (attentions,attentions_soft)
        return feature_matrix, attentions

class CNN1(model_base.CNN):

    def __init__(self, output_size, size_s, size_t, size_attributes=None, args=None):

        assert args.atn_non_linearity in ['sigmoid', 'none']
        self.non_lin = args.atn_non_linearity
        self.in_s = size_s
        self.in_t = size_t

        self.supervision_type = args.supervision_type

        self.n_attn_maps_supervised = size_t
        self.n_attn_maps = args.nr_attention_maps
        self.attention_kernel_size = args.attention_kernel_size


        # init basemodel with student input only
        super().__init__(output_size=output_size, input_size=size_s, args=args)

        self.L = 768
        if '2_9' in self.args.tag:
            self.L = 2**9
        elif '2_10' in self.args.tag:
            self.L = 2**10
        elif '2_12' in self.args.tag:
            self.L = 2**12
        elif '2_13' in self.args.tag:
            self.L = 2**13
        elif '2_14' in self.args.tag:
            self.L = 2**14
        # Attention Maps
        if self.attention_kernel_size == "1x1":
            self.prelogits_attentions = nn.Sequential(
                nn.Conv2d(self.nfeat_backbone, self.L, kernel_size=1, bias=True),
                nn.ReLU(True))
            self.attentions = nn.Conv2d(self.L, self.n_attn_maps, kernel_size=1, bias=True)
            self.attentions_a = nn.Conv2d(self.L, 1, kernel_size=1, bias=True)

        else:
            raise NotImplementedError

        # self.pool = BAP(pool='GAP', non_lin=self.non_lin)
        # max pool conv layer
        # if 'maxpool' in self.args.tag:
            # cnnmaxpool = nn.Conv2d(self.nfeat_backbone, self.nfeat_backbone, kernel_size=1, bias=False)
            # self.bmp = BAP(pool='GMP', non_lin=self.non_lin, convmaxpool=cnnmaxpool)


        self.is_rnn = 'rnn' in self.args.tag

        nfeat_ = self.nfeat_backbone
        if 'attncropping' in self.args.tag:
            nfeat_ = nfeat_ * 2
        if 'maxpool' in self.args.tag:
            self.model_out = nn.Sequential(nn.Linear(2*nfeat_ * self.n_attn_maps, output_size),
                                            nn.LogSoftmax(dim=1))
            self.model_out_maxpool = nn.Sequential(nn.Linear(nfeat_ * self.n_attn_maps, output_size),
                                            nn.LogSoftmax(dim=1))
            self.model_out_avgpool = nn.Sequential(nn.Linear(nfeat_ * self.n_attn_maps, output_size),
                                            nn.LogSoftmax(dim=1))

            # self.feature_center = torch.zeros(output_size, self.n_attn_maps, 2*self.nfeat_backbone).to(
            #     torch.device("cuda"))  # net.expansion)#
        assert not 'iSQRT' in self.args.tag, 'not implemented'

        self.model_out = nn.Sequential(
                                nn.Linear(self.nfeat_backbone + self.n_attn_maps, output_size),
                                nn.LogSoftmax(dim=1))

        # self.feature_center = torch.zeros(output_size, self.n_attn_maps, self.nfeat_backbone).to(
        #     torch.device("cuda"))  # net.expansion)#

        if "resnet"  in self.args.backbone:
            x = np.linspace(-1,1,28)
        elif "inception" in self.args.backbone:
            x = np.linspace(-1,1,26)
        else:
            raise NotImplementedError

        # x,y = np.meshgrid(x,x)
        # self.x_grid = torch.from_numpy(x).float().to(torch.device("cuda")).reshape((1,1,x.shape[0],x.shape[0]))
        # self.y_grid = torch.from_numpy(y).float().to(torch.device("cuda")).reshape((1,1,x.shape[0],x.shape[0]))
        ref_ = 448 if self.args.is_crop_train else 800
        x = np.linspace(0.1, 1, ref_)
        self.x_grid = torch.from_numpy(x).float().to(torch.device("cuda"))
        self.dropout = nn.Dropout2d(p=0.5)
    def my_CE_loss(self,pred, target):
        return -torch.sum(target * pred)


    def forward(self, input, is_crop_atn=True):
        return_dict = dict()

        # if 'temperature' in input.keys():
        #     tao = input['temperature']
        # else:
        #     tao = 0.5

        input_list = [input[key] for key in self.input_keys]
        x = torch.cat(input_list, dim=1)
        x1 = x

        feature_maps = self.backbone(x1)
        if isinstance(feature_maps,tuple):
            feature_maps = feature_maps[0]
        if 'regularizedcenter' in self.args.tag:
            return_dict['feature_maps'] = feature_maps

        raw_attention_maps, attention_maps, feature_matrix, dict_ = self.forward_atention(feature_maps)

        return_dict = {**return_dict,**dict_}

        dict_ = self.forward_attention_teacher(input=input, x=x, raw_attention_maps=raw_attention_maps)
        return_dict = {**return_dict,**dict_}

        # if 'attncropping' in self.args.tag:
        #     threshold = 0.5
        #
        #     atn_maps = torch.sigmoid(raw_attention_maps)
        #     # normalize
        #     atn_maps = utils_nn.normminmax(atn_maps)
        #     atn_maps = atn_maps.mean(dim=1, keepdim=True)
        #     atn_maps = (atn_maps > threshold).float()
        #     atn_maps = nn.UpsamplingBilinear2d(size=x.size()[-2:])(atn_maps)
        #     theta_attn = utils_nn.get_gt_theta(ref_width=atn_maps.size()[-1], keypoint_map=atn_maps.squeeze(1),
        #                                     x_grid=self.x_grid, margin=20)
        #
        #     grid = F.affine_grid(theta_attn[:,:2,:], x.size())
        #     x1 = F.grid_sample(x, grid)
        #     return_dict['x_trans_attn'] = x1
        #
        #     feature_maps1, _ = self.backbone(x1)
        #
        #     raw_attention_maps1, attention_maps1, feature_matrix1, dict1_ = self.forward_atention(feature_maps1)
        #     return_dict['feature_m_max'] = torch.stack((return_dict['feature_m_max'], dict1_['feature_m_max']),dim=-1)
        #     return_dict['feature_m_avg'] = torch.stack((return_dict['feature_m_avg'], dict1_['feature_m_avg']),dim=-1)
        #
        #     feature_matrix = torch.stack((feature_matrix1, feature_matrix), dim=-1)
        # if 'weaktrans' in self.args.tag:
        #     B = x.shape[0]
        #     threshold = 0.5
        #     is_max = False
        #     with torch.no_grad():
        #         ref_size = input['s'].size()[-2:]
        #         atn_maps = return_dict['zs']  # .sum(dim=1, keepdim=True) > 1e-5
        #         # normalize
        #         atn_maps = utils_nn.normminmax(atn_maps)
        #         if is_max:
        #             atn_maps, _ = atn_maps.max(dim=1, keepdim=True)
        #         else:
        #             atn_maps = atn_maps.mean(dim=1, keepdim=True)
        #         atn_maps = (atn_maps > threshold).float()
        #         atn_maps = nn.UpsamplingBilinear2d(size=ref_size)(atn_maps)
        #         theta_attn = utils_nn.get_gt_theta(ref_width=ref_size[-1], keypoint_map=atn_maps.squeeze(1),
        #                                         x_grid=self.x_grid, margin=20)
        #         a = torch.tensor([0., 0., 1.], device=theta_attn.device).view(1, 1, 3)
        #         a = torch.cat(B * [a], dim=0)
        #         thetas = torch.cat((return_dict['theta'], a), dim=1)
        #
        #         theta_gt = torch.bmm(thetas, theta_attn)
        #         return_dict['theta_gt_weak'] = theta_gt


        if self.is_rnn:
            imout, rnn_out = self.model_out(feature_matrix)
            return_dict['rnn_out'] = rnn_out

        else:
            if 'dropout' in self.args.tag:
                feature_matrix = self.dropout(feature_matrix)
            imout = self.model_out(feature_matrix.view(feature_matrix.shape[0], -1))

        if not self.training and 'testatncropping' in self.args.tag and is_crop_atn:
            mean_attention_map = return_dict['zs'][:,:self.n_attn_maps_supervised].mean(dim=1,keepdim=True)
            img_cropped, bboxes = self.batch_augment(images=x1, attention_map=mean_attention_map,mode='crop')
            return_dict1 = self.forward({'s':img_cropped}, is_crop_atn=False)
            imout = (return_dict1['s'] + imout) / 2
            return_dict['x_trans'] = img_cropped
            return_dict['bboxes'] = bboxes

        return_dict['s'] = imout
        return_dict['feature_m'] = feature_matrix

        return return_dict

    def forward_atention(self, feature_maps):
        dict_ = {}

        raw_attention_maps = self.prelogits_attentions(feature_maps)

        attentions_a = self.attentions_a(raw_attention_maps)
        attentions_a = torch.softmax(attentions_a,dim=1)

        attentions_k = self.attentions(raw_attention_maps)
        attention_maps = torch.cat((attentions_k,feature_maps),dim=1)
        attention_maps = self.dropout(attention_maps)

        feature_matrix = self.avgpool(attention_maps * attentions_a)

        # if self.n_attn_maps - self.n_attn_maps_supervised > 0:
        #     raw_attention_maps_attributes = self.attentions_attributes(feature_maps)
        #     raw_attention_maps = torch.cat((raw_attention_maps, raw_attention_maps_attributes), dim=1)

        if self.non_lin == 'sqrt':
            raw_attention_maps = torch.relu(raw_attention_maps)

        # feature_matrix, attention_maps = self.pool(feature_maps, raw_attention_maps)
        if 'maxpool' in self.args.tag:
            feature_matrix_avgpool = feature_matrix
            feature_matrix_maxpool, _ = self.bmp(feature_maps, raw_attention_maps)
            dict_['feature_m_max'] = feature_matrix_maxpool
            dict_['feature_m_avg'] = feature_matrix_avgpool
            feature_matrix = torch.cat((feature_matrix_avgpool, feature_matrix_maxpool), dim=1)

        # if self.supervision_type == "keyloc":
            # dict_['zs_loc'] = self.get_centers(attention_maps)

        # if self.non_lin == 'gumbel':
            # attention_maps, attention_maps_soft = attention_maps

        dict_['zs'] = raw_attention_maps

        return raw_attention_maps, attention_maps, feature_matrix, dict_

    def forward_attention_teacher(self,input, x, raw_attention_maps):

        dict_ = {}
        ### spatial attention
        ##TODO this should be modified if ultimately we always use input supervision
        if 't' in input.keys() and self.supervision_type != 'no':
            if self.supervision_type in ['direct', 'positive', 'ce', 'l2']:
                attention_maps_t = input['t']
                # TODO check if the transformer should be applied before the max pooling
                attention_maps_t = F.adaptive_max_pool2d(attention_maps_t, output_size=(raw_attention_maps.shape[2], raw_attention_maps.shape[3]))
        else:
            attention_maps_t = None

        dict_['zt'] = attention_maps_t

        return dict_

    def get_loss(self, input, output, label, phi_model, alpha):

        loss = 0
        B = output['feature_m'].shape[0]
        if 'maxpoolreg' in self.args.tag:
            y_ = self.model_out_avgpool(output['feature_m_avg'].view(B,-1))
            loss += self.nll_loss(y_, label)

            y_ = self.model_out_maxpool(output['feature_m_max'].view(B,-1))
            loss += self.nll_loss(y_, label)

        if self.supervision_type != 'no':
            shared_maps = min(output['zt'].shape[1], output['zs'].shape[1])
        else:
            shared_maps = 0

        if 'resnet' in self.args.backbone:
            poolings = [(1, 1), (4, 4), (7, 7), (14, 14)]  # (28,28)
        elif self.args.backbone == "inception":
            poolings = [(1, 1), (3, 3), (6, 6), (13, 13)]  # (26,26)
        else:
            raise NotImplementedError

        if self.args.supervision_type == 'direct':

            zs_shared = output['zs'][:, 0:shared_maps]
            zt_shared = output['zt'][:, 0:shared_maps].detach()

            reg_ = 'only_labeled_samples'
            if reg_ == 'only_labeled_samples':
                w = zt_shared.sum(dim=(1, 2, 3)) > 0
                if 'cct' in self.args.dataset: # in cct empty class with no animal counts as labeled
                    w = w + (label == 2)
                zs_shared = zs_shared[w]
                zt_shared = zt_shared[w]
            elif reg_ == 'only_labeled_atn_maps':

                w = zt_shared.sum(dim=(2, 3)) > 0
                zs_shared = zs_shared[w]
                zt_shared = zt_shared[w]
            else:
                w = torch.tensor(1)

            zt_shared = zt_shared.detach()
            if w.sum() > 0:
                for pooling in poolings:
                    loss += self.bce_loss(torch.nn.functional.max_pool2d(zs_shared, kernel_size=pooling),
                                          torch.nn.functional.max_pool2d(zt_shared, kernel_size=pooling))

            if 'regularizedbbox_shared' in self.args.tag:
                assert 'bbox' in self.args.x_star
                dim_bbox = self.n_attn_maps_supervised - 1
                bbox = output['zt'][:, dim_bbox]

                negbbox_pixels = bbox == 0
                z_ = output['zt'][:, :shared_maps].transpose(1, -1)[negbbox_pixels]
                loss += self.bce_loss(z_, torch.zeros_like(z_))
            elif 'regularizedbbox_unshared' in self.args.tag:

                assert 'bbox' in self.args.x_star
                dim_bbox = self.n_attn_maps_supervised - 1
                bbox = output['zt'][:, dim_bbox]
                # if bbox.sum() > 0: #only regularize if there is a bbox in the image
                negbbox_pixels = bbox == 0
                z_ = output['zs'][:, shared_maps:].transpose(1, -1)[negbbox_pixels]
                loss += self.bce_loss(z_, torch.zeros_like(z_))
            elif 'regularizedbbox_all' in self.args.tag:
                assert 'bbox' in self.args.x_star
                dim_bbox = self.n_attn_maps_supervised - 1
                bbox = output['zt'][:, dim_bbox]

                negbbox_pixels = bbox == 0
                z_ = output['zs'].transpose(1, -1)[negbbox_pixels]
                loss += self.bce_loss(z_, torch.zeros_like(z_))
            elif 'regularizedvariance' in self.args.tag:
                n_unshared = self.n_attn_maps - shared_maps
                if n_unshared > 1:

                    # target_attention, _ = zs_shared.max(dim=1)
                    # target_attention = target_attention.detach()
                    z_ = output['zs'][:, shared_maps:]
                    z_mean = z_.mean(dim=(2, 3))
                    z_var = z_mean * (1-z_mean)
                    # max the variance of the attention maps
                    loss -= z_var.mean()
            elif 'regularizedsum' in self.args.tag:
                n_unshared = self.n_attn_maps - shared_maps
                if n_unshared > 1:

                    # target_attention, _ = zs_shared.max(dim=1)
                    # target_attention = target_attention.detach()
                    z_ = output['zs'][:, shared_maps:]
                    z_ = z_.view(B, -1)
                    z_sum = z_.sum(-1)
                    loss +=  z_sum.mean() - z_.shape[-1]
            elif 'regularizedkl' in self.args.tag:
                n_unshared = self.n_attn_maps - shared_maps
                if n_unshared > 1:

                    target_attention, _ = zs_shared.max(dim=1)
                    target_attention = target_attention.detach()
                    z_ = output['zt'][:, :shared_maps]
                    z_ = torch.log(z_ + 1e-12)
                    kl_loss = 0
                    for i in range(n_unshared):
                        kl_loss += torch.exp(-self.KL_loss(input=z_[:,i], target=target_attention))
                    loss += kl_loss*alpha
            elif 'regularizedcenter' in self.args.tag:
                n_unshared = self.n_attn_maps - shared_maps
                if n_unshared > 1:
                    centers_batch = self.feature_center[label][:, shared_maps:]

                    feature_map = output['feature_maps'].detach()
                    atn = output['zs'][:, shared_maps:]

                    wfeat = feature_map.unsqueeze(1) * atn.unsqueeze(2)

                    center_l2 = atn.unsqueeze(2) * (wfeat - centers_batch.unsqueeze(-1).unsqueeze(-1))**2
                    if 'centersum' in self.args.tag:
                        center_l2 = center_l2.sum((3,4))
                    else:
                        assert 'centermean' in self.args.tag, 'choose either centermean or centersum'

                    with torch.no_grad():
                        beta = 1e-4
                        diff = (1 - beta) * (centers_batch - wfeat.mean((3,4)))
                        self.feature_center[label][:, shared_maps:] -= diff

                    loss += center_l2.mean()

                    if 'exp1' in self.args.tag:
                        B, a, w, h = atn.shape
                        n_wo_diag = B * w * h * a * (a - 1)

                        diff_atn = atn.unsqueeze(1) - atn.unsqueeze(2)
                        exp1_l2 = diff_atn.pow(2).sum() / n_wo_diag * -1
                        loss += exp1_l2 *100
                    elif 'exp2' in self.args.tag:
                        wfeat_pooled = wfeat.mean((3, 4))

                        B, a, f = wfeat_pooled.shape
                        n_wo_diag = B * f * a * (a - 1)

                        diff_feat = wfeat_pooled.unsqueeze(1) - wfeat_pooled.unsqueeze(2)
                        exp2_l2 = diff_feat.pow(2).sum() / n_wo_diag * -1
                        loss += exp2_l2 *100

            elif 'regularized' in self.args.tag:
                # center reg
                feature_m = output['feature_m'][:, shared_maps:]
                n_maps = feature_m.shape[1]
                feature_m = feature_m.view(feature_m.shape[0], -1)

                centers_batch = self.feature_center[label][:, shared_maps:]
                centers_batch = centers_batch.view(centers_batch.shape[0], -1)
                centers_batch = nn.functional.normalize(centers_batch, dim=-1, p=2)
                beta = 1e-4
                diff = (1 - beta) * (centers_batch - feature_m.detach())
                diff = diff.view(diff.shape[0], n_maps, -1)
                self.feature_center[label][:, shared_maps:] -= diff

                loss += self.L2_loss(feature_m, centers_batch)

            if "t_attr_pred" in output:
                loss_att = 0
                for idx in range(0, len(input['t_attr'])):
                    sum_target = torch.sum(input['t_attr'][idx].float(), dim=1, keepdim=True)
                    p_t = input['t_attr'][idx].float()

                    valid_samples = sum_target.shape[0] - torch.sum(sum_target == 0)
                    sum_target[sum_target == 0] = 1
                    sum_target = sum_target.float()
                    p_t = p_t / sum_target

                    loss_att += (1. / valid_samples) * self.my_CE_loss(output['t_attr_pred'][idx], p_t)

                loss = loss + loss_att  * alpha
        elif self.args.supervision_type == 'l2':

            zs_shared = output['zs'][:, 0:shared_maps]
            zt_shared = output['zt'][:, 0:shared_maps].detach()

            reg_ = 'no'
            if reg_ == 'only_labeled_samples':
                w = zt_shared.sum(dim=(1, 2, 3)) > 0
                zs_shared = zs_shared[w]
                zt_shared = zt_shared[w]
            elif reg_ == 'only_labeled_atn_maps':

                w = zt_shared.sum(dim=(2, 3)) > 0
                zs_shared = zs_shared[w]
                zt_shared = zt_shared[w]
            else:
                w = torch.tensor(1)

            zt_shared = zt_shared.detach()
            if w.sum() > 0:
                for pooling in poolings:
                    loss += self.L2_loss(F.max_pool2d(zs_shared, kernel_size=pooling),
                                          F.max_pool2d(zt_shared, kernel_size=pooling))
        elif self.args.supervision_type == 'ce':
            zt = output['zt'].detach()

            for pooling in poolings:
                zt_ = torch.nn.functional.max_pool2d(zt, kernel_size=pooling)
                zs_ = torch.nn.functional.max_pool2d(output['zs'], kernel_size=pooling)

                probs_ = torch.log_softmax(zs_, dim=1)
                if probs_.shape[1] > shared_maps:
                    shared_probs = probs_[:, :shared_maps, ...]
                    unshared_probs = torch.log(torch.exp(probs_[:, shared_maps:, ...]).sum(dim=1)).unsqueeze(1)
                    probs_ = torch.cat((shared_probs, unshared_probs), dim=1)
                    zt_ = torch.cat((zt_, torch.zeros_like(unshared_probs) + 1e-12), dim=1)

                nll_loss = torch.nn.NLLLoss()
                _, atn_label = zt_.max(dim=1)

                loss += nll_loss(probs_, atn_label)

        elif self.args.supervision_type == 'conv':
            for pooling in poolings:
                loss = self.L2_loss(torch.nn.functional.max_pool2d(output['zs'][:, 0:shared_maps], kernel_size=pooling),
                                    torch.nn.functional.max_pool2d(output['zt'][:, 0:shared_maps], kernel_size=pooling))
        elif self.args.supervision_type == 'positive':

            pos_atn = (output['zt'][:, 0:shared_maps] > 0).float().detach()
            loss += self.L2_loss(output['zs'][:, 0:shared_maps] * pos_atn,
                                 output['zt'][:, 0:shared_maps] * pos_atn)
        elif self.args.supervision_type == 'keyloc':

            key_loc_pred = output['zs_loc'][:,:shared_maps,0:2]
            key_loc = output['zt_loc'][:,:,0:2]
            mask = output['zt_loc'][:,:,2:]

            loss += self.L2_loss(mask*key_loc_pred,mask*key_loc)


        elif self.args.supervision_type == 'no':
            # regularization
            regularize = False
            if 'regularizedvariance' in self.args.tag:
                z_ = output['zs']
                z_mean = z_.mean(dim=(2, 3))
                z_var = z_mean * (1 - z_mean)
                loss -= z_var.mean()
            elif regularize:
                # center reg
                feature_m = output['feature_m']
                n_maps = feature_m.shape[1]
                feature_m = feature_m.view(feature_m.shape[0], -1)

                centers_batch = self.feature_center[label]
                centers_batch = centers_batch.view(centers_batch.shape[0], -1)
                centers_batch = nn.functional.normalize(centers_batch, dim=-1, p=2)
                beta = 1e-4
                diff = (1 - beta) * (centers_batch - feature_m.detach())
                diff = diff.view(diff.shape[0], n_maps, -1)
                self.feature_center[label] -= diff

                loss += self.L2_loss(feature_m, centers_batch)

        if self.is_rnn:
            loss_seq = 0
            if self.model_out.type_ in ['mean', 'sum']:

                for i in range(self.n_attn_maps):
                    y_ = self.model_out.fc(output['rnn_out'][:,i])
                    loss_seq += self.nll_loss(y_, label)

            elif self.model_out.type_ == 'concatmean':

                loss_seq = 0
                for i in range(self.n_attn_maps):
                    y_ = self.model_out.fc1(output['rnn_out'][:, i])
                    loss_seq += self.nll_loss(y_, label)

            loss += loss_seq

        return loss

    def batch_augment(self,images, attention_map, mode='crop', theta=0.5):
        batches, _, imgH, imgW = images.size()
        padding_ratio = 0.2 if 'ratio02' in self.args.tag else 0.1

        if mode == 'crop':
            crop_images = []
            bboxes = []
            for batch_index in range(batches):
                atten_map = attention_map[batch_index:batch_index + 1]
                if isinstance(theta, tuple):
                    theta_c = np.random.uniform(*theta) * atten_map.max()
                else:
                    theta_c = theta * atten_map.max()

                crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
                nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
                if len(nonzero_indices) > 0:
                    height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
                    height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
                    width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
                    width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

                    crop_images.append(
                        F.upsample_bilinear(
                            images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                            size=(imgH, imgW)))
                    bboxes.append(torch.tensor([[height_min,height_max,width_min,width_max]], device=images.device))
                else:
                    crop_images.append(images[batch_index:batch_index + 1])
                    bboxes.append(torch.tensor([[0, imgH, 0, imgW]], device=images.device))
            crop_images = torch.cat(crop_images, dim=0)
            bboxes = torch.cat(bboxes,dim=0)
            return crop_images, bboxes

        elif mode == 'drop':
            drop_masks = []
            for batch_index in range(batches):
                atten_map = attention_map[batch_index:batch_index + 1]
                if isinstance(theta, tuple):
                    theta_d = np.random.uniform(*theta) * atten_map.max()
                else:
                    theta_d = theta * atten_map.max()

                drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
            drop_masks = torch.cat(drop_masks, dim=0)
            drop_images = images * drop_masks.float()
            return drop_images

        else:
            raise ValueError(
                'Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)
