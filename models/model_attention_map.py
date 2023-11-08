import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.backbone import build_backbone
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from models import model_base

from models.iSQRT.src.representation import MPNCOV

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if True:
        # args.cuda:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return torch.sigmoid(y / temperature)
    # return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """

    y = gumbel_softmax_sample(logits, temperature)
    y_soft = y # [..., 0].view(*shape)

    if not hard:
        return y_soft

    y_hard = (y > 0.5).float()

    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y

    return y_hard, y_soft

class COVPool(nn.Module):
    def __init__(self, input_dim, non_lin='sigmoid', args=None, n_supervised = None):
        super(COVPool, self).__init__()
        representation_args = {'iterNum': 5,
                          'is_sqrt': True,
                          'is_vec': True,
                          'input_dim': input_dim,
                          'dimension_reduction': None}

        self.non_lin = non_lin
        self.norm_attentions = False
        self.tag = args.tag
        self.n_supervised = n_supervised

        self.representation = MPNCOV(**representation_args)
        self.output_dim = self.representation.output_dim
        self.softmax2d = nn.Softmax2d()

    def forward(self, x, raw_attentions):
        if self.non_lin == 'sigmoid':
            attentions = torch.sigmoid(raw_attentions) + 1e-12
        elif self.non_lin == 'softmax':
            attentions = self.softmax2d(raw_attentions)
        elif self.non_lin == 'unsupervisedsoftmax':
            n_ = self.n_supervised
            atn_sup = torch.sigmoid(raw_attentions[:, :n_])
            # atn_unsup = F.log_softmax(raw_attentions[:,n_:],dim=1) # was like that 2020
            atn_unsup = self.softmax2d(raw_attentions[:,n_:],dim=1)
            attentions = torch.cat((atn_sup, atn_unsup),dim=1)
        else:
            raise NotImplemented

        if self.norm_attentions:

            D1, D2 = x.shape[-2], x.shape[-1]
            attention = 1 - torch.exp(torch.sum(torch.log(1 - attentions), dim=1, keepdim=True))
            sum_att = F.adaptive_avg_pool2d(attention,(1,1)) * (D1 * D2)
            attention_norm = attention / sum_att

            x = self.representation._cov_pool(attention_norm * x)
            x = x * (D1 * D2)
        else:

            if 'unsupervisednorm' in self.tag:
                n_ = 15
                x_sup = attentions[:,:n_].unsqueeze(1) * x.unsqueeze(2)

                # Merged unsupervised attentions
                D1, D2 = x.shape[-2], x.shape[-1]
                attention = 1 - torch.exp(torch.sum(torch.log(1 - attentions[:,n_:]+1e-12), dim=1, keepdim=True))
                sum_att = F.adaptive_avg_pool2d(attention, (1, 1)) * (D1 * D2) +1e-12
                attention_norm = attention / sum_att

                x_unsup = (attention_norm * x).unsqueeze(2)

                x = torch.cat((x_sup,x_unsup),dim=2)
                x = self.representation._cov_pool_atn(x)
            else:
                if 'weightedatn1' in self.tag:
                    x = self.representation._cov_pool_atn_weighted(x, attentions)
                elif 'weightedatn' in self.tag:
                    x = attentions.unsqueeze(1) * x.unsqueeze(2)

                    x = self.representation._cov_pool_atn_weighted(x, attentions)

                else:
                    x = attentions.unsqueeze(1) * x.unsqueeze(2)

                    x = self.representation._cov_pool_atn(x)

        x = self.representation._sqrtm(x)
        x = self.representation._triuvec(x)
        return x, attentions


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP', non_lin='sigmoid', convmaxpool=None, n_supervised=None):
        super(BAP, self).__init__()
        self.non_lin = non_lin
        self.n_supervised = n_supervised
        self.softmax2d = nn.Softmax2d()
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
        elif self.non_lin == 'softmax':
            attentions = self.softmax2d(raw_attentions)
        elif self.non_lin == 'unsupervisedsoftmax':
            n_ = self.n_supervised
            atn_sup = torch.sigmoid(raw_attentions[:, :n_])
            atn_unsup = self.softmax2d(raw_attentions[:,n_:])
            attentions = torch.cat((atn_sup, atn_unsup),dim=1)

        elif self.non_lin == 'gumbel':
            attentions, attentions_soft = gumbel_softmax(raw_attentions, temperature=temperature, hard=True)
            # slow transition to gumbel from sigmoid
            if temperature > 1.0:
                attentions = temperature * torch.sigmoid(raw_attentions) + (10.0 - temperature) * attentions_soft
            else:
                attentions = attentions_soft
        else:
            attentions = raw_attentions

        feature_matrix = []
        for i in range(M):
            AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, 1, -1)

            if self.non_lin == 'sqrt':
                AiF = torch.sign(AiF) * torch.sqrt(torch.abs(AiF) + 1e-12)

            AiF = F.normalize(AiF, dim=2, p=2)

            feature_matrix.append(AiF)

        feature_matrix = torch.cat(feature_matrix, dim=1)

        if self.non_lin == 'sqrt':
            attentions = torch.sigmoid(attentions)
        if self.non_lin == 'gumbel':
            return feature_matrix, (attentions,attentions_soft)
        return feature_matrix, attentions


class CNN1(model_base.CNN):

    def __init__(self, output_size, size_s, size_t, size_attributes=None, args=None):

        assert args.atn_non_linearity in ['sigmoid', 'gumbel', 'sqrt', 'unsupervisedsoftmax', 'softmax']
        self.non_lin = args.atn_non_linearity
        self.in_s = size_s
        self.in_t = size_t
        self.device = "cuda"

        self.supervision_type = args.supervision_type

        self.n_attn_maps_supervised = size_t
        self.n_attn_maps = args.nr_attention_maps
        self.attention_kernel_size = args.attention_kernel_size
        if size_attributes is not None:
            self.n_attributes = len(size_attributes)
        else:
            self.n_attributes = 0

        # init basemodel with student input only
        super().__init__(output_size=output_size, input_size=size_s, args=args)

        # attribute attentions
        self.is_separate_atn_maps = False
        if size_attributes is not None:
            if self.is_separate_atn_maps:
                assert (self.n_attn_maps - self.n_attn_maps_supervised > 0)
                if self.n_attn_maps - self.n_attn_maps_supervised == self.n_attributes:
                    self.attribute_prediction = nn.ModuleList(
                        [nn.Linear(self.nfeat_backbone, size_attributes[i]) for i in range(self.n_attributes)])
                else:
                    self.attribute_prediction = nn.ModuleList([nn.Linear(
                        (self.n_attn_maps - self.n_attn_maps_supervised) * self.nfeat_backbone, size_attributes[i]) for i in
                                                               range(self.n_attributes)])
            else:
                self.attribute_prediction = nn.ModuleList(
                    [nn.Linear(self.nfeat_backbone, size_attributes[i]) for i in range(self.n_attributes)])

                self.w_attention_featurematrix =  nn.Linear(self.n_attn_maps, self.n_attributes, bias=False)

        # Attention Maps
        if self.attention_kernel_size == "1x1":
            self.attentions = nn.Conv2d(self.nfeat_backbone, self.n_attn_maps_supervised, kernel_size=1, bias=False)
        elif self.attention_kernel_size == "3x3":
            self.attentions = nn.Sequential(
                nn.Conv2d(in_channels=self.nfeat_backbone, out_channels=512, kernel_size=(3, 3), padding=1, stride=1,
                          bias=True),
                nn.ReLU(True),
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=self.n_attn_maps_supervised, kernel_size=(3, 3), padding=1, stride=1,
                          bias=True))
        elif self.attention_kernel_size == "3x3l":
            self.attentions = nn.Sequential(
                nn.Conv2d(in_channels=self.nfeat_backbone, out_channels=512, kernel_size=(3, 3), padding=1, stride=1,
                          bias=True),
                nn.ReLU(True),
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(in_channels=128, out_channels=self.n_attn_maps_supervised, kernel_size=(3, 3), padding=1, stride=1,
                          bias=True))
        elif self.attention_kernel_size == "3x3s":
            self.attentions = nn.Sequential(
                nn.Conv2d(in_channels=self.nfeat_backbone, out_channels=256, kernel_size=(3, 3), padding=1, stride=1,
                          bias=True),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=self.n_attn_maps_supervised, kernel_size=(3, 3), padding=1, stride=1,
                          bias=True))
        elif self.attention_kernel_size == '1x1pre':
            pass
        else:
            raise NotImplementedError

        if self.n_attn_maps - self.n_attn_maps_supervised > 0:
            if self.args.attention_kernel_size_attributes == "1x1":
                self.attentions_attributes = nn.Conv2d(self.nfeat_backbone, self.n_attn_maps - self.n_attn_maps_supervised, kernel_size=1, bias=False)
            elif self.args.attention_kernel_size_attributes == '3x3':
                self.attentions_attributes = nn.Sequential(
                    nn.Conv2d(in_channels=self.nfeat_backbone, out_channels=512, kernel_size=(3, 3), padding=1,
                              stride=1,
                              bias=True),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=True),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels=256, out_channels=self.n_attn_maps - self.n_attn_maps_supervised, kernel_size=(3, 3), padding=1,
                              stride=1,
                              bias=True))
            else:
                raise NotImplementedError

        if 'iSQRT' in self.args.model:
            self.pool = COVPool(input_dim=self.nfeat_backbone,non_lin=self.non_lin, args=self.args, n_supervised=self.n_attn_maps_supervised)
        else:
            self.pool = BAP(pool='GAP', non_lin=self.non_lin, n_supervised=self.n_attn_maps_supervised)
        # max pool conv layer
        if 'maxpool' in self.args.tag:
            cnnmaxpool = nn.Conv2d(self.nfeat_backbone, self.nfeat_backbone, kernel_size=1, bias=False)
            self.bmp = BAP(pool='GMP', non_lin=self.non_lin, convmaxpool=cnnmaxpool)


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

            assert not 'iSQRT' in self.args.tag, 'not implemented'
        else:
            if 'iSQRT' in self.args.model:
                self.model_out = nn.Sequential(nn.Linear(self.pool.output_dim, output_size),
                                                nn.LogSoftmax(dim=1))
            else:
                self.model_out = nn.Sequential(nn.Linear(nfeat_ * self.n_attn_maps, output_size),
                                                nn.LogSoftmax(dim=1))
        self.feature_center = torch.zeros(output_size, self.n_attn_maps, self.nfeat_backbone).to(
            torch.device(self.device))  # net.expansion)#

        if "resnet"  in self.args.backbone:
            x = np.linspace(-1,1,28)
        elif "inception" in self.args.backbone:
            x = np.linspace(-1,1,26)
        else:
            raise NotImplementedError

        if self.supervision_type == 'keyloc':
            ref_ = 448 if self.args.is_crop_train else 800
            x = np.linspace(0.1, 1, ref_)
            self.x_grid = torch.from_numpy(x).float().to(torch.device(self.device))
        if 'regularizedoverlapcenter' in self.args.tag:
            x = np.linspace(0.1, 1, 28)
            self.x_grid_small = torch.from_numpy(x).float().to(torch.device(self.device))

        self.dropout = nn.Dropout2d(p=0.5)

        self.temperature = 0.07

    # Spatial transformer network forward function
    def get_stn_theta(self,x):

        xs, _ = self.transformer(x)
        xs = self.localization(xs).view(x.shape[0], -1)
        theta = self.fc_loc(xs)
        theta = utils_nn.reshape_theta(theta, xs, n_params=self.params_per_trans)
        return theta

    def stn(self, x):
        N, C, _,_ = x.size()
        W = self.args.crop_size_transformer
        theta = self.get_stn_theta(x)
        grid = F.affine_grid(theta, size=[N, C, W, W])
        x = F.grid_sample(x, grid)
        x = utils_nn.Grad_damp.apply(x, 1e-4)
        return x, theta


    def my_CE_loss(self,pred, target):
        return -torch.sum(target * pred)

    def get_attribute_predictions(self, feature_matrix):

        if not self.is_separate_atn_maps:

            preds = []
            # sigmoid on w
            feature_attribute = F.linear(feature_matrix.transpose(2,1), torch.sigmoid(self.w_attention_featurematrix.weight))
            for i in range(0, self.n_attributes):
                pred = torch.log_softmax(self.attribute_prediction[i](feature_attribute[...,i]), dim=1)
                preds.append(pred)

            return preds

        preds = []
        B = feature_matrix.shape[0]

        for i in range(0, self.n_attributes):

            if self.n_attn_maps - self.n_attn_maps_supervised == self.n_attributes:
                feature_attribute = feature_matrix[:,
                                    self.n_attn_maps_supervised + i:self.n_attn_maps_supervised + i + 1]
            else:
                feature_attribute = feature_matrix[:, self.n_attn_maps_supervised:]

            pred = torch.log_softmax(self.attribute_prediction[i](feature_attribute.view(B, -1)), dim=1)
            preds.append(pred)

        return preds

    def get_centers(self,unnormalized_map,prediction=True):

        B = unnormalized_map.shape[0]
        centers = []
        for i in range(0, unnormalized_map.shape[1]):
            sum_map = torch.sum((unnormalized_map[:, i:i + 1]).view(B, -1), dim=1, keepdim=True)
            val_map = sum_map > 1e-6

            t_x = torch.sum((unnormalized_map[:, i:i + 1] * self.x_grid).view(B, -1), dim=1, keepdim=True)
            t_y = torch.sum((unnormalized_map[:, i:i + 1] * self.y_grid).view(B, -1), dim=1, keepdim=True)

            if prediction:
                t_x = t_x / sum_map
                t_y = t_y / sum_map
                t = torch.cat((t_x.unsqueeze(2), t_y.unsqueeze(2)), dim=2)
            else:
                t_x[val_map] = t_x[val_map] / sum_map[val_map]
                t_y[val_map] = t_y[val_map] / sum_map[val_map]
                val_map = val_map.float()
                t = torch.cat((t_x.unsqueeze(2), t_y.unsqueeze(2), val_map.unsqueeze(2)), dim=2)

            centers.append(t)

        centers = torch.cat(centers, dim=1)

        return centers
    def get_centers_sm(self,unnormalized_map,prediction=True):

        B = unnormalized_map.shape[0]
        centers = []
        for i in range(0, unnormalized_map.shape[1]):
            sum_map = torch.sum((unnormalized_map[:, i:i + 1]).view(B, -1), dim=1, keepdim=True)
            val_map = sum_map > 1e-6

            t_x = torch.sum((unnormalized_map[:, i:i + 1] * self.x_grid_small).view(B, -1), dim=1, keepdim=True)
            t_y = torch.sum((unnormalized_map[:, i:i + 1] * self.x_grid_small).view(B, -1), dim=1, keepdim=True)

            if prediction:
                t_x = t_x / sum_map
                t_y = t_y / sum_map
                t = torch.cat((t_x.unsqueeze(2), t_y.unsqueeze(2)), dim=2)
            else:
                t_x[val_map] = t_x[val_map] / sum_map[val_map]
                t_y[val_map] = t_y[val_map] / sum_map[val_map]
                val_map = val_map.float()
                t = torch.cat((t_x.unsqueeze(2), t_y.unsqueeze(2), val_map.unsqueeze(2)), dim=2)

            centers.append(t)

        centers = torch.cat(centers, dim=1)

        return centers

    
    def info_nce_loss(self, features, option=1):

        B = features.shape[0]
        n_maps = features.shape[1]

        if option == 1:
            # take all batch samples and atn maps as different neg samples
            labels = torch.arange(B*n_maps)
            
        elif option == 2:
            # take each batch sample as positive example, each atn as negative 
            features = torch.transpose(features, 0,1)
            labels = torch.cat([torch.arange(n_maps) for _ in range(B)], dim=0)
        elif option == 3:
            # take each atn as negative, no positive samples 
            features = torch.transpose(features, 0,1)
            labels = torch.cat([torch.arange(n_maps) for _ in range(B)], dim=0)
        else:
            raise NotImplementedError
        
        features = features.reshape(B*n_maps,-1)

        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels


    def forward(self, input, is_crop_atn=True):
        return_dict = dict()

        # if 'temperature' in input.keys():
        #     tao = input['temperature']
        # else:
        #     tao = 0.5

        input_list = [input[key] for key in self.input_keys]
        x = torch.cat(input_list, dim=1)

        feature_maps = self.backbone(x)
        if isinstance(feature_maps,tuple):
            feature_maps = feature_maps[0]
        if 'regularizedcenter' in self.args.tag:
            return_dict['feature_maps'] = feature_maps

        raw_attention_maps, attention_maps, feature_matrix, dict_ = self.forward_atention(feature_maps)

        return_dict = {**return_dict,**dict_}

        dict_ = self.forward_attention_teacher(input=input, x=x, raw_attention_maps=raw_attention_maps)
        return_dict = {**return_dict,**dict_}


        if 'dropout' in self.args.tag:
            feature_matrix = self.dropout(feature_matrix)
        imout = self.model_out(feature_matrix.view(feature_matrix.shape[0], -1))

        if not self.training and 'testatncropping' in self.args.tag and is_crop_atn:
            mean_attention_map = return_dict['zs'][:,:self.n_attn_maps_supervised].mean(dim=1,keepdim=True)
            img_cropped, bboxes = self.batch_augment(images=x, attention_map=mean_attention_map,mode='crop')
            return_dict1 = self.forward({'s':img_cropped}, is_crop_atn=False)
            imout = (return_dict1['s'] + imout) / 2
            return_dict['x_trans'] = img_cropped
            return_dict['bboxes'] = bboxes

        return_dict['s'] = imout
        return_dict['feature_m'] = feature_matrix

        return return_dict

    def forward_atention(self, feature_maps):
        dict_ = {}
        if self.attention_kernel_size == '1x1pre':
            raw_attention_maps = self.backbone.Mixed_7a.branch3x3_1(feature_maps)[:, :self.n_attn_maps, ...]
        else:
            raw_attention_maps = self.attentions(feature_maps)

        if self.n_attn_maps - self.n_attn_maps_supervised > 0:
            raw_attention_maps_attributes = self.attentions_attributes(feature_maps)
            raw_attention_maps = torch.cat((raw_attention_maps, raw_attention_maps_attributes), dim=1)

        if self.non_lin == 'sqrt':
            raw_attention_maps = torch.relu(raw_attention_maps)

        feature_matrix, attention_maps = self.pool(feature_maps, raw_attention_maps)
        if 'maxpool' in self.args.tag:
            feature_matrix_avgpool = feature_matrix
            feature_matrix_maxpool, _ = self.bmp(feature_maps, raw_attention_maps)
            dict_['feature_m_max'] = feature_matrix_maxpool
            dict_['feature_m_avg'] = feature_matrix_avgpool
            feature_matrix = torch.cat((feature_matrix_avgpool, feature_matrix_maxpool), dim=1)

        if self.supervision_type == "keyloc":
            dict_['zs_loc'] = self.get_centers(attention_maps)

        if self.non_lin == 'gumbel':
            attention_maps, attention_maps_soft = attention_maps

        if self.supervision_type in ['direct', 'positive','l2', 'keyloc']:
            dict_['zs'] = attention_maps
            if self.non_lin == 'gumbel':
                dict_['zs'] = attention_maps_soft
        else:
            dict_['zs'] = raw_attention_maps

        ### feature attention
        if self.n_attributes > 0:
            feature_matrix_att = feature_matrix
            dict_['t_attr_pred'] = self.get_attribute_predictions(feature_matrix_att)

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
                if 'trans' in self.args.tag:
                    with torch.no_grad():
                        theta = self.get_stn_theta(x)
                    grid = F.affine_grid(theta, attention_maps_t.size())
                    attention_maps_t = F.grid_sample(attention_maps_t, grid)

            elif self.supervision_type == "keyloc":
                with torch.no_grad():
                    attention_maps_t = input['t']
                    attention_maps_t = F.adaptive_max_pool2d(attention_maps_t, output_size=(raw_attention_maps.shape[2], raw_attention_maps.shape[3]))
                    dict_['zt_loc'] = self.get_centers(attention_maps_t,prediction=False)

        else:
            attention_maps_t = None

        dict_['zt'] = attention_maps_t

        return dict_

    def get_loss(self, input, output, label, phi_model, alpha):

        loss = 0
        B = output['feature_m'].shape[0]
        if 'trans' in self.args.tag:
            thetas = output['theta']

            if 'weaktrans' in self.args.tag:
                if alpha > 0.01:
                    loss += self.L2_loss(output['theta_gt_weak'][:,:2,:], thetas) * alpha
            else:
                loss += self.L2_loss(output['theta_gt'][:,:2,:], output['theta'])

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
            if 'overlapcenter' in self.args.tag:
                n_unshared = self.n_attn_maps - shared_maps
                d = self.args.dist_center
                if n_unshared > 1:
                    z_ = output['zs'][:, shared_maps:]
                    z_centers = self.get_centers_sm(z_)

                    for i in range(n_unshared-1):
                        for j in range(i+1,n_unshared):
            
                            dist_center = (z_centers[:,i] - z_centers[:,j]).pow(2).sum(-1)
                            
                            if 'regularizedoverlapcenter100' in self.args.tag:
                                loss -=  100*(dist_center[dist_center < d]).mean()
                            else:
                                loss +=  (dist_center[dist_center < d]-d).mean()
            if 'regularizedoverlap' in self.args.tag and not 'overlapcenter' in self.args.tag:
                n_unshared = self.n_attn_maps - shared_maps
                if n_unshared > 1:
                    z_ = output['zs'][:, shared_maps:]

                    for i in range(n_unshared):
                        for j in range(i,n_unshared):
                            zi = z_[i].detach()
                            
                            overlap =  + z_[j]/z_[i].sum(-1)
                            loss += self.bce_loss(overlap > 1, torch.zeros_like(overlap))

            elif 'regularizednceloss' in self.args.tag:
                n_unshared = self.n_attn_maps - shared_maps
                if n_unshared > 1:
                    z_ = output['zs'][:, shared_maps:]

                    if 'regularizednceloss1' in self.args.tag:
                        option = 1
                        logits, labels = self.info_nce_loss(z_, option=option)
                        loss += self.ce_loss(logits, labels)
                    elif 'regularizednceloss2' in self.args.tag:
                        option = 2
                        logits, labels = self.info_nce_loss(z_, option=option)
                        loss += self.ce_loss(logits, labels)

                    elif  'regularizednceloss3' in self.args.tag:
                        for i in range(z_.shape[0]):
                            z_i = z_[i].unsqueeze(0)
                            logits, labels = self.info_nce_loss(z_i, option=1)
                            loss += self.ce_loss(logits, labels)
                            
            elif 'regularizedbbox_shared' in self.args.tag:
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
            elif 'regularizedvariancerevised' in self.args.tag:
                n_unshared = self.n_attn_maps - shared_maps
                if n_unshared > 1:

                    z_ = output['zs'][:, shared_maps:]
                    z_mean = z_.mean(dim=(2, 3), keepdim=True)
                    z_var = (z_-z_mean)**2
                    # max the variance of the attention maps
                    loss -= z_var.mean()

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
