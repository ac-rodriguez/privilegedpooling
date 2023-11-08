import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.backbone import build_backbone
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from models import model_base

# class grad_damp(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, alpha=0.1):
#         ctx.alpha = alpha
#         """ x = I(x) """
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output * ctx.alpha, None

class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)

# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()

        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions, temperature=None):
        B = features.size(0)
        M = attentions.size(1)
        EPSILON = 1e-12
        feature_matrix = []
        for i in range(M):
            AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
            #AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, 1, -1)
            #AiF = torch.sign(AiF) * torch.sqrt(torch.abs(AiF) + 1e-12)
            #AiF = F.normalize(AiF, dim=2, p=2)
            feature_matrix.append(AiF)

        feature_matrix = torch.cat(feature_matrix, dim=1)
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)
        feature_matrix = F.normalize(feature_matrix, dim=-1)

        return feature_matrix


class CNN1(model_base.CNN):

    def __init__(self, output_size, size_s, args=None):

        self.in_s = size_s
        self.n_attn_maps = args.nr_attention_maps
        self.attention_kernel_size = args.attention_kernel_size

        # init basemodel with student input only
        super().__init__(output_size=output_size, input_size=size_s, args=args)

        # Attention Maps
        assert self.attention_kernel_size in ["1x1","1x1pre","1x1b"]
        if self.attention_kernel_size == "1x1":
            self.attentions = nn.Sequential(nn.Conv2d(self.nfeat_backbone, self.n_attn_maps, kernel_size=1, bias=False),nn.BatchNorm2d(self.n_attn_maps, eps=0.001), nn.ReLU(True))
        elif self.attention_kernel_size == "1x1b":
            self.attentions = nn.Sequential(nn.Conv2d(self.nfeat_backbone, self.n_attn_maps, kernel_size=1, bias=True), nn.BatchNorm2d(self.n_attn_maps, eps=0.001), nn.ReLU(True))
        elif self.attention_kernel_size == '1x1pre':
            assert args.backbone == 'inception'
        else:
            raise NotImplementedError

        self.bap = BAP(pool='GAP')

        #self.feature_center = torch.zeros(output_size, self.n_attn_maps, self.nfeat_backbone).to(torch.device("cuda"))
        self.feature_center = torch.zeros(output_size, self.n_attn_maps * self.nfeat_backbone).to(torch.device("cuda"))

        self.model_out = nn.Sequential(nn.Linear(self.nfeat_backbone * self.n_attn_maps, output_size),
                                       nn.LogSoftmax(dim=1))

        self.center_loss = CenterLoss()

    def forward(self, input):

        input_list = [input[key] for key in self.input_keys]
        x = torch.cat(input_list, dim=1)

        feature_maps, _ = self.backbone(x)

        if self.attention_kernel_size == '1x1pre':
            attention_maps = self.backbone.Mixed_7a.branch3x3_1(feature_maps)[:, :self.n_attn_maps, ...]
            attention_maps = torch.relu(attention_maps)
        else:
            attention_maps = self.attentions(feature_maps)

        feature_matrix = self.bap(feature_maps, attention_maps)
        imout = self.model_out(feature_matrix.view(feature_matrix.shape[0], -1))

        # Randomly choose one of attention maps Ak
        attention_map = []
        for i in range(input['s'].shape[0]):
            attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + 1e-12)
            attention_weights = F.normalize(attention_weights, p=1, dim=0)
            k_index = np.random.choice(self.n_attn_maps, 2, p=attention_weights.cpu().numpy())
            attention_map.append(attention_maps[i, k_index, ...])
        attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping

        return_dict = dict()
        return_dict['s'] = imout
        return_dict['feature_m'] = feature_matrix
        return_dict['zs'] = attention_maps
        return_dict['crop_drop'] = attention_map

        return return_dict

    def get_loss(self, input, output, label, phi_model, alpha):

        loss = 0
        feature_m = output['feature_m']
        n_maps = feature_m.shape[1]

        if self.args.WS_DAN_regularization == "code":
            beta = 5e-2
            feature_center_batch = F.normalize(self.feature_center[label], dim=-1)
            self.feature_center[label] += beta * (feature_m.detach() - feature_center_batch)

            #feature_m = feature_m.view(feature_m.shape[0], -1)
            #centers_batch = self.feature_center[label]
            #centers_batch = centers_batch.view(centers_batch.shape[0], -1)
            #centers_batch = nn.functional.normalize(centers_batch, dim=-1, p=2)
            #diff = (1 - beta) * (centers_batch - feature_m.detach())
            #diff = diff.view(diff.shape[0], n_maps, -1)
            #self.feature_center[label] -= diff
            loss += self.center_loss(feature_m, feature_center_batch)  #self.L2_loss(feature_m, centers_batch)

        elif self.args.WS_DAN_regularization == "paper":
            loss += self.L2_loss(feature_m, self.feature_center[label].detach())
            beta = 0.01
            self.feature_center[label] += beta * (feature_m.detach() - self.feature_center[label]) * (alpha * self.args.lr)

        return loss

    def batch_augment(self,images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
        batches, _, imgH, imgW = images.size()

        if mode == 'crop':
            crop_images = []
            for batch_index in range(batches):
                atten_map = attention_map[batch_index:batch_index + 1]
                if isinstance(theta, tuple):
                    theta_c = np.random.uniform(*theta) * atten_map.max()
                else:
                    theta_c = theta * atten_map.max()

                crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
                nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
                height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
                height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
                width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
                width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

                crop_images.append(
                    F.upsample_bilinear(
                        images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                        size=(imgH, imgW)))
            crop_images = torch.cat(crop_images, dim=0)
            return crop_images

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
