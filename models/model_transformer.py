import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.backbone import build_backbone
from models import model_base

from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Grad_damp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha=0.1):
        ctx.alpha = alpha
        """ x = I(x) """
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None


def crop(variable, tw, th):
    _, _, w, h = variable.size()
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[...,x1:x1 + tw, y1:y1 + th]

class CNN1(model_base.CNN):
    def __init__(self, output_size, input_size, args, input_keys=['s']):

        super(CNN1, self).__init__(output_size=output_size, input_size=input_size, args=args, input_keys=input_keys)

        if 'fullkeypoints' in self.args.tag:
            assert 'keypoints' in self.args.x_star
            self.trans_type = 'keypoints'
            self.params_per_trans = 6
        elif 'full' in self.args.tag:
            self.trans_type = 'affine'
            self.params_per_trans = 6
        else:
            self.trans_type = 'scale-rotation'
            self.params_per_trans = 6

        self.transformer, nfeat_trans = build_backbone('resnet18_original', pretrained=True, output_stride=16)

        self.localization = nn.Sequential(
            nn.Conv2d(nfeat_trans, 128, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Linear(128 * 7 * 7, self.params_per_trans)
        self.fc_loc.weight.data.zero_()
        if self.params_per_trans == 6:
            # Initialize the weights/bias with identity transformation
            self.fc_loc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif self.params_per_trans == 3:
            # 3 param
            self.fc_loc.bias.data.copy_(torch.tensor([1, 0, 0], dtype=torch.float))
        else:
            raise NotImplementedError

        x = np.linspace(0.1, 1, 448)
        self.x_grid = torch.from_numpy(x).float().to(torch.device("cuda"))

    # Spatial transformer network forward function
    def get_stn_theta(self,x):

        xs, _ = self.transformer(x)
        xs = self.localization(xs)
        xs = F.adaptive_max_pool2d(xs, output_size=(7, 7))
        xs = xs.view(xs.shape[0], -1)
        theta_flat = self.fc_loc(xs)
        return theta_flat

    def reshape_theta(self, theta_flat, x):
        if self.params_per_trans == 6:
            theta = theta_flat.view(-1, 2, 3)
        elif self.params_per_trans == 3:
            theta = torch.zeros(x.shape[0], 2, 3, device=x.device)
            # scale
            theta[:, 0, 0] = theta_flat[:,0]
            theta[:, 1, 1] = theta_flat[:,0]
            # theta[:, 0, 0] = torch.clamp(theta_flat[:,0], -1, 1)
            # theta[:, 1, 1] = torch.clamp(theta_flat[:,0], -1, 1)

            # tx, ty
            theta[:, 0, 2] = theta_flat[:,1]
            theta[:, 1, 2] = theta_flat[:,2]
        else:
            raise NotImplementedError

        return theta

    def stn(self, x, scale=1.0):
        theta_flat = self.get_stn_theta(x)
        theta = self.reshape_theta(theta_flat, x)
        size = x.size()
        if scale != 1:
            size = [size[0], size[1], int(size[2]//scale), int(size[3]//scale)]

        grid = F.affine_grid(theta, size)
        x = F.grid_sample(x, grid)
        return Grad_damp.apply(x, 1e-4), theta

    def forward(self, input):

        return_dict = {}
        input_list = [input[key] for key in self.input_keys]
        x = torch.cat(input_list, dim=1)

        x, theta = self.stn(x, scale=1.0)

        return_dict['x_trans'] = x

        x, _ = self.backbone(x)
        zs = self.avgpool(x).flatten(1)

        return_dict['s'] = self.model_out(zs)
        return_dict['zs'] = zs
        return_dict['theta'] = theta

        return return_dict

    def get_gt_theta(self, input):
        all_keypoints = input['t'].sum(dim=1)

        ax = (all_keypoints.sum(dim=(-2)) > 0).float()
        _, xmin = ((1-self.x_grid)*ax).max(dim=-1)
        _, xmax = (self.x_grid*ax).max(dim=-1)

        ay = (all_keypoints.sum(dim=(-1)) > 0).float()
        _, ymin = ((1-self.x_grid)*ay).max(dim=-1)
        _, ymax = (self.x_grid*ay).max(dim=-1)

        w0 = input['s'].shape[-1] // 2
        wx = (xmax - xmin) // 2 + 40
        wy = (ymax - ymin) // 2 + 40

        dx = xmin + wx
        dy = ymin + wy

        sx = wx.float() / w0
        sy = wy.float() / w0

        tx = (dx.float() - w0) / w0
        ty = (dy.float() - w0) / w0

        theta_gt = torch.stack([sx, torch.zeros_like(sx), tx,
                                torch.zeros_like(sx), sy, ty], dim=1).view(-1, 2, 3)

        return theta_gt

    def get_loss(self, input, output, label, phi_model, alpha):

        loss = 0

        if self.args.supervision_type == 'l2':

            theta_gt = self.get_gt_theta(input)

            loss = self.L2_loss(theta_gt,output['theta'])
        elif self.args.supervision_type == 'no':
            pass
        else:
            raise NotImplementedError




        # if self.args.model == 'dist-vanilla':
        #     with torch.no_grad():
        #         output_t = phi_model(input)
        #     loss = self.KL_loss(output_t['s'], output['s'], self.args.temperature)
        # elif self.args.model == 'student':
        #     loss = 0
        # else:
        #     raise NotImplementedError

        return loss
