import torch
import torch.nn as nn
import numpy as np
from models import model_base
import torch.nn.functional as F


class CNN_multitask(model_base.CNN):
    def __init__(self, output_size, size_s, size_t, size_attributes=None, args=None):

        # init basemodel with student input only
        super().__init__(output_size=output_size, input_size=size_s, args=args)

        self.in_s = size_s
        self.in_t = size_t


        if size_attributes is not None:
            if args.attr_index == -1:
                self.n_attributes = len(size_attributes)
            else:
                self.n_attributes = 1
        else:
            self.n_attributes = 0

        nfeat = 256
        # Attention Maps
        if size_t > 0:
            if args.attention_kernel_size == "1x1":
                self.reconstruct = nn.Conv2d(self.nfeat_backbone, self.size_t, kernel_size=1, bias=True)
            elif args.attention_kernel_size == "3x3":
                self.reconstruct = nn.Sequential(
                    nn.Conv2d(in_channels=self.nfeat_backbone, out_channels=512, kernel_size=(3, 3), padding=1, stride=1,
                              bias=True),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=True),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels=256, out_channels=self.in_t, kernel_size=(3, 3), padding=1, stride=1,
                              bias=True))
            elif args.attention_kernel_size == "3x3s":
                self.reconstruct = nn.Sequential(
                    nn.Conv2d(in_channels=self.nfeat_backbone, out_channels=256, kernel_size=(3, 3), padding=1, stride=1,
                              bias=True),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels=256, out_channels=self.in_t, kernel_size=(3, 3), padding=1, stride=1, bias=True))
            else:
                raise NotImplementedError

        if size_attributes is not None:
            if self.n_attributes > 1:
                self.attribute_prediction = nn.ModuleList([nn.Linear(self.nfeat_backbone, size_attributes[i]) for i in range(self.n_attributes)])
            else:
                self.attribute_prediction = nn.ModuleList([nn.Linear(self.nfeat_backbone, size_attributes[args.attr_index]) for i in range(self.n_attributes)])


    def my_CE_loss(self,pred, target):
        return -torch.sum(target * pred)

    def get_attribute_predictions(self, feature_matrix):

        preds = []
        B = feature_matrix.shape[0]
        for i in range(0, self.n_attributes):
            pred = self.attribute_prediction[i](feature_matrix.view(B, -1))

            if  "catattributes" in self.args.x_star:
                pred = torch.log_softmax(pred, dim=1)
            elif "attributes" in self.args.x_star:
                pred = torch.sigmoid(pred)

            preds.append(pred)

        return preds

    def forward(self, input):

        input_list = [input[key] for key in self.input_keys]
        im = torch.cat(input_list, dim=1)

        im, _ = self.backbone(im)
        zs = self.avgpool(im)
        zs = torch.flatten(zs, 1)

        imout = self.model_out(zs)

        if 't' in input:
            xst = torch.sigmoid(self.reconstruct(im))

        t_attr_pred = None
        if self.n_attributes > 0:
            t_attr_pred = self.get_attribute_predictions(zs)
        # if 't' in input:
        #    xst = nn.functional.interpolate(xst,size=(input['t'].shape[2],input['t'].shape[3]), mode='bilinear')

        return_dict = dict()
        return_dict['s'] = imout
        return_dict['zs'] = zs

        if 't' in input:
            return_dict['xst'] = xst

        if self.n_attributes > 0:
            return_dict['t_attr_pred'] = t_attr_pred

        return return_dict

    def get_loss(self, input, output, label, phi_model, alpha):

        if ("resnet" in self.args.backbone):
            poolings = [(1, 1), (4, 4), (7, 7), (14, 14)]  # (28,28)
        elif "inception" in self.args.backbone:
            poolings = [(1, 1), (3, 3), (6, 6), (13, 13)]  # (26,26)
        else:
            raise NotImplementedError

        loss = 0.
        if 'xst' in output:
            t_maps_interp = F.adaptive_max_pool2d(input['t'],output_size=(output['xst'].shape[2], output['xst'].shape[3]))
            for pooling in poolings:
                loss += self.bce_loss(torch.nn.functional.max_pool2d(output['xst'], kernel_size=pooling),
                                      torch.nn.functional.max_pool2d(t_maps_interp, kernel_size=pooling))\

        if "t_attr_pred" in output:
            if "attributes" in self.args.x_star:
                for idx in range(0, len(input['t_attr'])):
                    loss += self.bce_loss(output['t_attr_pred'][idx].squeeze(),input['t_attr'][idx].float())

            if "catattributes" in self.args.x_star and self.args.attr_index == -1:
                for idx in range(0, len(input['t_attr'])):
                    sum_target = torch.sum(input['t_attr'][idx].float(), dim=1, keepdim=True)
                    p_t = input['t_attr'][idx].float()

                    valid_samples = sum_target.shape[0] - torch.sum(sum_target==0)
                    sum_target[sum_target == 0] = 1
                    sum_target = sum_target.float()
                    p_t = p_t / sum_target

                    loss += ( 1. / valid_samples) *self.my_CE_loss(output['t_attr_pred'][idx], p_t)
            elif "catattributes" in self.args.x_star:
                    sum_target = torch.sum(input['t_attr'][self.args.attr_index].float(), dim=1, keepdim=True)
                    p_t = input['t_attr'][self.args.attr_index].float()

                    valid_samples = sum_target.shape[0] - torch.sum(sum_target == 0)
                    sum_target[sum_target == 0] = 1
                    sum_target = sum_target.float()
                    p_t = p_t / sum_target

                    loss += (1. / valid_samples) * self.my_CE_loss(output['t_attr_pred'][0], p_t)

        return loss