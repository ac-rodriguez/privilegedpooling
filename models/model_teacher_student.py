import torch
import torch.nn as nn
import numpy as np


from models import model_base


class CNN1(model_base.CNN):
    def __init__(self,output_size, size_s,size_t,args):

        # init basemodel with student input only
        super().__init__(output_size=output_size,input_size=size_s,args=args,input_keys=['s'])

        self.in_s = size_s
        self.in_t = size_t

        nfeat = 128
        self.teacher = nn.Sequential(nn.Conv2d(in_channels=size_t,out_channels=nfeat//2,kernel_size=3),
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channels=nfeat//2, out_channels=nfeat//2, kernel_size=3, stride=2),
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channels=nfeat//2, out_channels=nfeat, kernel_size=3, stride=2),
                                     nn.ReLU(True),
                                     nn.Conv2d(nfeat, nfeat, kernel_size=3, stride=2),
                                     nn.ReLU(True),
                                     nn.Conv2d(nfeat, self.nfeat_backbone, kernel_size=3, stride=2),
                                     nn.ReLU(True)
                                     )


        self.teacher_out = nn.Sequential(nn.Linear(self.nfeat_backbone, output_size),
                                         nn.LogSoftmax(dim=1))



    def forward(self, input):

        output_dict = super().forward(input)

        if 't' in input.keys():
            x_t = self.teacher(input['t'])

            zt = self.avgpool(x_t).flatten(1)

            imt_out = self.teacher_out(zt)

            output_dict['t'] = imt_out
            output_dict['zt'] = zt

        return output_dict

    def get_loss(self, input, output, label, phi_model, alpha=1.0):

        if self.args.model == 'L2_s':
            loss = self.L2_loss(output['zs'], output['zt'])

        elif self.args.model == 'L2_st':
            loss = self.L2_loss(output['zs'], output['zt']) + self.nll_loss(output['t'], label)

        elif self.args.model == 'DANN':

            dom_s = phi_model(output['zs'], alpha)
            dom_t = phi_model(output['zt'], alpha)

            domain_lab = torch.cat((torch.ones(dom_s.shape[0]),
                                    torch.zeros(dom_t.shape[0]))).long()
            domain_logps = torch.cat((dom_s, dom_t), dim=0)

            loss_domain = self.nll_loss(domain_logps.to(self.device), domain_lab.to(self.device))

            loss = loss_domain
        else:
            raise NotImplementedError

        return loss

