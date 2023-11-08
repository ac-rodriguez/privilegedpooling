import torch
import torch.nn as nn

from models.backbone import build_backbone
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from models import model_base


class CNN1(model_base.CNN):
    def __init__(self, output_size, size_s, size_t, args):

        # init basemodel with student input only
        super().__init__(output_size=output_size, input_size=size_s, args=args, input_keys=['s'])

        self.in_s = size_s
        self.in_t = size_t
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # assert (args.x_star == "attributes")

        self.is_masked_images = 'masked' in args.x_star
        if not self.is_masked_images:
            nfeat_t = 256
            self.teacher = nn.Sequential(nn.Conv2d(in_channels=size_t,out_channels=nfeat_t//2,kernel_size=3),
                                         nn.ReLU(True),
                                         nn.Conv2d(in_channels=nfeat_t//2, out_channels=nfeat_t//2, kernel_size=3, stride=2),
                                         nn.ReLU(True),
                                         nn.Conv2d(in_channels=nfeat_t//2, out_channels=nfeat_t, kernel_size=3, stride=2),
                                         nn.ReLU(True),
                                         nn.Conv2d(nfeat_t, nfeat_t, kernel_size=3, stride=2),
                                         nn.ReLU(True),
                                         # nn.Conv2d(nfeat_t, args.latent_size_upper, kernel_size=3, stride=2),
                                         # nn.ReLU(True)
                                         )

            self.model_out1 = nn.Sequential(
                nn.Linear(self.nfeat_backbone + nfeat_t, output_size),
                nn.LogSoftmax(dim=1))
        else:
            self.model_out1 = nn.Sequential(
                nn.Linear(self.nfeat_backbone * 2, output_size),
                nn.LogSoftmax(dim=1))

    def forward(self, input):

        x_s, _ = self.backbone(input['s'])

        if self.is_masked_images:
            x_t, _ = self.backbone(input['t'])
        else:
            x_t = self.teacher(input['t'])
        # x_t = nn.Upsample(size=(x_s.shape[2],x_s.shape[3]), mode='bilinear')(x_t)

        # x = torch.cat((x_s,x_t),dim=1)
        zt = self.avgpool(x_t).flatten(1)
        zs = self.avgpool(x_s).flatten(1)

        zs = torch.cat((zs, zt), dim=1)
        logits_s = self.model_out1[0](zs)

        imout = self.model_out1[1](logits_s)

        return {'s': imout,
                'zs': zs,
                'logits_s':logits_s}

    def get_loss(self, input, output, label, phi_model, alpha):

        if self.args.model == 'upper':
            loss = 0
        else:
            raise NotImplementedError

        return loss

