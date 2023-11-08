import torch
import torch.nn as nn
import numpy as np

from models.model_base import CNN as CNN_Base

def vae_reparametrize_original(mu, std, log_normal=False):

    if log_normal:
        std.mul(0.5).exp_()

    rand_noise = torch.cuda.FloatTensor(std.size()).normal_() # [torch.cuda.FloatTensor of size 1x1x4096 (GPU 0)]

    out = mu + rand_noise*std

    return out


def vae_reparametrize(mu, std, log_normal=False):

    rand_noise = torch.cuda.FloatTensor(std.size()).normal_() # [torch.cuda.FloatTensor of size 1x1x4096 (GPU 0)]

    out = mu + rand_noise*std

    if log_normal:
        out.exp_()

    return out


class clamp_grad_to_zero(torch.autograd.Function):
    # Layer Definition -- layer just between convolutions and fc of x^*
    # operate on tensors

    @staticmethod
    def forward(ctx, x):
        """ x = I(x) """
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.fill_(0), None # or the tensor version of zero torch.cuda.FloatTensor(*x_star.size() ).fill_(0)
        # return 0


class CNN1(CNN_Base):
    def __init__(self,output_size, size_s,size_t,args):

        # init basemodel with student input only
        super().__init__(output_size=output_size,input_size=size_s,args=args,input_keys=['s'])

        # self.middle_layer = 'middlelayer' in self.args.tag  # as in paper
        self.is_log_normal = args.lognormal
        self.is_masked_images = 'masked' in args.x_star
        self.register_buffer('running_std', torch.ones(1,self.nfeat_backbone))

        self.reparametrize = vae_reparametrize_original if 'originalparametrize' in self.args.tag else vae_reparametrize

        self.is_2fc = '2fc' in self.args.tag
        self.running_avg_momentum = 0.9
        if self.is_2fc:
            self.fc_s2 = nn.Sequential(nn.Linear(self.nfeat_backbone, self.nfeat_backbone),
                                       nn.ReLU(True))

            self.register_buffer('running_std2', torch.ones(1, self.nfeat_backbone))
            self.running_avg_momentum2 = 0.9

        self.in_s = size_s
        self.in_t = size_t

        if self.is_masked_images:
            # masked images share network with the student

            self.fc_t = nn.Sequential(nn.Linear(self.nfeat_backbone, self.nfeat_backbone),
                                      nn.ReLU(True))

            if self.is_2fc:
                self.fc_t2 = nn.Sequential(nn.Linear(self.nfeat_backbone, self.nfeat_backbone),
                                          nn.ReLU(True))

        else:
            nfeat = 64
            self.teacher = nn.Sequential(nn.Conv2d(in_channels=size_t,out_channels=nfeat,kernel_size=3),
                                         nn.ReLU(True),
                                         nn.Conv2d(in_channels=nfeat, out_channels=nfeat, kernel_size=3),
                                         nn.ReLU(True),
                                         nn.Conv2d(in_channels=nfeat, out_channels=nfeat, kernel_size=3),
                                         nn.ReLU(True))
            self.fc_t = nn.Sequential(nn.Linear(nfeat,self.nfeat_backbone),
                                      nn.ReLU(True))

        # self.fc = nn.Sequential(nn.Linear(self.nfeat_backbone, self.nfeat_backbone), nn.ReLU(True))
        self.teacher_out = nn.Sequential(nn.Linear(self.nfeat_backbone, output_size),nn.LogSoftmax(dim=1))
        self.clamp_grad = clamp_grad_to_zero()


    def forward(self, input):

        x, _ = self.backbone(input['s'])
        zs = self.avgpool(x).flatten(1)
        if self.training:
            if self.is_masked_images:
                assert input['t'].shape[1] == 3

                x_t, _ = self.backbone(input['t'])
                x_t = self.avgpool(x_t).flatten(1)
                x_t = self.clamp_grad.apply(x_t)
                zt = self.fc_t(x_t)
            else:
                x_t = self.teacher(input['t'])
                x_t = self.avgpool(x_t)
                x_t = torch.flatten(x_t, 1)

                zt = self.fc_t(x_t)

            imt_out = self.teacher_out(zt)
        else:
            imt_out = None
            zt = None

        zs, sigma = self.compute_droput(zs,zt, fc='fc1')
        if self.is_2fc:
            zs = self.fc_s2(zs)
            if self.training:
                zt = self.fc_t2(zt)
            else:
                zt = None
            zs, sigma1 = self.compute_droput(zs, zt, fc='fc1')

        imout = self.model_out(zs)
        if self.is_log_normal:
            sigma = sigma.exp()
            if self.is_2fc:
                sigma = torch.cat([sigma.unsqueeze(0), sigma1.unsqueeze(0)], 0)
        return {'s': imout,
                'zs': zs,
                'sigma':sigma,
                't': imt_out,
                'zt': zt}

    def compute_droput(self, x,x_star, fc='fc1'):

        if self.training:
            sigma = x_star
            noise = self.reparametrize(mu=1.0, std=sigma, log_normal=self.is_log_normal)
            # self.running_std *= self.running_avg_momentum
            sample_std = sigma.mean(dim=0)
            # sample_std = (1 - self.running_avg_momentum) * per_image_sigma

            if fc == 'fc1':
                self.running_std = self.running_std * self.running_avg_momentum + \
                                   sample_std.data * (1-self.running_avg_momentum)
            else:
                self.running_std2 = self.running_std2 * self.running_avg_momentum + \
                                   sample_std.data * (1 - self.running_avg_momentum)
        else:
            if fc == 'fc1':
                sigma = self.running_std.expand(x.shape[0], *self.running_std.size()).squeeze()
            else:
                sigma = self.running_std2.expand(x.shape[0], *self.running_std2.size()).squeeze()

            noise = self.reparametrize(mu=1.0, std=sigma, log_normal=self.is_log_normal)


        return x.mul(noise), sigma


    def get_loss(self, input, output,label, model_phi, alpha):

        # teacher supervision only for non shared backbones
        if 'masked' in self.args.x_star:
            loss = 0
        else:
            loss = self.nll_loss(output['t'], label)

        noise_L2 = torch.norm(output['sigma'], 2)

        loss += noise_L2 * self.args.lambda_noise

        return loss
