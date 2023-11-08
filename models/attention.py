import torch
import torch.nn as nn


class attentionNet(nn.Module):
    def __init__(self,latent_size, fullattn=False):
        super(attentionNet, self).__init__()

        self.fullatn = fullattn
        if self.fullatn:
            self.w = torch.nn.Parameter(torch.randn(latent_size, latent_size) / 100)
        else:
            self.w = torch.nn.Parameter(torch.randn(latent_size)/100)
        
    def forward(self, input,is_detach = True):

        if is_detach:
            zs = input['zs'].detach()
            zt = input['zt'].detach()
        else:
            zs = input['zs']
            zt = input['zt']

        w = torch.sigmoid(self.w)

        if self.fullatn:
            zst = zs + torch.mm(zt, w)
        else:
            zst = (1- w)*zs + w*(zs+zt)/2.
            
        
        return {'zst':zst}


class attentionNetsample(nn.Module):
    def __init__(self, latent_size, fullattn=False):
        super(attentionNetsample, self).__init__()

        n_feat = 128
        self.fullatn = fullattn
        self.latent_size = latent_size
        if self.fullatn:
            self.attn_module = nn.Sequential(nn.Linear(latent_size * 2, n_feat),
                                             nn.ReLU(True),
                                             nn.Linear(n_feat, latent_size**2))
        else:
            self.attn_module = nn.Sequential(nn.Linear(latent_size * 2, n_feat),
                                             nn.ReLU(True),
                                             nn.Linear(n_feat, latent_size))

    def forward(self, input):
        zs = input['zs'].detach()
        zt = input['zt'].detach()

        w = torch.sigmoid(self.attn_module(torch.cat((zs, zt),dim=1)))
        if self.fullatn:
            w = w.view(w.shape[0], self.latent_size,self.latent_size)
            zst = zs + torch.bmm(zt.unsqueeze(1), w).squeeze(1)
        else:
            zst = (1 - w) * zs + w * (zs + zt) / 2.

        return {'zst': zst, 'w': w}


if __name__ == '__main__':
    import numpy as np

    atn = attentionNet(1000)

    model_parameters = filter(lambda p: p.requires_grad, atn.parameters())
    print('N param in model ', sum([np.prod(p.size()) for p in model_parameters]))

    print(atn)