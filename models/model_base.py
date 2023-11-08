import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import build_backbone
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


def crop(variable, tw, th):
    _, _, w, h = variable.size()
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[...,x1:x1 + tw, y1:y1 + th]

class CNN(nn.Module):
    def __init__(self, output_size, input_size, args, input_keys=['s']):
        super(CNN, self).__init__()

        self.args=args

        self.bce_loss = torch.nn.BCELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.L2_loss = torch.nn.MSELoss()
        self.KL_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.nll_loss = torch.nn.NLLLoss()

        if args.sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.in_ch = input_size
        self.input_keys = input_keys

        self.backbone, self.nfeat_backbone = build_backbone(args.backbone, output_stride=16, BatchNorm=BatchNorm,
                                                            in_ch=self.in_ch, pretrained=args.pretrained, final_stride=args.final_stride)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        if args.latent_size != -1:
            assert(args.latent_size == self.nfeat_backbone)

        if 'maxpool' in self.args.tag:
            self.model_out = nn.Sequential(nn.Linear(self.nfeat_backbone*2, output_size),
                                           nn.LogSoftmax(dim=1))
        else:
            self.model_out = nn.Sequential(nn.Linear(self.nfeat_backbone, output_size),
                                           nn.LogSoftmax(dim=1))


    def forward(self, input):

        input_list = [input[key] for key in self.input_keys]
        x = torch.cat(input_list,dim=1)

        x = self.backbone(x)
        if isinstance(x,tuple):
            x = x[0]
        zs = self.avgpool(x).flatten(1)
        if 'maxpool' in self.args.tag:
            zs_max = self.maxpool(x).flatten(1)
            zs = torch.cat((zs, zs_max), dim=1)

        return_dict = {}
        logits_s = self.model_out[0](zs)
        return_dict['logits_s'] = logits_s
        return_dict['s'] = self.model_out[1](logits_s)
        return_dict['zs'] = zs

        return return_dict

    def get_multiscale_preds(self, input):

        input_list = [input[key] for key in self.input_keys]
        X = torch.cat(input_list, dim=1)

        preds = []
        # for t in range(input['t'].shape[1]):
        #     key_point = input['t'][:,t]
        for s in [0.8, 1, 1.1]:
            x = F.interpolate(X, scale_factor=s, mode='bilinear')
            x = crop(x, 448, 448)

            x, _ = self.backbone(x)
            zs = self.avgpool(x).flatten(1)

            imout = self.model_out(zs)

            preds.append(imout)
        preds = torch.stack(preds,dim=1) #.mean(dim=1)

        preds, _ = torch.max(preds,dim=1)

        return {'s': preds,
                'zs': zs}

    def get_lr_params(self, keyword='backbone', inclusive=True):

        modules = self._modules.items()
        if inclusive:
            modules = [x for x in modules if keyword in x]
        else:
            modules = [x for x in modules if not keyword in x]

        for k, m1 in modules:
            for m in m1.named_modules():
                # if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                #         or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.Linear):
                for p in m[1].parameters():
                    if p.requires_grad:
                        yield p


    def get_loss(self, input, output, label, phi_model, alpha):

        if self.args.model == 'dist-vanilla':
            with torch.no_grad():
                teacher_logits = phi_model(input)['logits_s']
            soft_teacher_targets = F.softmax(teacher_logits / self.args.temperature, dim=1)

            soft_log_student_targets = F.log_softmax(output['logits_s'] / self.args.temperature, dim=1)

            loss = self.KL_loss(soft_log_student_targets, soft_teacher_targets)
        elif self.args.model == 'student':
            loss = 0
        else:
            raise NotImplementedError

        return loss

