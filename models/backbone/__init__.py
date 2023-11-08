from torch import nn
from models.backbone import resnet, xception, drn, mobilenet, resnet_model_HBP, inception, resnet_original
from models.iSQRT.src.network.mpncovresnet import mpncovresnet50,mpncovresnet101

def build_backbone(backbone, output_stride, BatchNorm=nn.BatchNorm2d, in_ch=3, pretrained=True,bilinear_pooling=False, final_stride=1):
    if not bilinear_pooling:
        if backbone == 'resnet101':
            return resnet.ResNet101(output_stride, BatchNorm,in_ch=in_ch,pretrained=pretrained), 512*4
        elif backbone == 'resnet50':
            return resnet.ResNet50(output_stride, BatchNorm,in_ch=in_ch,pretrained=pretrained), 512*4
        elif backbone == 'resnet50_original':
            return resnet_original.resnet50(pretrained, final_stride=final_stride), 512*4
        elif backbone == 'resnet101_original':
            return resnet_original.resnet101(pretrained, final_stride=final_stride), 512*4
        elif backbone == 'resnet18_original':
            return resnet_original.resnet18(pretrained), 512
        elif backbone == 'xception':
            return xception.AlignedXception(output_stride, BatchNorm,in_ch=in_ch,pretrained=pretrained)
        elif backbone == 'drn':
            return drn.drn_d_54(BatchNorm,pretrained=pretrained)
        elif backbone == 'mobilenet':
            return mobilenet.MobileNetV2(output_stride, BatchNorm,pretrained=pretrained)
        elif backbone == 'inception':
            return inception.inception_v3(pretrained=pretrained), 768*1
        elif backbone == 'mpncovresnet50':
            model = mpncovresnet50(pretrained)
            return _reconstruct_mpncovresnet(model,pretrained), 256
        elif backbone == 'mpncovresnet101':
            model = mpncovresnet101(pretrained)
            return _reconstruct_mpncovresnet(model,pretrained), 256
        else:
            raise NotImplementedError
    else:
        if backbone == 'resnet50':
            return resnet_model_HBP.resnet50(pretrained=pretrained, BatchNorm=BatchNorm)
        else:
            raise NotImplementedError


def _reconstruct_mpncovresnet(basemodel, pretrained):
    model = nn.Module()
    if pretrained:
        model.features = nn.Sequential(*list(basemodel.children())[:-1])
        model.representation_dim=basemodel.layer_reduce.weight.size(0)
    else:
        model.features = nn.Sequential(*list(basemodel.children())[:-4])
        model.representation_dim=basemodel.layer_reduce.weight.size(1)
    model.representation = None
    model.classifier = basemodel.fc
    return model.features
