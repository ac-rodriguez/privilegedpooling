import torch.nn as nn



from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DANN(nn.Module):
    def __init__(self,latent_size, n_feat=10):
        super(DANN, self).__init__()

        self.domain_classifier = nn.Sequential(nn.Linear(latent_size, n_feat),
                                               nn.ReLU(True),
                                               nn.Linear(n_feat, 2),
                                               nn.LogSoftmax(dim=1))

    def forward(self, z, alpha):

        reversed_input = ReverseLayerF.apply(z, alpha)
        dom_prob = self.domain_classifier(reversed_input)

        return dom_prob

