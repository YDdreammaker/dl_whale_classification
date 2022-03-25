import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # phi = cosine * self.cos_m - sine * self.sin_m
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # for autocast
        cosine, sine, phi = cosine.float(), sine.float(), phi.float()

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(output.shape) # B, 15587
        return output


class SpecieClassifier(nn.Module):
    def __init__(self, model_name, emb_size=512, n_classes=26, easy_margin=False, ls_eps=0.0, p=0.):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features=in_features, out_features=512)
        self.out = nn.Linear(in_features=512, out_features=n_classes)

        self.act = nn.ReLU(inplace=True)
        
    def forward(self, image):
        emb = self.act(self.model(image))
        output = self.out(emb)

        return output


class IndividualClassifier(nn.Module):
    def __init__(self, model_name, emb_size=512, n_classes=15587, s=30.0, m=0.30, easy_margin=False, ls_eps=0.0, p=0.):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features=in_features, out_features=emb_size)
        self.arc = ArcMarginProduct(emb_size, n_classes, s=s, m=m, easy_margin=easy_margin, ls_eps=ls_eps)
        
    def forward(self, image, label):
        emb1 = self.model(image) # later 1 return test
        emb2 = self.attn(emb1)
        output = self.arc(emb2, label)
        return output, emb2