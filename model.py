import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        return cosine


class HappyWhaleModel(nn.Module):
    def __init__(self, model_name, n_classes=15587, emb_size=2048, pretrain=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrain)
        self.model.conv_head = nn.Identity()
        self.model.bn2 = nn.Identity()
        self.model.act2 = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.fc = nn.Linear(512*2, emb_size)
        self.bn = nn.BatchNorm1d(emb_size)
        self.arc_margin_product = ArcMarginProduct(emb_size, n_classes)

    def forward(self, x, y):
        x = self.model(x)
        x = torch.cat((nn.AdaptiveAvgPool2d(1)(x), nn.AdaptiveMaxPool2d(1)(x)), dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        feature = self.bn(x)
        cosine = self.arc_margin_product(feature)

        return cosine, feature