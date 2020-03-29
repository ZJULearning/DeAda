from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from collections import OrderedDict

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


class DenseNet(nn.Module):
    __factory = {
        121: torchvision.models.densenet121,
        169: torchvision.models.densenet169,
        201: torchvision.models.densenet201,
        161: torchvision.models.densenet161,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,reset_acell = True):
        super(DenseNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) densenet
        if depth not in DenseNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = DenseNet.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.classifier.in_features

            # Append new layers
            if self.has_embedding:
                self.embed = nn.Linear(out_planes, self.num_features)
                self.bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.embed.weight, mode='fan_out')
                init.constant_(self.embed.bias, 0)
                init.constant_(self.bn.weight, 1)
                init.constant_(self.bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

        if not self.pretrained:
            self.densenet_params()


    def forward(self, x):
        x1 = self.base.features.pool0(self.base.features.relu0(self.base.features.norm0(self.base.features.conv0(x))))
        x2 = self.base.features.transition1(self.base.features.denseblock1(x1))
        x3 = self.base.features.transition2(self.base.features.denseblock2(x2))
        x4 = self.base.features.transition3(self.base.features.denseblock3(x3))
        x = self.base.features.denseblock4(x4)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.embed(x)
        x = self.bn(x)
        feature = F.normalize(x)
        x = F.relu(x)
        y = self.classifier(x)

        if self.training:
            return y
        
        return feature

    def densenet_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def densenet121(**kwargs):
    return DenseNet(121, **kwargs)


def densenet169(**kwargs):
    return DenseNet(169, **kwargs)


def densenet201(**kwargs):
    return DenseNet(201, **kwargs)


def densenet161(**kwargs):
    return DenseNet(161, **kwargs)

