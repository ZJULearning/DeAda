from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import copy
from collections import OrderedDict

__all__ = ['DenseNet', 'densenet169_sif', 'densenet201_sif', 'densenet161_sif', 'densenet121_sif']


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
        self.neg_denseblock3 = copy.deepcopy(self.base.features.denseblock3)
        self.neg_transition3 = copy.deepcopy(self.base.features.transition3)
        self.neg_denseblock4 = copy.deepcopy(self.base.features.denseblock4)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            out_planes = self.base.classifier.in_features

            # Append new layers
            if self.has_embedding:
                self.embed_pos = nn.Linear(out_planes, self.num_features)
                self.embed_neg = nn.Linear(out_planes, self.num_features)
                self.bn_pos = nn.BatchNorm1d(self.num_features)
                self.bn_neg = nn.BatchNorm1d(self.num_features)

                init.kaiming_normal_(self.embed_pos.weight, mode='fan_out')
                init.kaiming_normal_(self.embed_neg.weight, mode='fan_out')
                init.constant_(self.bn_pos.weight, 1)
                init.constant_(self.bn_neg.weight, 1)

                init.constant_(self.embed_pos.bias, 0)
                init.constant_(self.embed_neg.bias, 0)
                init.constant_(self.bn_pos.bias, 0)
                init.constant_(self.bn_neg.bias, 0)
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

        pos = self.base.features.transition3(self.base.features.denseblock3(x3))
        pos = F.relu(self.base.features.denseblock4(pos))
        pos = F.relu(pos, inplace=True)
        pos = F.avg_pool2d(pos, pos.size()[2:])
        pos = pos.view(pos.size(0), -1)
        feature = self.embed_pos(pos)#num_feature

        if self.training:
            neg = self.neg_transition3(self.neg_denseblock3(x3))
            neg = F.relu(self.neg_denseblock4(neg))
            neg = F.relu(neg, inplace=True)
            neg = F.avg_pool2d(neg, neg.size()[2:])
            neg = neg.view(neg.size(0), -1)
            f_neg = self.embed_neg(neg)#num_feature

            f_pos = self.bn_pos(feature)
            f_pos = F.relu(f_pos, inplace=True)
            y_pos = self.classifier(f_pos)#num_classes

            f_neg = self.bn_neg(f_neg)
            f_neg = F.relu(f_neg, inplace=True)
            y_neg = self.classifier(f_neg)#num_classes

            return y_pos, y_neg
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


def densenet169_sif(**kwargs):
    return DenseNet(169, **kwargs)


def densenet201_sif(**kwargs):
    return DenseNet(201, **kwargs)


def densenet161_sif(**kwargs):
    return DenseNet(161, **kwargs)


def densenet121_sif(**kwargs):
    return DenseNet(121, **kwargs)
