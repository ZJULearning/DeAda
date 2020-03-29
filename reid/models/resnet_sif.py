from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import copy

__all__ = ['ResNet', 'resnet50_sif', 'resnet101_sif', 'resnet152_sif']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,reset_acell=True):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)
        self.neg_layer3 = copy.deepcopy(self.base.layer3)
        self.neg_layer4 = copy.deepcopy(self.base.layer4)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

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
            self.reset_params()

    def forward(self, x):
        x1 = self.base.maxpool(self.base.relu(self.base.bn1(self.base.conv1(x))))
        x2 = self.base.layer1(x1)
        x3 = self.base.layer2(x2)

        pos = self.base.layer3(x3)
        pos = self.base.layer4(pos)
        pos = F.relu(pos)
        pos = F.avg_pool2d(pos, pos.size()[2:])
        pos = pos.view(pos.size(0), -1)

        feature = self.embed_pos(pos)

        if self.training:
            neg = self.neg_layer3(x3)
            neg = self.neg_layer4(neg)
            neg = F.relu(neg)
            neg = F.avg_pool2d(neg, neg.size()[2:])
            neg = neg.view(neg.size(0), -1)
            f_neg = self.embed_neg(neg)#num_feature

            f_pos = self.bn_pos(feature)
            f_pos = F.relu(f_pos)
            y_pos = self.classifier(f_pos)#num_classes

            f_neg = self.bn_neg(f_neg)
            f_neg = F.relu(f_neg)
            y_neg = self.classifier(f_neg)#num_classes

            return y_pos, y_neg
        return feature

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet50_sif(**kwargs):
    return ResNet(50, **kwargs)


def resnet101_sif(**kwargs):
    return ResNet(101, **kwargs)


def resnet152_sif(**kwargs):
    return ResNet(152, **kwargs)
