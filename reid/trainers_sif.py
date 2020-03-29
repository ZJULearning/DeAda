from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter
from .models.densenet import DenseNet
import torch.nn.functional as F


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=50):
        self.model.train()

        self.alpha1 = 0.3
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets, i)
            bs = targets.size(0) if not isinstance(targets, tuple) else targets[0].size(0)
            losses.update(loss.item(), bs)
            precisions.update(prec1, bs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
        #return uni_step,clist

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError

class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [imgs]
        targets = pids.cuda()
        return inputs, targets

    def _CELoss(self, p1, p2):
        h = - p1 * torch.log(p2)
        h = torch.mean(h.sum(dim=1))
        return h

    def _forward(self, inputs, targets, i):
        y_pos, y_neg = self.model(*inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss_pos = self.criterion(y_pos, targets)
            bsize = y_pos.size(0)
            y_pos = F.softmax(y_pos, dim=1)
            y_pos_renormal = y_pos.clone()
            mask_pos = torch.ones(y_pos_renormal.size()).cuda()
            for j in range(bsize):
                mask_pos[j,targets[j]] = 0.0
            y_pos_renormal = y_pos_renormal * mask_pos
            y_pos_renormal = y_pos_renormal / y_pos_renormal.sum(dim=1, keepdim=True)
            y_neg = F.softmax(y_neg, dim=1)
            y_neg_truncate = y_neg.clone()
            thre = 1e-8
            mask = y_neg_truncate < thre
            y_neg_truncate[mask] = thre
            y_neg_truncate = y_neg_truncate / y_neg_truncate.sum(dim=1, keepdim=True)
            loss_res = self._CELoss(y_pos_renormal, y_neg_truncate)
            loss = loss_pos \
                + self.alpha1 * loss_res 
            if (i + 1) % 50 == 0:
                print('loss positive: {:.6f}\t'
                      'loss residual: {:.6f}\t'
                      .format(loss_pos.item(), loss_res.item()))
            prec, = accuracy(y_pos, targets.data)
            prec = prec[0] 
        return loss, prec
