from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
import scipy.io as scio
from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter

from .models.densenet import DenseNet

def extract_features(model, data_loader, print_freq=50, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, data_sample in enumerate(data_loader):
        imgs, fnames, pids = data_sample[:3]
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)

        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features, query=None, gallery=None, metric=None,save_feature=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    if save_feature is not None:
        scio.savemat('{}'.format(save_feature),{'query':x.numpy(),'gallery':y.numpy()})
        print('save:{}'.format(save_feature))
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist

def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 query_cids = None, gallery_cids = None,
                 cmc_topk=(1, 5, 10), prec_topk = (1, 5, 10, 20,50), filter_cloth = False):
    if query is not None and gallery is not None:
        if filter_cloth:
            _, query_ids, query_dids,query_clothids,query_cams = zip(*query)
            _, gallery_ids, gallery_dids,gallery_clothids, gallery_cams = zip(*gallery)
        else:
            # To enable unpacking when cloth id is also returned
            query_ids = map(lambda x: x[1], query)
            gallery_ids = map(lambda x: x[1], gallery)
            query_cams = map(lambda x: x[2], query)
            gallery_cams = map(lambda x: x[2], gallery)
            query_dids = None
            gallery_dids = None
            query_clothids =None
            gallery_clothids=None
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    if filter_cloth is True:
        mAP, topks = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams,
                             query_clothids, gallery_clothids, topk=50, filter_cloth=filter_cloth)
    else:
        mAP, topks = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams,
                             query_clothids, gallery_clothids, topk = 50, filter_cloth = filter_cloth)
    print('Mean AP: {:4.1%}'.format(mAP))

    prec_str = ''
    for k in prec_topk:
        prec_str = prec_str + 'Precision@{:>2}'.format(k) + '\t'
    print(prec_str)

    prec_str = ''
    for k in prec_topk:
        prec_str = prec_str + '{:>12.1%}'.format(topks[k - 1]) + '\t'
    print(prec_str)
    print('\n')
    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams,
                            query_clothids, gallery_clothids,
                            filter_cloth = filter_cloth, **params)
                  for name, params in cmc_configs.items()}
    print('CMC Scores{:>12}{:>12}'.format('allshots', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['allshots'][0]

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None, filter_cloth = False):
        """
        filter_cloth: if True, also evaluate on the results that filter out the same clothes
        """
        features, _ = extract_features(self.model, data_loader)
        distmat = pairwise_distance(features, query, gallery, metric=metric)

        if filter_cloth:
            print('=== Metrics after filtering out begin ===')
            evaluate_all(distmat, query = query, gallery = gallery, filter_cloth = filter_cloth)
            print('=== Metrics after filtering out end ===\n\n')
        top1 = evaluate_all(distmat, query=query, gallery=gallery)
        return top1
