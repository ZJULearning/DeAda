from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import re
import os
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.nn.parameter import Parameter
import torchvision
import setproctitle
from reid import models
from reid.dist_metric import DistanceMetric
from reid.trainers import Trainer
from reid.evaluators import Evaluator,extract_features,pairwise_distance
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from torch.nn import init

def get_data(data_dir, height, width, batch_size, workers, combine_trainval, 
             train_list, val_list, query_list, gallery_list, dataset_type):
    root = data_dir
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = train_list + val_list if combine_trainval else train_list  # a list

    train_transformer = T.Compose([
        T.RectScale(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=root,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomSampler(train_set),
        pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(val_list, root=root,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(query_list) | set(gallery_list)),
                     root=root, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader

def main(args):
    setproctitle.setproctitle(args.project_name)
    logs_dir = osp.join(args.root_dir,'logs/',args.project_name)
    if osp.exists(logs_dir) is False:
        os.makedirs(logs_dir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    data_dir = osp.join(args.data_dir, args.dataset)
    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(logs_dir, 'log.txt'))
    print('{}'.format(vars(parser.parse_args())))

    # Create data loaders
    def readlist(path):
        lines=[]
        with open(path, 'r') as f:
            data = f.readlines()

        for line in data:
            name, pid, camid = line.split()
            lines.append((name, int(pid), int(camid)))
        return lines

    if osp.exists(osp.join(data_dir, 'train.txt')):
        train_list = readlist(osp.join(data_dir, 'train.txt'))
    else:
        print("The training list doesn't exist")

    if osp.exists(osp.join(data_dir, 'val.txt')):
        val_list = readlist(osp.join(data_dir, 'val.txt'))
    else:
        print("The validation list doesn't exist")

    if osp.exists(osp.join(data_dir, 'query.txt')):
        query_list = readlist(osp.join(data_dir, 'query.txt'))
    else:
        print("The query.txt doesn't exist")

    if osp.exists(osp.join(data_dir, 'gallery.txt')):
        gallery_list = readlist(osp.join(data_dir, 'gallery.txt'))
    else:
        print("The gallery.txt doesn't exist")

    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)

    train_loader, val_loader, test_loader = \
        get_data(data_dir, args.height,
                 args.width, args.batch_size, args.workers,
                 args.combine_trainval, train_list, val_list, query_list, gallery_list,dataset_type=args.dataset)
    # Create model
    num_classes = args.ncls
    model = models.create(args.arch, num_features=args.features,
                        dropout=args.dropout, num_classes=num_classes)

    cnt = 0
    for p in model.parameters():
        cnt += p.numel()
    print('Parameter number:{}\n'.format(cnt))
    # Load from checkpoint
    start_epoch = best_top1 = 0
    model = nn.DataParallel(model).cuda()
    #model = model.cuda()
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.module.load_state_dict(checkpoint['state_dict'])

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        with torch.no_grad():
            print('Test with latest model:')
            checkpoint = load_checkpoint(osp.join(logs_dir, 'checkpoint.pth.tar'))
            model.module.load_state_dict(checkpoint['state_dict'])
            print('best epoch: ', checkpoint['epoch'])
            metric.train(model, train_loader)
            evaluator.evaluate(test_loader, query_list, gallery_list, clist=clist, metric=metric)

            print('Test with best model:')
            checkpoint = load_checkpoint(osp.join(logs_dir, 'model_best.pth.tar'))
            model.module.load_state_dict(checkpoint['state_dict'])
            print('best epoch: ', checkpoint['epoch'])
            metric.train(model, train_loader)
            evaluator.evaluate(test_loader, query_list, gallery_list, clist=clist, metric=metric)
        return

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer
    if args.training_method == 'plain':
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                  id(p) not in base_param_ids]
        param_groups = [
                {'params': model.module.base.parameters(), 'lr_mult': 0.1},
                {'params': new_params, 'lr_mult': 1.0}]
    elif args.training_method == 'deada':
        param_class_ids = set(map(id, model.module.classifier.parameters()))
        param_extrac = [p for p in model.parameters() if
                        id(p) not in param_class_ids]
        param_groups = [
                {'params': param_extrac, 'lr_mult': 0.1},
                {'params': model.module.classifier.parameters(), 'lr': args.lr_classifier}]
    else:
        raise KeyError('Unknown training method: ', args.training_method)

    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    # Trainer
    trainer = Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch, args):
        step_size = args.step_size
        lr = args.lr if epoch <= step_size else \
             args.lr * (0.1 ** ((epoch - step_size) // step_size + 1))
        if args.training_method == 'plain':
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)
        elif args.training_method == 'deada':
            for g in optimizer.param_groups[:1]:
                # only update lr of feature extractor, keep lr of classifier constant
                g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    waits=0
    for epoch in range(start_epoch, args.epochs):
        print('Project Name:{}'.format(args.project_name))
        if waits >= args.patience:
            print('Patience is exceeded\n')
            break
        print('\nWaits: {}'.format(waits))
        adjust_lr(epoch, args)
        if args.training_method == 'deada':
            lr_extrac = optimizer.param_groups[0]['lr']
            lr_class = optimizer.param_groups[1]['lr']
            print('feature extractor lr: ', lr_extrac, ' classifier lr: ', lr_class)
            init.normal_(trainer.model.module.classifier.weight, std=0.001)
            init.constant_(trainer.model.module.classifier.bias, 0)

        trainer.train(epoch, train_loader, optimizer)
        if epoch < args.start_save:
            continue
        top1 = evaluator.evaluate(val_loader, val_list, val_list)
        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)

        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(logs_dir, 'checkpoint.pth.tar'))
        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))
        if (epoch+1) % 5 == 0:
            print('Test model: \n')
            model_name = 'epoch_'+ str(epoch) + '.pth.tar'
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
            }, False, fpath=osp.join(logs_dir, model_name))
        if is_best:
            waits=0
        else:
            waits+=1
    # Final test
    with torch.no_grad():
        print('Test with latest model:')
        checkpoint = load_checkpoint(osp.join(logs_dir, 'checkpoint.pth.tar'))
        model.module.load_state_dict(checkpoint['state_dict'])
        print('best epoch: ', checkpoint['epoch'])
        metric.train(model, train_loader)
        evaluator.evaluate(test_loader, query_list, gallery_list, metric=metric)

        print('Test with best model:')
        checkpoint = load_checkpoint(osp.join(logs_dir, 'model_best.pth.tar'))
        model.module.load_state_dict(checkpoint['state_dict'])
        print('best epoch: ', checkpoint['epoch'])
        metric.train(model, train_loader)
        evaluator.evaluate(test_loader, query_list, gallery_list, metric=metric)


if __name__ == '__main__':
    print('torch.version:{}'.format(torch.__version__))
    print('torchvision.version:{}'.format(torchvision.__version__))
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    parser.add_argument('--project_name', type=str, default='densnet')
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=48)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    # model
    parser.add_argument('-a', '--arch', type=str, default='densenet161',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--ncls', type=int, default=755)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--lr_classifier', type=float, default=0.2,
                        help="learning rate of classifier")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--step_size', type=int, default=10)
    # training configs
    parser.add_argument('--training_method', type=str, default='plain',
                        choices=['deada', 'plain'])
    parser.add_argument('--pretrained_model', type=str, default='', metavar='PATH')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=15)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--print-info', type=int, default=50)
    parser.add_argument('--patience', type=int, default=100)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    parser.add_argument('--root_dir', type=str, metavar='PATH',
                        default='./')
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='/home/data/')
    main(parser.parse_args(),)
