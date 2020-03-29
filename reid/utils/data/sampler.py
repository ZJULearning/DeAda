from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)

class RandomIdentitySamplerCloth(Sampler):
    def __init__(self, data_source, num_instances = 1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.cid_dic = defaultdict(set)
        self.index_dic = defaultdict(list)

        for index, (_, pid, _, cid) in enumerate(data_source):
            self.cid_dic[pid].add(cid)
            self.index_dic[(pid, cid)].append(index)

        self.pids = [pid for pid in self.cid_dic.keys() if len(self.cid_dic[pid]) > 1]
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances 

    def __iter__(self):
        indices = torch.randperm(self.num_samples) 
        ret = []
        for i in indices:
            pid = self.pids[i]
            if len(self.cid_dic[pid]) >= self.num_instances:
                cids = np.random.choice(list(self.cid_dic[pid]), size = self.num_instances, replace = False)
            else:
                cids = np.random.choice(list(self.cid_dic[pid]), size = self.num_instances, replace = True)
            for cid in cids:
                this_t = np.random.choice(self.index_dic[(pid, cid)], size = 1)
                ret.append(this_t[0])
        return iter(ret)
