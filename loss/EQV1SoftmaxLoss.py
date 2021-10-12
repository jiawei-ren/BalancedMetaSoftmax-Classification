"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json
import pdb


class EQV1Softmax(_Loss):
    """
    EQV1 Softmax Loss
    """
    def __init__(self, freq_path, lambda_=5e-3, gamma=0.9):
        super(EQV1Softmax, self).__init__()
        # self.freq_info = freq_info
        self.lambda_ = lambda_
        self.gamma = gamma
        with open(freq_path, 'r') as fd:
            freq = json.load(fd)
        freq = torch.tensor(freq)
        # self.sample_per_class = freq
        self.freq_ratio = freq / freq.sum()
        # pdb.set_trace()

    def forward(self, logit, label, reduction='mean'):

        self.n_b, self.n_c = logit.size()
        self.gt_classes = label
        self.pred_class_logits = logit

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_b, self.n_c)
            target[torch.arange(self.n_b), gt_classes] = 1
            return target

        target = expand_label(logit, label)

        eql_w = 1 - self.beta() * self.threshold_func() * (1 - target)
        eql_w = torch.log(eql_w)

        logit = logit + eql_w

        # pdb.set_trace()
        


        loss = F.cross_entropy(input=logit, target=label, reduction=reduction)
        return loss

    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_ratio < self.lambda_] = 1
        weight = weight.view(1, self.n_c).expand(self.n_b, self.n_c)
        return weight
    
    def beta(self):
        return (torch.rand(self.n_b,self.n_c) < self.gamma).cuda()

def create_loss(freq_path):
    print('Loading EQV1 Softmax Loss.')
    return EQV1Softmax(freq_path)

