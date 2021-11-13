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


class BCEWithLogitsLoss(_Loss):
    """
    EQV1 Softmax Loss
    """
    def __init__(self, weight=None, reduction='mean', pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__()
        # self.freq_info = freq_info
        # with open(freq_path, 'r') as fd:
        #     freq = json.load(fd)
        # self.freq = torch.tensor(freq)
        print('init BCEWithLogitsLoss')

        # criterion = torch.nn.BCEWithLogitsLoss(weight=weight, reduction=reduction, pos_weight=pos_weight)
    
    def forward(self, logit, label, weight=None, reduction='mean' ):

        weight = None
        pos_weight = None

        self.n_b, self.n_c = logit.size()
        self.gt_classes = label

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_b, self.n_c)
            target[torch.arange(self.n_b), gt_classes] = 1
            return target

        target = expand_label(logit, label)

        # pdb.set_trace()
        cls_loss =  F.binary_cross_entropy_with_logits(input=logit, target=target, weight=weight, pos_weight=pos_weight)

        # cls_loss = torch.sum(cls_loss) / self.n_b

        return cls_loss



def create_loss():
    print('Loading BCEWithLogitsLoss.')
    return BCEWithLogitsLoss()

