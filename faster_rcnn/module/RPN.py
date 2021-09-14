r'''
Author       : PiKaChu_wcg
Date         : 2021-09-13 09:41:30
LastEditors  : PiKachu_wcg
LastEditTime : 2021-09-14 10:00:22
FilePath     : \detectors\faster_rcnn\module\RPN.py
'''

import torch.nn as nn
import torch
from utils.anchor import _enumerate_shifted_anchor,generate_anchor_base
from utils.utils import loc2bbox
from torchvision.ops import nms
import numpy as np
import torch.nn.functional as F

class ProposalCreator():
    def __init__(self,
        mode,
        nms_thresh=0.7,
        n_train_pre_nms=12000,
        n_train_post_nms=600,
        n_test_pre_nms=3000,
        n_test_post_nms=300,
        min_size=16
    ):
        self.mode=mode
        self.nms_thresh=nms_thresh
        self.n_train_pre_nms=n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc,score,anchor,img_size,scale=1.) :
        """we have the loc and the scores of the locs
        the loc is the relative loc from the anchor 
        so we transform it to the absolute bbox ,which need img_size and scale

        Args:
            loc (tensor): the relative loc from anchor to the bbox 
            score (tensor): the score of the locs
            anchor (tensor):the anchors
            img_size (int): img_size
            scale (float, optional):thr proportion of the img shrink. Defaults to 1..

        Returns:
            [tensor]: the rois
        """

        '''
        initialize the nms
        '''
        if self.mode=="training":
            n_pre_nms=self.n_train_pre_nms
            n_post_nms=self.n_train_post_nms
        else:
            n_pre_nms=self.n_test_pre_nms
            n_post_nms=self.n_train_post_nms
        '''
        from the anchor to the loc
        '''
        anchor=torch.from_numpy(anchor)
        if loc.is_cuda:
            anchor=anchor.cuda()
        roi=loc2bbox(anchor,loc)
        '''
        Prevent the proposal box from going beyond the edge of the image
        '''
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])
        '''
        remove the box who is smaller than the min_size
        '''
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size)
                           & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]
        '''
        keep the top n_pre_nms score bbonx
        '''
        order=torch.argsort(score,descending=True)
        if n_pre_nms>0:
            order=order[:n_pre_nms]
        roi=roi[order,:]
        score=score[order]
        '''
        nms
        '''
        keep=nms(roi,score,self.nms_thresh)
        roi=roi[keep]
        '''
        keep the top n_post_nms score bbonx
        '''
        order = torch.argsort(score, descending=True)
        if n_post_nms > 0:
            order=order[:n_post_nms]
        roi=roi[order,:]
        score=score[order]
        return roi


class RegionProposalNet(nn.Module):
    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            mode="training",anchor_base=None,
    ):
        super(RegionProposalNet, self).__init__()
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(mode)
        self.anchor_base = anchor_base if anchor_base else generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)


    def forward(self, x, img_size, scale=1.):
        """input the feature map

        Args:
            x (tensor): the feature map
            img_size (int): the size of the image
            scale (float, optional): [description]. Defaults to 1..

        Returns:
            rpn_locs: the loc of the rp
            rpn_scores : the score of the rp
            rois : the feature of the region 
            roi_indices : the index of the rois batch
            anchor : the anchor
        """
        n, _, h, w = x.shape
        x = F.relu(self.conv1(x))
        rpn_locs = self.loc(x)# one grid have 9 anchor 
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base), self.feat_stride, h, w)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi)
            roi_indices.append(batch_index)
        rois = torch.cat(rois, dim=0)
        roi_indices = torch.cat(roi_indices, dim=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor
