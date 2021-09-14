r'''
Author       : PiKaChu_wcg
Date         : 2021-09-13 16:08:55
LastEditors  : PiKachu_wcg
LastEditTime : 2021-09-14 10:26:07
FilePath     : \faster_rcnn\module\classifier.py
'''
import torch
import torch.nn as nn
from torchvision.ops import RoIPool


class RoIHead(nn.Module):
    def __init__(self,input_channel, n_class, roi_size, spatial_scale, classifier):
        super(RoIHead,self).__init__()
        self.classifier=classifier
        self.cls_loc = nn.Linear(input_channel, n_class*4)# every anchor predict every class loc num_anchor \times num_classes
        self.score = nn.Linear(input_channel, n_class)
        self.roi = RoIPool((roi_size, roi_size), spatial_scale)
    def forward(self,x,rois,roi_indices,img_size):
        n,_,_,_=x.shape
        if x.is_cuda:
            roi_indices=roi_indices.cuda()
            rois=rois.cuda()
        rois_feature_map=torch.zeros_like(rois)
        '''
        roi is the absolute Coordinate in image
        we need the Coordinate in feature map
        '''
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]
        indices_and_rois = torch.cat(
            [roi_indices[:, None], rois_feature_map], dim=1)
            
        pool =self.roi(x,indices_and_rois)
        fc=self.classifier(pool)
        fc=fc.view(fc.size(0),-1)
        roi_cls_locs=self.cls_loc(fc)
        roi_cls_locs=roi_cls_locs.view(n,-1,roi_cls_locs.size(1))
        roi_scores=self.score(fc)
        roi_scores=roi_scores.view(n,-1,roi_scores.size(1))
        return roi_cls_locs,roi_scores
