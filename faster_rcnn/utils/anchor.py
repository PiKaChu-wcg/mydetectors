r'''
Author       : PiKaChu_wcg
Date         : 2021-09-13 10:05:38
LastEditors  : PiKachu_wcg
LastEditTime : 2021-09-13 21:26:02
FilePath     : \detectors\faster-rcnn\utils\anchor.py
'''

import numpy as np


def generate_anchor_base(base_size=16,ratios=[0.5,1,2],anchor_scales=[8,16,32]):
    """use the ratios and size to generate the anchor

    Args:
        base_size (int, optional): the base size. Defaults to 16.
        ratios (list, optional): the ratios of the width and height. Defaults to [0.5,1,2].
        anchor_scales (list, optional): the scales of the base size. Defaults to [8,16,32].

    Returns:
        array: the anchor_base
    """
    anchor_base = np.zeros(
        (len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size*anchor_scales[j]*np.sqrt(ratios[i])
            w = base_size*anchor_scales[j]*np.sqrt(1./ratios[i])
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # 计算网格中心点
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(),
                      shift_x.ravel(), shift_y.ravel(),), axis=1)

    # 每个网格点上的9个先验框
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
        shift.reshape((K, 1, 4))
    # 所有的先验框
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor
