r'''
Author       : PiKaChu_wcg
Date         : 2021-09-13 09:38:14
LastEditors  : PiKachu_wcg
LastEditTime : 2021-09-14 10:32:11
FilePath     : \faster_rcnn\module\faster-rcnn.py
'''



from RPN import RegionProposalNet
import torch.nn as nn
class FasterRCNN(nn.Module):
    def __init__(self,backbone,rpn,head):
        """what we need are the backbone, rpn and head
        
        Args:
            backbone (nn.Mudole): we can choice the resnet50 or VGG as the backbone and other backbone is all right
            rpn (nn.Mudole): the rpn 
            head (nn.Mudole): the detect head, with different backbone, we may choice different head
        """
        super(FasterRCNN,self).__init__()
        self.extrator=backbone
        self.rpn=rpn
        self.head=head
    def forward(self,x,scale=1.):
        img_size=x.shape[2:]
        base_feature=self.extrator(x)
        _, _, rois, roi_indices, _ = self.rpn(base_feature, img_size, scale)
        roi_cls_locs, roi_scores = self.head(
            base_feature, rois, roi_indices, img_size)
        return roi_cls_locs, roi_scores, roi_indices


if __name__ == "__main__":
    from module.RPN import RegionProposalNet
    import copy
    import numpy as np
    import timm
    import torch
    from PIL import Image
    from module.classifier import RoIHead
    model = timm.create_model("resnet50")
    backbone = nn.Sequential(*list(model.children())[:-3])
    classifier = nn.Sequential(*list(model.children())[-3:-1])
    img = Image.open("test/street.jpg")
    image = img.convert("RGB")
    image_shape = np.array(np.shape(image)[0:2])
    old_width, old_height = image_shape[1], image_shape[0]
    old_image = copy.deepcopy(image)
    backbone.eval()
    photo = np.transpose(np.array(image, dtype=np.float32)/255, (2, 0, 1))
    with torch.no_grad():
        image = torch.from_numpy(np.asarray([photo]))
        feature_map = backbone(image)
    rpn = RegionProposalNet(
        1024, 512,
    )
    t = rpn(feature_map, photo.shape[1:])
    head = RoIHead(2048, 10, 14, 1, classifier)
    head(feature_map, t[2], t[3], photo.shape[1:])

