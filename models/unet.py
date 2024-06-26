from itertools import chain  # 串联多个迭代对象

import torch.nn as nn

from .util import _BNReluConv, upsample


class UNet(nn.Module):
    def __init__(self, backbone, num_classes, use_bn=True):
        super(UNet, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)

    def forward(self, rgb_inputs, depth_inputs=None):
        x = self.backbone(rgb_inputs, depth_inputs)
        logits = self.logits.forward(x)
        logits_upsample = upsample(logits, rgb_inputs.shape[2:])
        return logits_upsample

    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))
