import warnings

from .triangnet import TriangNet
from ..builder import POSENETS
from ..utils import set_requires_grad

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class EmTriangNet(TriangNet):
    """
    a top-down 3d pose detectors based on TriangNet with a keypoint_head and a score head
    difference from TriangNet:
        update the score head and keypoint head iteratively
    """

    def __init__(self,
                 backbone,
                 keypoint_head,
                 triangulate_head=None,
                 score_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, keypoint_head, triangulate_head, score_head, train_cfg, test_cfg, pretrained)
        self.heatmap_steps = 2  # the iter steps to optimize the backbone and heatmap
        self.score_steps = 2  # the iter steps to update the score_head

    def train_step(self, data_batch, optimizer, **kwargs):
        """

        Args:
            data_batch: (dict): The output of dataloader
            optimizer: (dict): a dict of optimizers for keypoint_head+backbone and score_head
            **kwargs:
        Returns:
        """

        # fix the score_head and update keypoint_head and backbone
        set_requires_grad(self.score_head, False)
        set_requires_grad(self.keypoint_head, True)
        set_requires_grad(self.backbone, True)
        for _ in range(self.heatmap_steps):
            optimizer['backbone'].zero_grad()
            optimizer['keypoint_head'].zero_grad()
            losses = self.forward(**data_batch)
            loss, _ = self._parse_losses(losses)
            loss.backward()
            optimizer['keypoint_head'].step()
            optimizer['backbone'].step()

        # fix the keypoint_head and backbone, update score_head
        set_requires_grad(self.score_head, True)
        set_requires_grad(self.keypoint_head, False)
        set_requires_grad(self.backbone, False)
        for _ in range(self.score_steps):
            optimizer['score_head'].zero_grad()
            losses = self.forward(**data_batch)
            loss, log_vars = self._parse_losses(losses)
            loss.backward()
            optimizer['score_head'].step()
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))
        return outputs
