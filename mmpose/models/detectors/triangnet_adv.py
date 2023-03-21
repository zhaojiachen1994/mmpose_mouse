import warnings

import torch

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
class AdvTriangNet(TriangNet):
    """
    a top-down 3d pose detectors based on TriangNet with a keypoint_head and a score head
    difference from TriangNet:
        1. fix backbone and keypoint_head, update score_head to minimize the unsupervised 3d loss, and get the pseudo 3d label
        2. fix score_head, update backbone and keypoint_head to minimize the supervised 3d loss, using the 1-score to weight,
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
        self.heatmap_steps = 10  # the iter steps to optimize the backbone and heatmap
        self.score_steps = 10  # the iter steps to update the score_head

    def forward_train(self, img, proj_mat, img_metas, target,
                      target_weight, joints_4d, joints_4d_visible, train_state=None, **kwargs):
        """Defines the computation performed at every call when training.
                img: input image [bs, num_cams, num_channel, h_img, w_img]
                proj_mat: project matrices [bs, num_cams, 3, 4]
                target: the ground-truth 2d keypoint heatmap, [bs, num_cams, h_map, w_map]
                target_weight: Weights across different joint types. [N, num_joints, 3]
                joints_4d: the ground-truth 3d keypoint coordinates, [bs, num_joints, 3]
                """
        # [bs, num_cams, _, _, _] = img.shape
        img = img.reshape(-1, *img.shape[2:])
        hidden_features = self.backbone(img)
        heatmap = self.keypoint_head(hidden_features)
        scores = self.score_head(hidden_features)  # [bs*num_cams, num_joints]

        losses = dict()
        if train_state == "update_score":
            kpt_3d_pred, res_triang, _, _ = self.triangulate_head(heatmap, proj_mat, scores)
            unsup_3d_loss = self.triangulate_head.get_unSup_loss(res_triang)
            losses.update(unsup_3d_loss)
        elif train_state == "update_backbone":
            adv_scores = 1 - scores.detach()
            kpt_3d_pred, res_triang, _, _ = self.triangulate_head(heatmap, proj_mat, adv_scores)
            sup_3d_loss = self.triangulate_head.get_sup_loss(kpt_3d_pred, joints_4d, joints_4d_visible)
            losses.update(sup_3d_loss)
        return losses

    def train_step(self, data_batch, optimizer, **kwargs):

        # step1: update score_head
        set_requires_grad(self.score_head, True)
        set_requires_grad(self.keypoint_head, False)
        set_requires_grad(self.backbone, False)
        train_state = "update_score"
        for _ in range(self.score_steps):
            optimizer['score_head'].zero_grad()
            losses = self.forward_train(**data_batch, train_state=train_state)
            loss, log_vars = self._parse_losses(losses)
            loss.backward()
            optimizer['score_head'].step()
        total_log_vars = log_vars

        # step2: get the pseudo 3d label
        set_requires_grad(self.score_head, False)
        set_requires_grad(self.keypoint_head, False)
        set_requires_grad(self.backbone, False)
        pseudo_4d_label = self.forward_test(return_heatmap=False, **data_batch)['preds']
        data_batch['joints_4d'] = torch.tensor(pseudo_4d_label, device=data_batch['joints_4d'].device)
        data_batch['joints_4d_visible'] = torch.ones_like(data_batch['joints_4d_visible'],
                                                          device=data_batch['joints_4d_visible'].device)

        # step3: update backbone and keypoint_head
        set_requires_grad(self.score_head, False)
        set_requires_grad(self.keypoint_head, True)
        set_requires_grad(self.backbone, True)
        train_state = "update_backbone"
        for _ in range(self.heatmap_steps):
            optimizer['backbone'].zero_grad()
            optimizer['keypoint_head'].zero_grad()
            losses = self.forward_train(**data_batch, train_state=train_state)
            loss2, log_vars = self._parse_losses(losses)
            loss2.backward()
            optimizer['keypoint_head'].step()
            optimizer['backbone'].step()
        total_log_vars.update(log_vars)
        outputs = dict(
            loss=loss,
            log_vars=total_log_vars,
            num_samples=len(next(iter(data_batch.values()))))
        return outputs
