# input multi-view dataset and triangulation and output 3d dataset


import warnings

import torch

from .base import BasePose
from .. import builder
from ..builder import POSENETS

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class TriangNet(BasePose):
    """
    top-down 3d pose detectors with keypoint_head and triangulate_head
        Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature, to get the heatmap
        triangulate_head (dict): Feature head to further extract features.

        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained backbone, without head parts.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self,
                 backbone,
                 keypoint_head,
                 triangulate_head,
                 score_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.fp16_enabled = False
        self.backbone = builder.build_backbone(backbone)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # init the keypoint_head
        if keypoint_head is not None:
            keypoint_head['train_cfg'] = train_cfg
            keypoint_head['test_cfg'] = test_cfg
            self.keypoint_head = builder.build_head(keypoint_head)

        # init the score_head
        if score_head is not None:
            score_head['train_cfg'] = train_cfg
            score_head['test_cfg'] = test_cfg
            self.score_head = builder.build_head(score_head)

        # init the triangulate_head
        if triangulate_head is not None:
            self.triangulate_head = builder.build_head(triangulate_head)

        self.pretrained = pretrained
        self.init_weights()

    @property
    def with_keypoint_head(self):
        return hasattr(self, 'keypoint_head')

    @property
    def with_score_head(self):
        return hasattr(self, 'score_head')

    @property
    def with_triangulate_head(self):
        return hasattr(self, 'triangulate_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        self.backbone.init_weights(self.pretrained)
        if self.with_keypoint_head:
            self.keypoint_head.init_weights()

    @auto_fp16(apply_to=('img',))
    def forward(self,
                img,
                img_metas=None,
                proj_matrices=None,
                target=None,
                target_weight=None,
                kpt_3d_gt=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
                        return_loss=True. Note this setting will change the expected inputs.
                        When `return_loss=True`, img and img_meta are single-nested (i.e.
                        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
                        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
                        the outer list indicating test time augmentations.
                    Args:
                        img: Input images, tensor [bs, num_cameras, 3, h_img, w_img]
                        proj_matrices: tensor [bs, num_cameras, 3, 4]
                        target (torch.Tensor[NxKxHxW]): Target heatmaps.
                        target_weight (torch.Tensor[NxKx1]): Weights across
                            different joint types.
                        kpt_3d_gt (tensor [bs, num_joints, 3]): the ground-truth 3d keypoint coordinates.
                        img_metas (list(dict)): Information about data augmentation
                        return_loss (bool): Option to `return loss`. `return loss=True`
                            for training, `return loss=False` for validation & test.
                        return_heatmap (bool) : Option to return heatmap.
                    Returns:
                        dict|tuple: if `return loss` is true, then return losses. \
                            Otherwise, return predicted poses, boxes, image paths \
                            and heatmaps.
        """
        if return_loss:
            return self.forward_train(img,
                                      proj_matrices,
                                      img_metas,
                                      target,
                                      target_weight,
                                      **kwargs
                                      )
        else:
            return self.forward_test(
                img,
                proj_matrices,
                img_metas,
                return_heatmap,
                **kwargs)

    def forward_train(self, img, proj_matrices, img_metas, target,
                      target_weight, kpt_3d_gt, **kwargs):
        """Defines the computation performed at every call when training.
                img: input image [bs, num_cams, num_channel, h_img, w_img]
                proj_matrices: project matrices [bs, num_cams, 3, 4]
                target: the ground-truth 2d keypoint heatmap, [bs, num_cams, h_map, w_map]
                target_weight: Weights across different joint types. [N, num_joints, 3]
                kpt_3d_gt: the ground-truth 3d keypoint coordinates, [bs, num_joints, 3]
                """
        [bs, num_cams, num_channel, h_img, w_img] = img.shape
        h_map, w_map = target.shape[-2], target.shape[-1]
        img = img.reshape(-1, *img.shape[2:])
        target = target.reshape(-1, *target.shape[2:])
        target_weight = target_weight.reshape(-1, *target_weight.shape[2:]),

        hidden_features = self.backbone(img)
        if self.with_keypoint_head:
            heatmap = self.keypoint_head(hidden_features)
        if self.with_score_head:
            scores = self.score_head(hidden_features)
        else:
            scores = torch.ones(*img.shape[:2]).type(torch.float)

        if self.with_triangulate_head:
            kpt_3d_pred, res_triang, kp_2d_croped, kp_2d_heatmap = self.triangulate_head(heatmap, proj_matrices)

        # if return loss
        losses = dict()
        if self.with_keypoint_head and \
                target is not None and \
                self.train_cfg.get('supervised_2d', True):
            keypoint2d_losses = self.keypoint_head.get_loss(heatmap, target, target_weight)
            losses.update(keypoint2d_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(heatmap, target, target_weight)
            losses.update(keypoint_accuracy)

        if self.with_triangulate_head and kpt_3d_gt is not None and self.train_cfg.get('supervised_3d', True):
            sup_3d_loss = self.triangulate_head.get_sup_loss(kpt_3d_pred, kpt_3d_gt, target_weight)
            losses.update(sup_3d_loss)
        return losses

    def forward_test(self, img, proj_matrices, img_metas, return_heatmap, **kwargs):
        """Defines the computation performed at every call when testing"""
        [bs, num_cams, num_channel, h_img, w_img] = img.shape
        img = img.reshape(-1, *img.shape[2:])
        h_img, w_img = img.shape[-2], img.shape[-1]
        result = {}
        features = self.backbone(img)
        if self.with_keypoint_head:
            heatmap = self.keypoint_head(features)  # for triangulate-head input
            output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)
            if self.with_keypoint_head:
                output_flipped_heatmap = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                output_heatmap = (output_heatmap + output_flipped_heatmap)
                if self.test_cfg.get('regression_flip_shift', False):
                    output_heatmap[..., 0] -= 1.0 / w_img
                output_heatmap = output_heatmap / 2

        if self.with_keypoint_head:
            if not return_heatmap:
                output_heatmap = None
            result['output_heatmap'] = output_heatmap

        if self.with_triangulate_head:
            kp_3d, res_triang, kp_2d_preds, _ = self.triangulate_head(heatmap, proj_matrices)
            result['preds'] = kp_3d.detach().cpu().numpy()
            result['kp_2d_preds'] = kp_2d_preds.detach().cpu().numpy()
            result['res_triang'] = res_triang.detach().cpu().numpy()

        if img_metas is not None:
            result['img_metas'] = img_metas.data[0]  # using the img_metas to get the 3d global ground truth
        return result

    def show_result(self,
                    imgs,
                    img_metas,
                    result,
                    skeleton=None,
                    visualize_2d=False,
                    dataset_info=None,
                    wait_time=0,
                    out_file=None,
                    show=False,
                    **kwargs):
        """
        Args:
            imgs: tensor or array, the image to visualize 2D keypoints, [bs, num_cams, 3, 256, 256]
            result:
            skeleton:
            visualize_2d:
            dataset_info:
            wait_time:
            out_file:
            show:
            **kwargs:

        Returns:

        """

        pose_3d = result['preds']  # [bs, num_joints, 3]
        bs = imgs.shape[0]
        num_cameras = imgs.shape[1]
        for i in range(bs):
            pose_3d_i = pose_3d[i]
            pose_3d_list = [pose_3d_i]
