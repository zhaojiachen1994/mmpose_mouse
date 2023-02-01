# modified from top_down model

import mmcv
import numpy as np
import torch
import warnings
from icecream import ic
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from mmpose.core import imshow_bboxes, imshow_keypoints
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
class MouseNet_2d(BasePose):
    """Top-down pose detectors.

        Args:
            backbone (dict): Backbone modules to extract feature.
            keypoint_head (dict): Keypoint head to process feature.
            feature_head (dict): Feature head to further extract features.
            train_cfg (dict): Config for training. Default: None.
            test_cfg (dict): Config for testing. Default: None.
            pretrained (str): Path to the pretrained models.
            loss_pose (None): Deprecated arguments. Please use
                `loss_keypoint` for heads instead.
        """

    def __init__(self,
                 backbone,
                 keypoint_head,
                 feature_head=None,
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

        # init the feature_head
        if feature_head is not None:
            feature_head['train_cfg'] = train_cfg
            feature_head['test_cfg'] = test_cfg
            self.feature_head = builder.build_head(feature_head)

        self.pretrained = pretrained
        self.init_weights()

    @property
    def with_keypoint_head(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    @property
    def with_feature_head(self):
        """check if has feature_head"""
        return hasattr(self, 'feature_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        self.backbone.init_weights(self.pretrained)
        if self.with_keypoint_head:
            self.keypoint_head.init_weights()
        if self.with_feature_head:
            self.feature_head.init_weights()

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_metas,
                target=None,
                target_weight=None,
                return_loss=True, **kwargs):
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
                                      img_metas,
                                      target,
                                      target_weight,
                                      **kwargs)
        else:
            return self.forward_test(
                img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, img_metas, **kwargs):
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
        if self.with_feature_head:
            features = self.feature_head(hidden_features, heatmap)

        # if return loss
        losses = dict()
        if self.with_keypoint_head:
            keypoint2d_losses = self.keypoint_head.get_loss(
                heatmap, target, target_weight)
            losses.update(keypoint2d_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                heatmap, target, target_weight)
            losses.update(keypoint_accuracy)

        if self.with_feature_head and self.train_cfg.get('contrastive_feature', True):
            sup_con_loss = self.feature_head.get_loss(features, labels)
            losses.update(sup_con_loss)

        return losses

    def forward_test(self, img, img_metas,
                     return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing.
        img: [bs, num_cams, num_channel, h_img, w_img]
        """
        assert img.size(0) == len(img_metas)
        img = img.reshape(-1, *img.shape[2:])
        batch_size, _, img_height, img_width = img.shape

        result = {}
        features = self.backbone(img)
        if self.with_keypoint_head:
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
                    output_heatmap[..., 0] -= 1.0 / img_width
                output_heatmap = output_heatmap / 2

        if self.with_keypoint_head:
            keypoint_result = self.keypoint_head.decode(
                img_metas, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)

            if not return_heatmap:
                output_heatmap = None
            result['output_heatmap'] = output_heatmap

        return result

    def foraward_dummy(self, img):
        """Used for computing network FLOPs.
                See ``tools/get_flops.py``.
                Args:
                    img (torch.Tensor): Input image.
                Returns:
                    Tensor: Output heatmaps.
        """
        hidden_features = self.backbone(img)
        if self.with_keypoint_head:
            heatmap = self.keypoint_head(hidden_features)
        if self.with_feature_head:
            output = self.feature_head(hidden_features, heatmap)
        return heatmap, output

    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes(
                img,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False)

        if pose_result:
            imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
