# modified the conv layers in the TopdownHeatmapConvHead
import torch
import torch.nn as nn
from icecream import ic
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)

from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead
from ..builder import HEADS


@HEADS.register_module()
class TopdownFeatureHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,

                 num_conv_layers=3,
                 num_conv_filters=(64, 128, 256),
                 num_conv_kernels=(3, 3, 3),
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 loss_feature=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels

        self.loss = build_loss(loss_feature)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        # self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        conv_layers = []
        conv_stride = 2 if num_deconv_layers > 0 else 1
        for i in range(num_conv_layers):
            # planes = num_conv_filters[i],
            # ic(num_conv_filters)
            # ic(self.in_channels, planes[0])
            conv_layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=self.in_channels,
                    out_channels=num_conv_filters[i],
                    kernel_size=num_conv_kernels[i],
                    stride=conv_stride,
                    padding=(num_conv_kernels[i] - 1) // 2)
            )

            conv_layers.append(
                build_norm_layer(dict(type='BN'), num_conv_filters[i])[1])
            conv_layers.append(nn.ReLU(inplace=True))
            self.in_channels = num_conv_filters[i]
            if len(conv_layers) > 1:
                self.conv_layers = nn.Sequential(*conv_layers)
            else:
                self.conv_layers = nn.Identity()

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        final_layers = [
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=self.in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0)]
        self.final_layer = nn.Sequential(*final_layers)

        self.fc = nn.Linear(64 * 64, 128)

    def get_loss(self, output, labels=None):
        """
        Calculate the supervised contrastive loss for joint features
        Args:
            output: the output of forward, features in shape [ N, num_joints, feature_dim]
            labels: the labels for joint that should have same features [num_joints]
        Returns:

        """
        features = output.permute(1, 0, 2)  # -> [num_joint, N, dim]
        losses = dict()
        losses['sup_con_loss'] = 0.0001 * self.loss(features, labels)  # [bsz, n_views, ...]
        return losses

    def forward(self, x, heatmap):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.conv_layers(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        features = x * heatmap  # [N, num_joints, 64, 64]
        N, num_joints, height, width = features.shape
        features_flatten = features.view([N * num_joints, -1])
        features = self.fc(features_flatten).reshape([N, num_joints, -1])
        return features

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize model weights."""
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.fc.modules():
            normal_init(m, std=0.001, bias=0)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding
