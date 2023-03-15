import numpy as np
import torch
import torch.nn as nn
from icecream import ic

from mmpose.models.builder import build_loss
from ..builder import HEADS


def euclidean_to_homogeneous(points):
    """Converts euclidean points to homogeneous

    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    elif torch.is_tensor(points):
        return torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=1)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean

    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


@HEADS.register_module()
class TriangulateHead(nn.Module):
    def __init__(self, num_cams=6, img_shape=[256, 256], heatmap_shape=[64, 64],
                 softmax_heatmap=True, loss_3d_sup=None, det_conf_thr=None,
                 train_cfg=None, test_cfg=None):
        """from the heatmap to 2d keypoints, then to 3d keypoints"""
        super().__init__()
        self.num_cams = num_cams
        [self.h_img, self.w_img] = img_shape
        [self.h_map, self.w_map] = heatmap_shape
        self.softmax = softmax_heatmap
        self.det_conf_thr = det_conf_thr  # weather use the 2d detect confidence to mask the fail detection points
        if loss_3d_sup is not None and train_cfg.get('use_3d_sup'):
            self.super_loss = build_loss(loss_3d_sup)

    def compute_kp_coords(self, heatmap):
        """
        compute keypoints from heatmaps for square_box-crop-resize pipeline, in the croped image size
        Args:
            heatmap: [bs*num_cams, num_joints, h_map, w_map]

        Returns:
            kp_2d_heatmap: keypoint coordinates in the heatmap size, [bs*num_cams, num_joints, 3]
            kp_2d_croped: keypoint coordinates in the croped image size, [bs*num_cams, num_joints, 3]
        """

        heatmap = heatmap * 100.0
        # compute the 2d keypoint coordinate in the heatmap size
        heatmap = heatmap.reshape((*heatmap.shape[:2], -1))

        if self.softmax:
            heatmap = nn.functional.softmax(heatmap, dim=2)
        else:
            heatmap = nn.functional.relu(heatmap)
        #     print("=============No==============")
        confidence = torch.amax(heatmap, 2) / torch.sum(heatmap, 2)
        confidence = torch.unsqueeze(confidence, -1)

        heatmap = heatmap.reshape((*heatmap.shape[:2], self.h_map, self.w_map))

        mass_x = heatmap.sum(dim=2)
        mass_y = heatmap.sum(dim=3)

        mass_times_coord_x = mass_x * torch.arange(self.w_map).type(torch.float).to(mass_x.device)
        mass_times_coord_y = mass_y * torch.arange(self.h_map).type(torch.float).to(mass_y.device)

        x = mass_times_coord_x.sum(dim=2, keepdim=True)
        y = mass_times_coord_y.sum(dim=2, keepdim=True)

        if not self.softmax:
            x = x / mass_x.sum(dim=2, keepdim=True)
            y = y / mass_y.sum(dim=2, keepdim=True)

        coordinates = torch.cat((x, y, confidence), dim=2)
        kp_2d_heatmap = coordinates.reshape((*heatmap.shape[:2], 3))  # [bs*num_cams, num_joints, 3]
        # upscale the 2d coordinates from heatmap size to image size
        kp_2d_croped = torch.zeros_like(kp_2d_heatmap, dtype=float)
        kp_2d_croped[:, :, 0] = kp_2d_heatmap[:, :, 0] * self.h_img / self.h_map
        kp_2d_croped[:, :, 1] = kp_2d_heatmap[:, :, 1] * self.w_img / self.w_map
        kp_2d_croped[:, :, 2] = kp_2d_heatmap[:, :, 2]
        # coordinates[confidence < 0.8] = torch.nan
        # ic(coordinates[confidence > 0.2])
        return kp_2d_heatmap, kp_2d_croped,

    def triangulate_point(self, proj_matrices, points, confidences=None):
        # modified from learnable triangulation/multiview
        """
        triangulate one joint in multi-views, in pytorch
        Args:
            proj_matrices: torch tensor in shape (n_cams, 3, 4), sequence of projection matricies (3x4)
            points: torch tensor in shape (N, 2), sequence of points' coordinates
            confidences: None or torch tensor of shape (N,), confidences of points [0.0, 1.0].
                                                        If None, all confidences are supposed to be 1.0

        Returns:
            point_3d numpy torch tensor of shape (3,): triangulated point
        """
        assert len(proj_matrices) == len(points)

        n_views = len(proj_matrices)

        if confidences is None:
            confidences = torch.ones(n_views, dtype=torch.float32, device=points.device)


        A = proj_matrices[:, 2:3].expand(n_views, 2, 4) * points.view(n_views, 2, 1)
        A -= proj_matrices[:, :2]
        A *= confidences.view(-1, 1, 1)

        u, s, vh = torch.svd(A.view(-1, 4))

        point_3d_homo = -vh[:, 3]
        point_3d = homogeneous_to_euclidean(point_3d_homo.unsqueeze(0))[0]

        # compute triangulation residual
        res_triang = torch.linalg.vector_norm(A @ point_3d_homo.unsqueeze(1), ord=1)

        return point_3d, res_triang

    def forward(self, heatmap, proj_matrices=None, confidences=None):
        """
        Args:
            heatmap: [bs*num_cams, num_joints, h_heatmap, w_heatmap]
            proj_matrices: [bs, num_cams, 3, 4]
            confidences: [bs*num_cams, num_joints]

        Returns:
            kp_3d: triangulation results, keypoint 3d coordinates, [bs, n_joints, 3]
            res_triang: triangulation residual, [bs, n_joints]
        """
        kp_2d_heatmap, kp_2d_croped = self.compute_kp_coords(
            heatmap)  # [bs*num_cams, num_joints, 3] with [x, y, confidence]
        kp_2d_croped = kp_2d_croped.reshape(
            [-1, self.num_cams, *kp_2d_croped.shape[1:]])  # [bs, num_cams, num_joints, 2]

        kp_2d_heatmap = kp_2d_heatmap.reshape([-1, self.num_cams, *kp_2d_heatmap.shape[1:]])
        batch_size, n_cams, n_joints = kp_2d_croped.shape[:3]

        kp_3d = torch.zeros(batch_size, n_joints, 3, dtype=torch.float32, device=kp_2d_croped.device)
        res_triang = torch.zeros(batch_size, n_joints, dtype=torch.float32, device=kp_2d_croped.device)

        # norm confidences
        confidences = confidences.view(batch_size, n_cams, *confidences.shape[1:])
        confidences = confidences / confidences.sum(dim=1, keepdim=True)
        ic(confidences.shape)
        confidences = confidences + 1e-5

        # ic(kp_2d_croped.requires_grad, confidences.requires_grad)

        if self.det_conf_thr is not None:
            for batch_i in range(batch_size):
                for joint_i in range(n_joints):
                    cams_detected = kp_2d_croped[batch_i, :, joint_i, 2] > self.det_conf_thr
                    cam_idx = torch.where(cams_detected)[0]
                    point = kp_2d_croped[batch_i, cam_idx, joint_i, :2]  # a joint in all views
                    confidence = confidences[batch_i, cam_idx, joint_i]
                    # ic(cam_idx, point.shape, proj_matrices.shape, confidences.shape)

                    if torch.sum(cams_detected) < 2:
                        continue
                    kp_3d[batch_i, joint_i], res_triang[batch_i, joint_i] = \
                        self.triangulate_point(proj_matrices[batch_i, cam_idx], point, confidence)
        else:
            for batch_i in range(batch_size):
                for joint_i in range(n_joints):
                    points = kp_2d_croped[batch_i, :, joint_i, :2]  # a joint in all views
                    confidence = confidences[batch_i, :, joint_i]
                    kp_3d[batch_i, joint_i], res_triang[batch_i, joint_i] = \
                        self.triangulate_point(proj_matrices[batch_i], points, confidence)
        return kp_3d, res_triang, kp_2d_croped, kp_2d_heatmap

    def get_sup_loss(self, output, target, target_visible=None):
        """Calculate supervised 3d keypoint regressive loss.

        Args:
            output (torch.Tensor[bs, num_joints, 3]): Output 3d keypoint coordinates.
            target (torch.Tensor[bs, num_joints, 3]): Target 3d keypoint coordinates.
            target_weight (torch.Tensor[bs, num_joints, 1]):
                Weights across different joint types.
        """

        losses = dict()

        assert not isinstance(self.super_loss, nn.Sequential)
        # assert target.dim() == 3 and target_weight.dim() == 3
        # print(output.shape, target.shape, target_visible.shape)
        # print(output)
        # print(target)
        # print(target_visible > 0)
        # print(output[target_visible > 0].shape)
        output = output[target_visible > 0].double()
        target = target[target_visible > 0].double()

        losses['sup_3d_loss'] = self.super_loss(output, target)
        return losses

    def get_unSup_loss(self, res_triang):
        """
        Calculate the triangulation residual loss, unsupervised 3d loss
        Args:
            res_triang: [bs, ??]
        Returns:

        """
        losses = dict()
        losses['unSup_3d_loss'] = torch.mean(res_triang)
        return losses
