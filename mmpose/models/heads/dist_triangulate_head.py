import numpy as np
import torch
import torch.nn as nn
from icecream import ic

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
class DistTriangHead(nn.Module):
    def __init__(self, num_cams=6, img_shape=[256, 256], heatmap_shape=[64, 64],
                 softmax_heatmap=True,  det_conf_thr=None,
                 train_cfg=None, test_cfg=None):
        super().__init__()
        self.num_cams = num_cams
        [self.h_img, self.w_img] = img_shape
        [self.h_map, self.w_map] = heatmap_shape
        self.softmax = False, softmax_heatmap
        self.det_conf_thr = det_conf_thr  # weather use the 2d detect confidence to mask the fail detection points
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

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

        # compute the heatmap confidence
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

    def triangulate_heatmap(self, proj_matrices, heatmaps):
        """
        Args:
            proj_matrices: [n_cams, 3, 4]
            heatmaps: [n_cams, 64, 64]

        Returns:

        """

        if self.softmax:
            heatmaps = heatmaps * 100.0
            heatmaps = heatmaps.reshape((heatmaps.shape[0], -1))
            heatmaps = nn.functional.softmax(heatmaps, dim=1)
            heatmaps = heatmaps.reshape((-1, 64, 64))
        else:
            heatmaps = nn.functional.relu(heatmaps)
        heatmaps = heatmaps/torch.sum(heatmaps, dim=[1,2], keepdim=True)
        # ic(heatmaps.shape)
        ic(torch.max(heatmaps.view(heatmaps.size(0), -1), dim=1).values)
        ic(torch.min(heatmaps.view(heatmaps.size(0), -1), dim=1).values)
        heatmaps = torch.clamp(heatmaps, min=0.2, max=1)
        ic(torch.max(heatmaps.view(heatmaps.size(0), -1), dim=1).values)
        ic(torch.min(heatmaps.view(heatmaps.size(0), -1), dim=1).values)
        heatmaps = torch.transpose(heatmaps, 1, 2)
        assert len(proj_matrices) == len(heatmaps)
        n_views = len(proj_matrices)
        x_coord = torch.arange(64)
        y_coord = torch.arange(64)
        grid_x, grid_y = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
        grid = torch.cat((grid_x[..., None], grid_y[..., None]), dim=-1).to(heatmaps.device)
        grid = grid*4
        # grid = torch.cat((grid[None, ...].repeat(n_views, 1,1,1), heatmaps[..., None]), dim=-1)
        AA = []
        for i in range(n_views):
            dist = torch.cat((grid, heatmaps[i, ..., None]), dim=-1)
            dist = dist[dist[..., -1]>self.det_conf_thr]  # [num, 3]
            A = proj_matrices[i, 2:3][0] * dist[:, :2, None]
            A -= proj_matrices[i, :2]   #
            A *= dist[:, -1, None, None]
            AA.append(A)
        AA = torch.cat(AA)
        # ic(AA.shape)
        u, s, vh = torch.svd(AA.view(-1, 4))

        point_3d_homo = -vh[:, 3]
        point_3d = homogeneous_to_euclidean(point_3d_homo.unsqueeze(0))[0]
        return point_3d, s[-1]

        # aa = torch.cat((grid_x[..., None], grid_y[..., None]), dim=-1).to(heatmaps.device)
        # ic(aa.shape)
        # # bb = torch.cat((aa[None, ...], heatmaps[..., None]), dim=-1)
        # # ic(aa[None, ...].repeat(n_views, 1,1,1))
        # bb = torch.cat((aa[None, ...].repeat(n_views, 1,1,1), heatmaps[..., None]), dim=-1)
        # ic(bb.shape)
        # cc = bb[bb[..., -1]>0.5]
        # ic(cc.shape)
        # ic(torch.where(bb[..., -1]>0.5))
        # ic(len(torch.where(bb[..., -1]>0.5)[0]))
        # AA = []
        # for i in range(n_views):
        #     ic(aa[None, ...].shape, heatmaps[i, ..., None].shape)
        #     bb = torch.cat((aa, heatmaps[i, ..., None]), dim=-1)
        #     ic(bb.shape)
        #     ic(torch.where(bb[..., -1]>0.4))
        #     cc = bb[bb[..., -1]>0.9]
        #     ic(cc)
        #     ic(cc.shape)
        #     A = proj_matrices[i, 2:3][0]*cc[:, :2, None]*4
        #     ic(A.shape)
        #     A -= proj_matrices[i, :2]
        #
        #     ic(A.shape)
        #     ic(cc[:, -1, None].shape)
        #     A *= cc[:, -1, None, None]
        #     ic(A.shape)
        #     AA.append(A)
        #     # A = A *
        # AA = torch.cat(AA)
        # ic(AA.shape)
        # u, s, vh = torch.svd(AA.view(-1, 4))
        # point_3d_homo = -vh[:, 3]
        # point_3d = homogeneous_to_euclidean(point_3d_homo.unsqueeze(0))[0]
        # ic(point_3d)

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
        # ic(heatmap.shape)
        # ic(proj_matrices.shape)

        kp_2d_heatmap, kp_2d_croped = self.compute_kp_coords(
            heatmap)  # [bs*num_cams, num_joints, 3] with [x, y, confidence]
        # ic(kp_2d_croped)
        batch_size = int(heatmap.shape[0]/self.num_cams)
        heatmap = heatmap.view([batch_size, self.num_cams, *heatmap.shape[1:]])
        n_joints = heatmap.shape[2]
        kp_3d = torch.zeros(batch_size, n_joints, 3, dtype=torch.float32, device=heatmap.device)
        res_triang = torch.zeros(batch_size, n_joints, dtype=torch.float32, device=heatmap.device)
        for batch_i in range(batch_size):
            for joint_i in range(n_joints): # n_joints
                kp_3d[batch_i, joint_i], res_triang[batch_i, joint_i] = \
                    self.triangulate_heatmap(proj_matrices[batch_i], heatmap[batch_i, :, joint_i])
        reproject_kp_2d = self.reproject(kp_3d, proj_matrices)
        return kp_3d, res_triang, kp_2d_croped, reproject_kp_2d, kp_2d_heatmap

    def reproject(self, kp_3d, proj_matrices):
        """
        Args:
            kp_3d: np.array, [bs, n_joints, 3]
            proj_matrices: [bs, num_cams, 3, 4]
        Returns:
            reproject_kp_2d: [bs, num_cams, num_joints, 2]
        """
        kp_3d_temp = kp_3d.detach().clone()
        if kp_3d_temp.shape[-1] == 3:
            kp_3d_temp = torch.cat(
                [kp_3d_temp, torch.ones([*kp_3d_temp.shape[:-1], 1], dtype=torch.float64, device=kp_3d_temp.device)],
                dim=-1)

        pseudo_kp_2d = torch.einsum('bcdk, bjk -> bcjd', proj_matrices, kp_3d_temp)
        pseudo_kp_2d = pseudo_kp_2d / (pseudo_kp_2d[..., -1].unsqueeze(-1))
        pseudo_kp_2d = pseudo_kp_2d[..., :-1]
        return pseudo_kp_2d

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
