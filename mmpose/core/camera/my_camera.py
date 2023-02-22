import numpy as np

from .camera_base import CAMERAS


@CAMERAS.register_module()
class MyCamera:
    """
     Args:
        param (dict): camera parameters including:
            - R: 3x3, camera rotation matrix
            - t: 3x1, camera translation
            - K: (optional) 3x3, camera intrinsic matrix
            - k: (optional) nx1, camera radial distortion coefficients
            - p: (optional) mx1, camera tangential distortion coefficients
            - f: (optional) 2x1, camera focal length
            - c: (optional) 2x1, camera center
        if K is not provided, it will be calculated from f and c.
    """

    def __init__(self, param):
        self.param = {}
        R = np.array(param['R'], dtype=np.float32).copy()
        assert R.shape == (3, 3)
        self.param['R'] = R

        t = np.array(param['t'], dtype=np.float32).copy()
        assert t.size == 3
        self.param['t'] = t.reshape(3, 1)

        if 'K' not in param and 'f' in param and 'c' in param:
            f = np.array(param['f'], dtype=np.float32).reshape([2, 1])
            c = np.array(param['c'], dtype=np.float32).reshape([2, 1])
            K = np.concatenate((np.diagflat(f), c), axis=-1).T
            self.param['K'] = np.hstack([K, np.array([0.0, 0.0, 1.0]).reshape([3, 1])]).T

    @property
    def extrinsics(self):
        return np.hstack([self.param['R'], self.param['t']])

    @property
    def projection(self):
        return self.param['K'].dot(self.extrinsics)

    def world_to_pixel(self, X):
        """
        Args:
            X:
            [J, C]: shape of joint coordinates of a person with J joints.
            [N, J, C]: shape of a batch of person joint coordinates.
            [N, T, J, C]: shape of a batch of pose sequences.
        Returns:
        """
        X = np.concatenate([X, np.ones(X[..., :1].shape)], axis=-1)
        X_2d = X @ self.projection.T
        X_2d = X_2d / X_2d[..., 2:]
        # X_2d = X_2d[..., :2]
        return X_2d
