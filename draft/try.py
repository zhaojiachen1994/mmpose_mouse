import h5py
if __name__ == "__main__":
    filename = "D:/Nutstore/000-inbox/valid.h5"
    with h5py.File(filename, "r") as f:
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        print(type(f[a_group_key]))
        # print("===")
    # P = torch.randn([3, 3, 4])
    # ic(P)
    # points = torch.randn([3, 5, 2, 1])
    # ic(points.shape)
    # ic(P[:, 2:3].shape)
    # A = P[:, 2:3].expand(3, 10, 4).reshape(3, 5, 2, 4)
    # ic(A.shape)
    #
    # A = A*points#.view(3,10,1)
    # ic(A.shape)
    # ic(P[:, :2].shape)
    # A -= P[:,None, :2]
    # ic(A.shape)
    # u, s, vh = torch.svd(A.view(-1, 4))
    # ic(s)
    # ic(vh.shape)
    # ic(torch.meshgrid())
    # point_3d_homo = -vh[:, 3]
    # point_3d = homogeneous_to_euclidean(point_3d_homo.unsqueeze(0))[0]
    # ic(point_3d)
