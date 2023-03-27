# python tools/train.py "configs/mouse/hrnet_w48_dannce_2d_p12_256x256.py" --gpu-id 0

# python tools/test.py "configs/mouse/hrnet_w48_dannce_2d_p12_256x256.py" "work_dirs/hrnet_w48_dannce_2d_p12_256x256/best_AP_epoch_100.pth" --gpu-id 0


# python tools/test.py "configs/mouse/try_score_head.py" "work_dirs/hrnet_w48_dannce_2d_p12_256x256/best_AP_epoch_100.pth" --gpu-id 0


# train the mars_p5 2d top detection
# python tools/train.py "configs/mouse/dataset_mars_p5_top.py" --gpu-id 0

# python tools/test.py "configs/mouse/dataset_mars_p5_top.py" "work_dirs/dataset_mars_p5/epoch_10.pth" --gpu-id 0

# train the mars_p5 2d front detection
# python tools/train.py "configs/mouse/dataset_mars_p5_front.py" --gpu-id 0

# python tools/test.py "configs/mouse/dataset_mars_p5_front.py" "work_dirs/dataset_mars_p5_front/best_AP_epoch_2.pth" --gpu-id 0


"""train 2d detector on concated mars dataset"""
# python tools/train.py "configs/mouse/hrnet_w48_concat_mars_p5_256x256.py" --gpu-id 0

"""train 2d detector on concated mars p9 dataset"""
# python tools/train.py "configs/mouse/hrnet_w48_mars_p9_256x256.py" --gpu-id 0
