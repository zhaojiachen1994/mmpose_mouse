# python tools/train.py "configs/mouse/hrnet_w48_dannce_2d_p12_256x256.py" --gpu-id 0

# python tools/test.py "configs/mouse/hrnet_w48_dannce_2d_p12_256x256.py" "work_dirs/hrnet_w48_dannce_2d_p12_256x256/best_AP_epoch_100.pth" --gpu-id 0


# python tools/test.py "configs/mouse/try_score_head.py" "work_dirs/hrnet_w48_dannce_2d_p12_256x256/best_AP_epoch_100.pth" --gpu-id 0


# train the mars_p5 2d detection
# python tools/train.py "configs/mouse/dataset_mars_p5.py" --gpu-id 0
