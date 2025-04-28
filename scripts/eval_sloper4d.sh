CUDA_VISIBLE_DEVICES=6

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python scripts/estimate_camera.py --img_folder "/home/junyi42/human_in_world/video-mimic/junyi_sloper4d/input_images/seq007_garden_001_imgs/cam01"
python scripts/estimate_humans.py --img_folder "/home/junyi42/human_in_world/video-mimic/junyi_sloper4d/input_images/seq007_garden_001_imgs/cam01"
python scripts/visualize_tram.py --img_folder "/home/junyi42/human_in_world/video-mimic/junyi_sloper4d/input_images/seq007_garden_001_imgs/cam01"