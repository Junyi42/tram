import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import numpy as np
from glob import glob
from lib.pipeline import visualize_tram

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='./example_video.mov', help='input video')
parser.add_argument('--img_folder', type=str, default=None, help='input image folder (alternative to video)')
parser.add_argument('--bin_size', type=int, default=-1, help='rasterization bin_size; set to [64,128,...] to increase speed')
parser.add_argument('--floor_scale', type=int, default=3, help='size of the floor')
args = parser.parse_args()

# File and folders
if args.img_folder is not None:
    # Use image folder as input
    img_folder = args.img_folder
    seq = os.path.basename(os.path.dirname(img_folder))
else:
    # Use video as input
    file = args.video
    root = os.path.dirname(file)
    seq = os.path.basename(file).split('.')[0]
    img_folder = f'results/{seq}/images'

seq_folder = f'results/{seq}'

##### Combine camera & human motion #####
# Render video
print('Visualize results ...')
visualize_tram(seq_folder, floor_scale=args.floor_scale, bin_size=args.bin_size)
