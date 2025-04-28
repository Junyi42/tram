import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import torch
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from glob import glob
from tqdm import tqdm
from collections import defaultdict

from lib.utils.eval_utils import *
from lib.utils.rotation_conversions import *
from lib.vis.traj import *
from lib.camera.slam_utils import eval_slam
from sloper4d_loader import SLOPER4D_Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='results/sloper4d_seq007')
parser.add_argument('--pred_cam_path', type=str, default='/home/junyi42/human_in_world/sloper4d_eval_script/tram/results/seq007_garden_001_imgs/camera.npy')
parser.add_argument('--pred_smpl_path', type=str, default='/home/junyi42/human_in_world/sloper4d_eval_script/tram/results/seq007_garden_001_imgs/hps/hps_track_0.npy')
parser.add_argument('--gt_pkl_path', type=str, default='/home/junyi42/human_in_world/demo_data/sloper4d/seq007_garden_001/seq007_garden_001_labels.pkl')
args = parser.parse_args()
input_dir = args.input_dir

# Load Sloper4D dataset
sloper4d_data = SLOPER4D_Dataset(
    args.gt_pkl_path, 
    device='cpu',  # Use CPU so we don't waste GPU memory
    return_torch=True,  # Get tensors directly
    fix_pts_num=False,  # No need to fix points number for evaluation
    print_info=False,   # Suppress detailed info
    return_smpl=True    # We need SMPL data
)

# SMPL
smpl = SMPL()
smpls = {g:SMPL(gender=g) for g in ['neutral', 'male', 'female']}

# Evaluations: world-coordinate SMPL
accumulator = defaultdict(list)
m2mm = 1e3
human_traj = {}
total_invalid = 0

# Process Sloper4D dataset
seq_name = os.path.basename(args.gt_pkl_path).split('_labels.pkl')[0]
print(f"Evaluating sequence: {seq_name}")

# Get ground truth data
gender = sloper4d_data.smpl_gender
# Check if smpl_pose is already a numpy array or a torch tensor
if isinstance(sloper4d_data.smpl_pose, np.ndarray):
    poses_body = sloper4d_data.smpl_pose[:, 3:]
    poses_root = sloper4d_data.smpl_pose[:, :3]
else:
    poses_body = sloper4d_data.smpl_pose[:, 3:].numpy()
    poses_root = sloper4d_data.smpl_pose[:, :3].numpy()

# Check if betas is a list, numpy array, or torch tensor
if isinstance(sloper4d_data.betas, list):
    betas_array = np.array(sloper4d_data.betas)
    betas = np.repeat(betas_array.reshape(1, -1), repeats=sloper4d_data.length, axis=0)
elif isinstance(sloper4d_data.betas, np.ndarray):
    betas = np.repeat(sloper4d_data.betas.reshape(1, -1), repeats=sloper4d_data.length, axis=0)
else:
    betas = np.repeat(sloper4d_data.betas.numpy().reshape(1, -1), repeats=sloper4d_data.length, axis=0)

# Check if global_trans is already a numpy array or a torch tensor
if isinstance(sloper4d_data.global_trans, np.ndarray):
    trans = sloper4d_data.global_trans
else:
    trans = sloper4d_data.global_trans.numpy()

# Get camera information
if isinstance(sloper4d_data.cam_pose[0], np.ndarray):
    cam_pose = np.array(sloper4d_data.cam_pose)
else:
    cam_pose = np.array([pose.numpy() for pose in sloper4d_data.cam_pose])
ext = np.zeros_like(cam_pose)
for i in range(len(cam_pose)):
    # Convert from world2cam to cam2world format needed by the evaluation
    R = cam_pose[i, :3, :3]
    t = cam_pose[i, :3, 3]
    ext[i, :3, :3] = R
    ext[i, :3, 3] = t

# Create valid mask (all frames are considered valid for Sloper4D)
valid = np.ones(sloper4d_data.length, dtype=bool)
total_invalid += (~valid).sum()

# Extract intrinsics
intrinsics = np.zeros((3, 3))
intrinsics[0, 0] = sloper4d_data.cam['intrinsics'][0]  # fx
intrinsics[1, 1] = sloper4d_data.cam['intrinsics'][1]  # fy
intrinsics[0, 2] = sloper4d_data.cam['intrinsics'][2]  # cx
intrinsics[1, 2] = sloper4d_data.cam['intrinsics'][3]  # cy
intrinsics[2, 2] = 1.0

tt = lambda x: torch.from_numpy(x).float()
gt = smpls[gender](body_pose=tt(poses_body), global_orient=tt(poses_root), betas=tt(betas), transl=tt(trans),
                pose2rot=True, default_smpl=True)
gt_vert = gt.vertices
gt_j3d = gt.joints[:,:24] 
gt_ori = axis_angle_to_matrix(tt(poses_root))

# Groundtruth local motion
poses_root_cam = matrix_to_axis_angle(tt(ext[:, :3, :3]) @ axis_angle_to_matrix(tt(poses_root)))
gt_cam = smpls[gender](body_pose=tt(poses_body), global_orient=poses_root_cam, betas=tt(betas),
                        pose2rot=True, default_smpl=True)
gt_vert_cam = gt_cam.vertices
gt_j3d_cam = gt_cam.joints[:,:24] 

# PRED
pred_cam = np.load(args.pred_cam_path, allow_pickle=True).item()
pred_smpl = np.load(args.pred_smpl_path, allow_pickle=True).item()

pred_rotmat = torch.tensor(pred_smpl['pred_rotmat'])    # T, 24, 3, 3
pred_shape = torch.tensor(pred_smpl['pred_shape'])      # T, 10
pred_trans = torch.tensor(pred_smpl['pred_trans'])      # T, 1, 3

mean_shape = pred_shape.mean(dim=0, keepdim=True)
pred_shape = mean_shape.repeat(len(pred_shape), 1)

pred = smpls['neutral'](body_pose=pred_rotmat[:,1:], 
                        global_orient=pred_rotmat[:,[0]], 
                        betas=pred_shape, 
                        transl=pred_trans.squeeze(),
                        pose2rot=False, 
                        default_smpl=True)
pred_vert = pred.vertices
pred_j3d = pred.joints[:, :24]

pred_camt = torch.tensor(pred_cam['pred_cam_T']) 
pred_camr = torch.tensor(pred_cam['pred_cam_R'])

pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:,None]
pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:,None]
pred_ori_w = torch.einsum('bij,bjk->bik', pred_camr, pred_rotmat[:,0])
pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)

# Make sure the predicted data matches GT size
min_len = min(len(gt_j3d), len(pred_j3d_w))
gt_j3d = gt_j3d[:min_len]
gt_ori = gt_ori[:min_len]
pred_j3d_w = pred_j3d_w[:min_len]
pred_ori_w = pred_ori_w[:min_len]
gt_j3d_cam = gt_j3d_cam[:min_len]
gt_vert_cam = gt_vert_cam[:min_len]
pred_j3d = pred_j3d[:min_len]
pred_vert = pred_vert[:min_len]
valid = valid[:min_len]

# Apply valid mask
gt_j3d = gt_j3d[valid]
gt_ori = gt_ori[valid]
pred_j3d_w = pred_j3d_w[valid]
pred_ori_w = pred_ori_w[valid]
gt_j3d_cam = gt_j3d_cam[valid]
gt_vert_cam = gt_vert_cam[valid]
pred_j3d = pred_j3d[valid]
pred_vert = pred_vert[valid]

# <======= Evaluation on the local motion
pred_j3d, gt_j3d_cam, pred_vert, gt_vert_cam = batch_align_by_pelvis(
    [pred_j3d, gt_j3d_cam, pred_vert, gt_vert_cam], pelvis_idxs=[1,2]
)
S1_hat = batch_compute_similarity_transform_torch(pred_j3d, gt_j3d_cam)
pa_mpjpe = torch.sqrt(((S1_hat - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
mpjpe = torch.sqrt(((pred_j3d - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
pve = torch.sqrt(((pred_vert - gt_vert_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm

accel = compute_error_accel(joints_pred=pred_j3d.cpu(), joints_gt=gt_j3d_cam.cpu())[1:-1]
accel = accel * (30 ** 2)       # per frame^s to per s^2

accumulator['pa_mpjpe'].append(pa_mpjpe)
accumulator['mpjpe'].append(mpjpe)
accumulator['pve'].append(pve)
accumulator['accel'].append(accel)
# =======>

# <======= Evaluation on the global motion
chunk_length = 100
w_mpjpe, wa_mpjpe = [], []
for start in range(0, valid.sum() - chunk_length, chunk_length):
    end = start + chunk_length
    if start + 2 * chunk_length > valid.sum(): end = valid.sum() - 1
    
    target_j3d = gt_j3d[start:end].clone().cpu()
    pred_j3d = pred_j3d_w[start:end].clone().cpu()
    
    w_j3d = first_align_joints(target_j3d, pred_j3d)
    wa_j3d = global_align_joints(target_j3d, pred_j3d)
    
    w_jpe = compute_jpe(target_j3d, w_j3d)
    wa_jpe = compute_jpe(target_j3d, wa_j3d)
    w_mpjpe.append(w_jpe)
    wa_mpjpe.append(wa_jpe)

w_mpjpe = np.concatenate(w_mpjpe) * m2mm
wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm
# =======>

# <======= Evaluation on the entier global motion
# RTE: root trajectory error
pred_j3d_align = first_align_joints(gt_j3d, pred_j3d_w)
rte_align_first= compute_jpe(gt_j3d[:,[0]], pred_j3d_align[:,[0]])
rte_align_all = compute_rte(gt_j3d[:,0], pred_j3d_w[:,0]) * 1e2 

# ERVE: Ego-centric root velocity error
erve = computer_erve(gt_ori, gt_j3d, pred_ori_w, pred_j3d_w) * m2mm
# =======>

# <======= Record human trajectory
human_traj[seq_name] = {'gt': gt_j3d[:,0], 'pred': pred_j3d_align[:, 0]}
# =======>

accumulator['wa_mpjpe'].append(wa_mpjpe)
accumulator['w_mpjpe'].append(w_mpjpe)
accumulator['rte'].append(rte_align_all)
accumulator['erve'].append(erve)
    
for k, v in accumulator.items():
    accumulator[k] = np.concatenate(v).mean()

# Evaluation: Camera motion
results = {}

# Process camera motion for the sequence
# Get GT camera trajectory
cam_r = cam_pose[:,:3,:3].transpose(0,2,1)
cam_t = np.einsum('bij, bj->bi', cam_r, -cam_pose[:, :3, 3])
cam_q = matrix_to_quaternion(torch.from_numpy(cam_r)).numpy()

# Get predicted camera trajectory
pred_camt = torch.tensor(pred_cam['pred_cam_T'])
pred_camr = torch.tensor(pred_cam['pred_cam_R'])
pred_camq = matrix_to_quaternion(pred_camr)
pred_traj = torch.concat([pred_camt, pred_camq], dim=-1).numpy()

# Cut to the same length if needed
min_len = min(len(cam_t), len(pred_traj))
cam_t = cam_t[:min_len]
cam_q = cam_q[:min_len]
pred_traj = pred_traj[:min_len]

stats_slam, _, _ = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=True)
stats_metric, traj_ref, traj_est = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=False)

# Save results
re = {'traj_gt': traj_ref.positions_xyz,
      'traj_est': traj_est.positions_xyz, 
      'traj_gt_q': traj_ref.orientations_quat_wxyz,
      'traj_est_q': traj_est.orientations_quat_wxyz,
      'stats_slam': stats_slam,
      'stats_metric': stats_metric}

results[seq_name] = re

ate = np.mean([re['stats_slam']['mean'] for re in results.values()])
ate_s = np.mean([re['stats_metric']['mean'] for re in results.values()])
accumulator['ate'] = ate
accumulator['ate_s'] = ate_s

# Save evaluation results
for k, v in accumulator.items():
    print(k, accumulator[k])

df = pd.DataFrame(list(accumulator.items()), columns=['Metric', 'Value'])
df.to_excel(f"{args.input_dir}/evaluation.xlsx", index=False)
