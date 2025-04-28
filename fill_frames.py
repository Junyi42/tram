import numpy as np
import torch
from pathlib import Path

in_path  = Path("sloper4d_eval_script/tram/results/seq007_garden_001_imgs/hps/hps_track_0_old.npy")
out_path = in_path.with_name(in_path.stem + "_filled.npy")

data = np.load(in_path, allow_pickle=True).item()

total_frames = 2400
existing = np.asarray(data['frame']).astype(int)
missing  = sorted(set(range(total_frames)) - set(existing))
print(f"Missing frames: {missing}")

def nearest(idx, pool):
    pool = np.asarray(pool)
    dist = np.abs(pool - idx)
    i    = np.argmin(dist)
    tie  = np.where(dist == dist[i])[0]
    return pool[min(tie)] if len(tie) > 1 else pool[i]

new_data = {}
for key, arr_orig in data.items():

    # 1) 统一成 numpy 做插值 / 复制
    if isinstance(arr_orig, torch.Tensor):
        arr_np   = arr_orig.cpu().numpy()
        torch_dt = arr_orig.dtype
        torch_dev= arr_orig.device          # 记住 dtype / device
    else:
        arr_np   = np.asarray(arr_orig)     # ndarray 或标量
        torch_dt = torch_dev = None

    # 2) 构建填补后的 full_np
    if key == "frame":
        full_np = np.arange(total_frames, dtype=arr_np.dtype)
    else:
        new_shape = (total_frames, *arr_np.shape[1:])
        full_np   = np.empty(new_shape, dtype=arr_np.dtype)
        full_np[existing] = arr_np
        for idx in missing:
            full_np[idx] = arr_np[nearest(idx, existing)]

    # 3) 恢复成原来的类型
    if torch_dt is not None:                     # 原来是 torch
        full_torch = torch.as_tensor(full_np, dtype=torch_dt, device=torch_dev)
        new_data[key] = full_torch
    else:                                        # 原来就是 numpy / 其他
        new_data[key] = full_np

# 4) 保存
np.save(out_path, new_data)
print(f"Saved padded dict to {out_path}")
