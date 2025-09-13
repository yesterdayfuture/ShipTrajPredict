import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 1. 全局归一化参数（一次性算好，推理时复用）
def compute_global_stats(df):
    lat_min, lat_max = df.LAT.min(), df.LAT.max()
    lon_min, lon_max = df.LON.min(), df.LON.max()
    sog_mean, sog_std = df.SOG.mean(), df.SOG.std()
    return dict(lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
                sog_mean=sog_mean, sos_std=sog_std)

# 2. 归一化 / 反归一化函数
def normalize(df, stats):
    df = df.copy()
    df['LAT'] = (df.LAT - stats['lat_min']) / (stats['lat_max'] - stats['lat_min'])
    df['LON'] = (df.LON - stats['lon_min']) / (stats['lon_max'] - stats['lon_min'])
    df['SOG'] = (df.SOG - stats['sog_mean']) / stats['sos_std']
    df['COG_sin'] = np.sin(np.deg2rad(df.COG))
    df['COG_cos'] = np.cos(np.deg2rad(df.COG))
    return df


# 数据增强 添加随机噪声
def jitter(traj, σ=0.01):
    return traj + torch.randn_like(traj) * σ

# 数据增强 添加随机旋转
def rotate(traj, angle):
    θ = torch.tensor(angle).float()
    cos, sin = torch.cos(θ), torch.sin(θ)
    R = torch.tensor([[cos, -sin], [sin, cos]])
    traj[..., :2] = traj[..., :2] @ R
    return traj


# 3. Dataset
# augment：表示 是否 进行数据增强
class AisDataset(Dataset):
    def __init__(self, filepath, SEQ_LEN, HORIZON, stats, augment = False):
        self.SEQ_LEN  = SEQ_LEN
        self.HORIZON  = HORIZON
        self.stats    = stats

        df = pd.read_csv(filepath)
        df = normalize(df, stats)

        # 船舶类型转 int id
        df['VType'] = pd.Categorical(df.VesselType).codes
        self.vtype_num = df.VType.max() + 1

        self.samples = []
        for mmsi, grp in df.groupby('MMSI'):
            grp = grp.sort_values('timeUnix')
            arr = grp[['LAT', 'LON', 'SOG', 'COG_sin', 'COG_cos', 'VType']].to_numpy()
            for i in range(SEQ_LEN, len(arr) - HORIZON + 1):
                x = arr[i-SEQ_LEN:i, :5]          # (SEQ, 5)
                vtype = int(arr[i-1, 5])          # 用上一帧的 type
                y = arr[i:i+HORIZON, :2] - arr[i-1, :2]  # (HORIZON, 2)  偏移
                raw_y = arr[i:i+HORIZON, :2]   # 未来 H 步的归一化坐标
                self.samples.append((x, vtype, y, raw_y))

        if augment:
            self.samples_augment = []
            #进行数据增强
            for x, vtype, y, raw_y  in self.samples:
                cur_x = jitter(x)
                cur_x = rotate(cur_x, np.radians(np.random.uniform(-5, 5)))
                self.samples_augment.append((cur_x, vtype, y, raw_y))

            #将 数据增强后的数据，追加到 数据样本中
            self.samples.extend(self.samples_augment)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, vtype, y, raw_y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(vtype, dtype=torch.long), \
               torch.tensor(y, dtype=torch.float32), \
               torch.tensor(raw_y, dtype=torch.float32)