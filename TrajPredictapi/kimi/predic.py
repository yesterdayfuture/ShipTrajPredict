'''
本文件 主要根据 已训练好的模型进行预测

主要被 ../utils/util_modelPredict.py 文件使用

'''
import sys
import os
# 将父目录添加到Python路径
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.getcwd())


import torch
from kimi.train import *
from pandas.core.frame import DataFrame
from kimi.model import TrajTransformer3

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 1. 全局归一化参数（一次性算好，推理时复用）
def compute_global_stats(df):
    lat_min, lat_max = df.LAT.min(), df.LAT.max()
    lon_min, lon_max = df.LON.min(), df.LON.max()
    sog_mean, sog_std = df.SOG.mean(), df.SOG.std()
    return dict(lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
                sog_mean=sog_mean, sos_std=sog_std)

# 2. 归一化
def normalize(df, stats):
    df = df.copy()
    df['LAT'] = (df.LAT - stats['lat_min']) / (stats['lat_max'] - stats['lat_min'])
    df['LON'] = (df.LON - stats['lon_min']) / (stats['lon_max'] - stats['lon_min'])
    df['SOG'] = (df.SOG - stats['sog_mean']) / stats['sos_std']
    df['COG_sin'] = np.sin(np.deg2rad(df.COG))
    df['COG_cos'] = np.cos(np.deg2rad(df.COG))
    return df


# 3. 反归一化函数
def denormalize(lat_norm, lon_norm, stats):
    lat = lat_norm * (stats['lat_max'] - stats['lat_min']) + stats['lat_min']
    lon = lon_norm * (stats['lon_max'] - stats['lon_min']) + stats['lon_min']
    return lat, lon

@torch.no_grad()
def predict(model, x, vtype, stats):
    model.eval()
    # print(vtype)
    # print(x.dim(),vtype.dim())

    # 统一 x 的维度 -> (1, SEQ, 5)
    if x.dim() == 2:
        x = x.unsqueeze(0)               # (1, SEQ, 5)
    x = x.float().to(device)

    # 统一 vtype 的维度 -> (1,)
    if isinstance(vtype, (int, np.integer)):
        vtype = torch.tensor([int(vtype)], dtype=torch.long)
    if vtype.dim() == 0:
        vtype = vtype.unsqueeze(0)
    vtype = vtype.to(device)
    # print(x.size(),vtype.dim())


    delta = model(x, vtype).squeeze(0).cpu()  # (H,2)

    # print(f'x size {x.size()} delta size {delta.size()}')
    # 反归一化
    last = x[:, -1, :2].cpu().view(-1,1,2)
    pred_norm = last + delta                  # 归一化坐标
    pred = torch.stack([torch.tensor(denormalize(*k, stats)) for p in pred_norm for k in p])
    return pred  # (H,2) 的经纬度


def predict_main(data:DataFrame, modelPath = r"kimi/kimi_model6.pt"):

    # 加载权重
    model_state = torch.load(modelPath)

    model = TrajTransformer3(d_input=5, d_model=256, nhead=8, nlayers=4,
                                seq_len=SEQ_LEN, horizon=HORIZON,
                                n_vtype=31)
    model.load_state_dict(model_state)

    model.to(device)

    # print(Fore.YELLOW + "成功 加载 预测模型...\n"  + Style.RESET_ALL)

    # 数据预处理
    # stats 包含当前数据的 经纬度的最大值、最小值，航速的平均值、标准差
    stats = compute_global_stats(data)

    # 数据进行归一化
    df = normalize(data,stats)

    arr = df[['LAT', 'LON', 'SOG', 'COG_sin', 'COG_cos', 'VType']].to_numpy()
    x = arr[-SEQ_LEN:, :5]
    vtype = int(arr[0, 5])
    # 进行预测
    pred = predict(model, torch.tensor(x, dtype=torch.float32), torch.tensor(vtype, dtype=torch.long), stats)

    return pred
