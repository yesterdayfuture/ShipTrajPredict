import sys
import os
# 将父目录添加到Python路径
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.getcwd())

import torch
import numpy as np
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]




# kimi_model.pth 与 kimi_model2.pth、kimi_model5.pth 使用的 模型结构和参数，参数如下：
# d_input=5, d_model=128, nhead=8, nlayers=2,seq_len=32, horizon=6
class TrajTransformer(nn.Module):
    def __init__(self, d_input, d_model, nhead, nlayers, seq_len, horizon, n_vtype):
        super().__init__()
        self.horizon = horizon
        self.embed = nn.Linear(d_input, d_model)
        self.vtype_embed = nn.Embedding(n_vtype, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2*horizon)   # 输出 Δlat, Δlon 共 2*H
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(d_model, d_model*2),
        #     nn.Linear(d_model*2, d_model*2),
        #     nn.Dropout(0.1)
        #     nn.Linear(d_model*2, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 2*horizon)   # 输出 Δlat, Δlon 共 2*H
        # )

    def forward(self, x, vtype):
        # x: (B, SEQ, d_feat)
        B = x.size(0)
        x = self.embed(x) + self.vtype_embed(vtype).unsqueeze(1)
        x = self.pos(x)
        out = self.encoder(x)          # (B, SEQ, d_model)
        last = out[:, -1]              # (B, d_model)
        delta = self.fc(last).view(B, self.horizon, 2)
        return delta
    


# kimi_model3.pth、kimi_model4.pth 使用的 模型结构和参数，参数如下：
# d_input=5, d_model=256, nhead=8, nlayers=4,seq_len=64, horizon=6
class TrajTransformer2(nn.Module):
    def __init__(self, d_input, d_model, nhead, nlayers, seq_len, horizon, n_vtype):
        super().__init__()
        self.horizon = horizon
        self.embed = nn.Linear(d_input, d_model)
        self.vtype_embed = nn.Embedding(n_vtype, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=nlayers)
       
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.Linear(d_model*2, d_model*2),
            nn.Dropout(0.1),
            nn.Linear(d_model*2, 64),
            nn.ReLU(),
            nn.Linear(64, 2*horizon)   # 输出 Δlat, Δlon 共 2*H
        )

    def forward(self, x, vtype):
        # x: (B, SEQ, d_feat)
        B = x.size(0)
        x = self.embed(x) + self.vtype_embed(vtype).unsqueeze(1)
        x = self.pos(x)
        out = self.encoder(x)          # (B, SEQ, d_model)
        last = out[:, -1]              # (B, d_model)
        delta = self.fc(last).view(B, self.horizon, 2)
        return delta
    


# kimi_model6.pth 使用的 模型结构和参数，参数如下：
# d_input=5, d_model=256, nhead=8, nlayers=4,seq_len=32, horizon=6
class TrajTransformer3(nn.Module):
    def __init__(self, d_input, d_model, nhead, nlayers, seq_len, horizon, n_vtype):
        super().__init__()
        self.horizon = horizon
        self.embed = nn.Linear(d_input, d_model)
        self.vtype_embed = nn.Embedding(n_vtype, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2*horizon)   # 输出 Δlat, Δlon 共 2*H
        )


    def forward(self, x, vtype):
        # x: (B, SEQ, d_feat)
        B = x.size(0)
        x = self.embed(x) + self.vtype_embed(vtype).unsqueeze(1)
        x = self.pos(x)
        out = self.encoder(x)          # (B, SEQ, d_model)
        last = out[:, -1]              # (B, d_model)
        delta = self.fc(last).view(B, self.horizon, 2)
        return delta





# 反归一化， 并转为 海里距离
def haversine(pred, target, stats):
    lat1, lon1 = pred[...,0]*(stats['lat_max']-stats['lat_min'])+stats['lat_min'], \
                 pred[...,1]*(stats['lon_max']-stats['lon_min'])+stats['lon_min']
    lat2, lon2 = target[...,0], target[...,1]
    lat1, lat2, dlat, dlon = map(torch.deg2rad, [lat1, lat2, lat2-lat1, lon2-lon1])
    a = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2)**2
    return 3440.065 * 2 * torch.asin(torch.sqrt(a))  # 海里

# 加权 Huber Loss
class WeightedHuber(nn.Module):
    def __init__(self, horizon, delta=0.1):
        super().__init__()
        self.delta = delta
        self.w = torch.linspace(1, 2, horizon).unsqueeze(0)  # (1,H)
    def forward(self, pred, target, stats):
        dist = haversine(pred, target, stats)  # (B,H)
        # Huber
        err = torch.abs(dist)
        loss = torch.where(err < self.delta,
                           0.5 * err ** 2,
                           self.delta * (err - 0.5 * self.delta))
        return (loss * self.w.to(loss.device)).mean()