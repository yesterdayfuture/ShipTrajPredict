import sys
import os
# 将父目录添加到Python路径
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from sklearn.metrics import mean_squared_error
# from kimi.dataProcess import *
# from kimi.train import *

from dataProcess import *
from train import *
from geopy.distance import geodesic

'''
AIS 轨迹预测通常用：
    ADE（Average Displacement Error）：单步平均距离误差（海里或米）
    FDE（Final Displacement Error）：预测末点误差
    在 1-5 海里范围、6-30 分钟预测时长下，SOTA 模型（如 TNT、TrajTransformer、ST-Transformer）在港口/海峡场景能做到：
        ADE 0.05-0.15 海里
        FDE 0.10-0.30 海里

'''


#文件 路径
file_path = r"../data/AIS_2023_12_28_test.csv"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


# =============== 三、模型评估 =========================

# =============== 3.1、模型评估函数 (直接评估 归一化后的数据误差) =========================
@torch.no_grad()
def evalADE_FDE(model, loader, stats):
    model.eval()
    ades, fdes = [], []
    for x, vtype, y, raw_y in loader:
        x, vtype, y = x.to(device), vtype.to(device), y.to(device)
        delta = model(x, vtype).to(device)               # (B,H,2)
        ade = torch.mean(torch.norm(delta - y, dim=-1))  # 归一化空间
        fde = torch.norm(delta[:, -1] - y[:, -1], dim=-1).mean()
        ades.append(ade.item())
        fdes.append(fde.item())
    return np.mean(ades), np.mean(fdes)


# =============== 3.2、模型评估函数（将预测结果转为真实经纬度，然后进行评估） =========================
# 1 海里 == 1.85 km
# 把经纬度转成海里（1° lat ≈ 60 海里；1° lon ≈ 60*cos(lat) 海里）
def haversine(pred, target):
    """
    pred, target: (B, H, 2) 的 (lat, lon) 度
    return: (B, H) 的距离（海里）
    """
    R = 3440.065  # 地球半径（海里）
    lat1, lon1 = pred[..., 0], pred[..., 1]
    lat2, lon2 = target[..., 0], target[..., 1]

    # 转弧度
    lat1, lat2, lon1, lon2 = map(torch.deg2rad, [lat1, lat2, lon1, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.asin(torch.sqrt(a))
    return R * c  # (B, H)

@torch.no_grad()
def evalADE_FDE_latLon(model, loader, stats):
    model.eval()
    ades, fdes = [], []

    lat_min, lat_max = stats['lat_min'], stats['lat_max']
    lon_min, lon_max = stats['lon_min'], stats['lon_max']
    lat_scale, lon_scale = lat_max - lat_min, lon_max - lon_min

    for x, vtype, y, raw_y in loader:
        x, vtype = x.to(device), vtype.to(device)

        # 原始归一化经纬度 进行 反归一化
        y_true_norm = raw_y.to(device)
        y_true_lat = y_true_norm[..., 0] * lat_scale + lat_min
        y_true_lon = y_true_norm[..., 1] * lon_scale + lon_min
        y_true = torch.stack([y_true_lat, y_true_lon], dim=-1)


        # 模型预测 偏移
        delta = model(x, vtype)  # (B, H, 2) 归一化坐标

        # 反归一化 -> 真实 lat/lon
        last = x[:, -1, :2].view(-1,1,2).to(delta.device)
        pred_norm = last + delta                  # 归一化坐标

        pred_lat = pred_norm[..., 0] * lat_scale + lat_min
        pred_lon = pred_norm[..., 1] * lon_scale + lon_min
        pred = torch.stack([pred_lat, pred_lon], dim=-1)  # (B, H, 2)

        # 距离矩阵 (B, H) 海里
        dist = haversine(pred, y_true)

        ade = dist.mean().item()
        fde = dist[:, -1].mean().item()
        ades.append(ade)
        fdes.append(fde)

    return np.mean(ades), np.mean(fdes)


# =============== 3.3、模型评估函数（将预测结果转为真实经纬度，然后进行评估准确率ACC） =========================

def great_circle_error(pred, y_true):
    """矢量大圆距离，单位 m"""
    # print(type(pred), type(y_true))
    # print(type(y_true[0]))
    # print(pred[0][0])
    return np.array([geodesic((y_true[i][0], y_true[i][1]),
                              (pred[i][0], pred[i][1])).m
                     for i in range(len(pred))])

def accuracy_at_tau(err, tau):
    """准确率@τ"""
    return np.mean(err <= tau)

def eval_traj(pred, y_true, tau_list=(100, 500, 1000, 1800, 3600, 5400)):
    err = great_circle_error(pred, y_true)
    acc = {f'ACC@{t}m': accuracy_at_tau(err, t) for t in tau_list}
    return err, acc

@torch.no_grad()
def eval_Acc_latLon(model, loader, stats):
    model.eval()
    y_trues, preds = [], []

    lat_min, lat_max = stats['lat_min'], stats['lat_max']
    lon_min, lon_max = stats['lon_min'], stats['lon_max']
    lat_scale, lon_scale = lat_max - lat_min, lon_max - lon_min

    for x, vtype, y, raw_y in loader:
        x, vtype = x.to(device), vtype.to(device)

        # 原始归一化经纬度 进行 反归一化
        y_true_norm = raw_y.to(device)
        y_true_lat = y_true_norm[..., 0] * lat_scale + lat_min
        y_true_lon = y_true_norm[..., 1] * lon_scale + lon_min
        y_true = torch.stack([y_true_lat, y_true_lon], dim=-1)
        y_true = y_true.reshape(-1,  2).detach().numpy()
        # print(y_true.shape)

        # 模型预测 偏移
        delta = model(x, vtype)  # (B, H, 2) 归一化坐标

        # 反归一化 -> 真实 lat/lon
        last = x[:, -1, :2].view(-1,1,2).to(delta.device)
        pred_norm = last + delta                  # 归一化坐标

        pred_lat = pred_norm[..., 0] * lat_scale + lat_min
        pred_lon = pred_norm[..., 1] * lon_scale + lon_min
        pred = torch.stack([pred_lat, pred_lon], dim=-1)  # (B, H, 2)
        pred = pred.reshape(-1, 2).detach().numpy()

        # print(pred.shape)
        y_trues.extend(y_true)
        preds.extend(pred)
        # break
    # print(y_trues[0])
    err, acc = eval_traj(preds,y_trues)

    print('---- 轨迹预测准确率 ----')
    for k, v in acc.items():
        print(f'{k}: {v:.1%}')

    return err, acc





# =============== 四、模型推理 =============================

# ================== 4.1、推理与反归一化 ========================
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
    vtype = vtype.to(device)
    # print(x.dim(),vtype.dim())


    delta = model(x, vtype).squeeze(0).cpu()  # (H,2)

    print(f'x size {x.size()} delta size {delta.size()}')
    # 反归一化
    last = x[:, -1, :2].cpu().view(-1,1,2)
    pred_norm = last + delta                  # 归一化坐标
    pred = torch.stack([torch.tensor(denormalize(*k, stats)) for p in pred_norm for k in p])
    return pred  # (H,2) 的经纬度


def true_label(y, stats):

    y_true = torch.stack([torch.tensor(denormalize(*k, stats)) for p in y for k in p])
    return y_true                               # (H,2) 的经纬度


if __name__ == '__main__':

    # ================ 一、数据加载 ==========================
    # 1. 计算全局统计
    raw_df = pd.read_csv(file_path)
    stats  = compute_global_stats(raw_df)

    val_ds = AisDataset(file_path, SEQ_LEN, HORIZON, stats)
    val_dl = DataLoader(val_ds, batch_size=512)

    # ============== 二、模型加载 ==============================
    model = torch.load( r"kimi_model6.pth", weights_only=False, map_location=torch.device('cpu'))
    print(Fore.YELLOW + "成功 加载 预测模型...\n"  + Style.RESET_ALL)

    # torch.save(model.state_dict(), 'kimi/kimi_model6.pt')


    # # 评估 预测误差
    # ade, fde = evalADE_FDE(model, val_dl, stats)
    # print(f'ADE(norm)={ade:.4f}, FDE(norm)={fde:.4f}')
    #
    # # 评估 反归一化后的 海里 误差
    # ade_nmi, fde_nmi = evalADE_FDE_latLon(model, val_dl, stats)
    # print(f'ADE={ade_nmi:.3f} nmi, FDE={fde_nmi:.3f} nmi')

    # 评估 反归一化后的 准确率
    err, acc = eval_Acc_latLon(model, val_dl, stats)


    pred = None
    y_true = None

    for x, vtype, y, raw_y in val_dl:
        #只预测第一个批次 里面的 第一条 数据
        # print(vtype)
        pred = predict(model, x[0], vtype[0].unsqueeze(0), stats)
        y_true = true_label(raw_y[0].unsqueeze(0), stats)
        break

    for cur_y, cur_pred in zip(y_true,pred):
        print(f'真实经纬度：{cur_y}  预测经纬度： {cur_pred}')

