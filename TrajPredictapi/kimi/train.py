
import sys
import os
# 将父目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
import torch.optim as optim
from kimi.model import *
from kimi.dataProcess import *
from colorama import Fore, Style
import time
from tqdm import tqdm
import os

SEQ_LEN  = 32
HORIZON  = 6
BATCH    = 128
EPOCHS   = 60
use_weight_loss = False

# 文件 路径
file_path = r"../data/AIS_2023_12_28_train.csv"

# 模型 保存路径
model_path = r"kimi_model6.pth"


if __name__ == '__main__':

    # 1. 计算全局统计
    raw_df = pd.read_csv(file_path)
    stats  = compute_global_stats(raw_df)

    # 2. 数据集
    train_ds = AisDataset(file_path, SEQ_LEN, HORIZON, stats)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)

    # 3. 模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    if os.path.exists(model_path):
        model = torch.load(model_path).to(device)
        print(Fore.YELLOW + "已加载预训练模型\n" + Style.RESET_ALL)
    else:
        # model  = TrajTransformer(d_input=5, d_model=128, nhead=8, nlayers=2,
        #                         seq_len=SEQ_LEN, horizon=HORIZON,
        #                         n_vtype=train_ds.vtype_num).to(device)

        # model  = TrajTransformer2(d_input=5, d_model=256, nhead=8, nlayers=4,
        #                         seq_len=SEQ_LEN, horizon=HORIZON,
        #                         n_vtype=train_ds.vtype_num).to(device)

        model  = TrajTransformer3(d_input=5, d_model=256, nhead=8, nlayers=4,
                                seq_len=SEQ_LEN, horizon=HORIZON,
                                n_vtype=train_ds.vtype_num).to(device)

    opt    = optim.AdamW(model.parameters(), lr=1e-3)

    # 判断是否使用 加权损失函数
    crit   = None
    if use_weight_loss:
        crit = WeightedHuber(HORIZON)
    else:
        crit   = nn.HuberLoss(delta=0.1)

    print(f' 模型结构：\n {model} \n')

    print(Fore.RED + "模型开始训练...\n" + Style.RESET_ALL)


    min_loss = None

    # 训练 开始时间
    start_train = time.time()

    # 4. 训练
    for epoch in range(1, EPOCHS+1):

        # 本轮次 训练 开始时间
        curEpoch_start_time = time.time()

        model.train()

        # 用 tqdm 包装 DataLoader
        loop = tqdm(train_dl, total=len(train_dl), leave=True,
                desc=f'Epoch {epoch}/{EPOCHS}')

        loss = None

        for x, vtype, y, raw_y in loop:
            x, vtype, y = x.to(device), vtype.to(device), y.to(device)

            # 模型 预测
            pred = model(x, vtype)

            # 判断是否使用 加权损失函数
            if use_weight_loss:
                last = x[:, -1, :2].cpu().view(-1,1,2)
                pred_norm = last + pred                  # 输入的最后一个时间步 加上 偏移量
                loss = crit(pred_norm, raw_y, stats)
            else:
                loss = crit(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # 实时把 loss 写到进度条右侧
            loop.set_postfix(loss=loss.item())

        print(f'Epoch {epoch:02d} | loss={loss.item():.4f}'
              f'====> curEpochTime {time.time() - curEpoch_start_time:.4f} s ')
        
        #如果 min_loss 为空，且是当前第一次迭代训练，则为 min_loss 初始化赋值
        if not min_loss and epoch == 1:
            min_loss = loss
            torch.save(model, model_path)
            print("模型已进行初始化保存")

       
        if min_loss > loss:
            torch.save(model, model_path)
            print(Fore.RED + " 模型已进行 迭代 保存" + model_path + Style.RESET_ALL)
            min_loss = loss

    print(Fore.GREEN + f'本模型共迭代 {EPOCHS} 次，共耗费时间 {time.time() - start_train :.4f} s'  + Style.RESET_ALL)

