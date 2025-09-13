from dotenv import load_dotenv
import os
import json
import sys

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 加载 本文件夹下的 .env
load_dotenv()

cfg_path = os.getenv('CoTrajConfigFilePath')
print(f'配置文件地址为 {cfg_path}')

# ---------------- 参数 ----------------
# cfg = {
#     'D_km': 0.5,
#     'min_dur': '2min',
#     'time_tol': '30s',
#     'cell_res': 6,
#     'n_workers': 8,
#     'max_span': '12h',
#     'threshold': '12h'
# }

with open(cfg_path,'r') as f:
    cfg = json.load(f)

print(f'轨迹共现 配置文件内容为： {cfg}')

with open( os.getenv('TrajConfigActivityFilePath'),'r') as f:
    activityCfg = json.load(f)

print(f'轨迹活动场景 配置文件内容为： {activityCfg}')