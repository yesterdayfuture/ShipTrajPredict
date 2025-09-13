import sys
import os
# 将父目录添加到Python路径
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.getcwd())

from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
from kimi.predic import predict_main
from kimi.train import SEQ_LEN, HORIZON
from colorama import Fore, Style
from kimi.model import *

# 调用 ../kimi文件夹 下的 predict.py 文件，对输入数据进行预测
def UtilModelPredic_main(df:DataFrame):
    df_group = df.groupby('MMSI')

    df_group_MMSI_list = list(df_group.groups.keys())

    print(f'{Fore.RED}当前输入数据共有 {len(df_group_MMSI_list)} 组{Style.RESET_ALL}')

    # 保存 所有 预测结果
    all_TrajResult = []
    # 长度符合的 组的 MMSI
    mmsi_yes = []
    # 长度不符合的 组的 MMSI
    mmsi_no = []

    for mmsi in df_group_MMSI_list:
        
        # 保存 当前预测结果的 字典
        cur_dict = {}

        # 获取当前组 数据
        cur_df = df_group.get_group(mmsi) 

        cur_df.loc[:,'VType'] = 0

        # 当前 组的数据 少于 SEQ_LEN, 无法进行预测
        if cur_df.shape[0] < SEQ_LEN:
            mmsi_no.append(mmsi)
            continue

        mmsi_yes.append(mmsi)

        # 目前 预测 未来 HORIZON 步的经纬度
        cur_TrajResult = predict_main(cur_df)
        
        cur_dict['MMSI'] = [mmsi]*HORIZON
        cur_dict['LAT'] = cur_TrajResult[:,0].tolist()
        cur_dict['LON'] = cur_TrajResult[:,1].tolist()


        # 将 本次预测结果 放到 最终结果中
        all_TrajResult.append(cur_dict)
        

    # 合并为一个字典
    merged = {}
    for i, d in enumerate(all_TrajResult):
        for key, value in d.items():
            if key not in merged:
                merged[key] = []
            merged[key].extend(value)

    return pd.DataFrame(merged)



# # 测试 预测函数
# if __name__ == "__main__":
    
#     df = pd.read_csv(r'data/AIS_2023_12_28_test.csv')
#     df['VType'] = 0
#     df_result = UtilModelPredic_main(df)
#     print(df_result.head())


