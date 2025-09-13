
'''
本文件将输入的多条船舶轨迹进行两两计算，判断是否共现

'''

import pandas as pd
import numpy as np
import h3
from geopy.distance import geodesic
from multiprocessing import Pool
from itertools import combinations
from datetime import timedelta
import json
from .loadEnv import cfg




# ---------------- 1. 模拟数据 ----------------
def fake_fleet(n_ship=20, n_point=100):
    np.random.seed(42)
    dfs = []
    base = pd.Timestamp('2025-08-26 08:00')
    for mmsi in range(n_ship):
        ts = pd.date_range(base, periods=n_point, freq='1min')
        lat = 31.230 + np.cumsum(np.random.randn(n_point) * 0.0002)
        lon = 121.473 + np.cumsum(np.random.randn(n_point) * 0.0002)
        dfs.append(pd.DataFrame({'MMSI': mmsi, 'timeUnix': ts, 'LAT': lat, 'LON': lon}))
    return pd.concat(dfs).reset_index(drop=True)

# ---------------- 2. 一级剪枝：H3 网格 ----------------
def add_h3(df, res=6):
    df['cell'] = df.apply(lambda r: h3.latlng_to_cell(r.LAT, r.LON, res), axis=1)
    return df

def candidate_pairs(df):
    """返回 (mmsi_A, mmsi_B) 列表；只要两船在同一 cell 出现过即保留"""
    g = df.groupby('cell')['MMSI'].unique()
    pairs = set()
    for mmsis in g:
        for a, b in combinations(sorted(mmsis), 2):
            pairs.add((a, b))
    return list(pairs)


# ---------------- 3. 单对船舶共现 ----------------
def co_occurrence(dfA, dfB):
    """返回 DataFrame，每行一次共现片段"""
    df = pd.merge_asof(
        dfA.sort_values('timeUnix').rename(columns={'LAT': 'LAT1', 'LON': 'LON1'}),
        dfB.sort_values('timeUnix').rename(columns={'LAT': 'LAT2', 'LON': 'LON2'}),
        on='timeUnix', direction='nearest', tolerance=pd.Timedelta(cfg['time_tol'])
    ).dropna(subset=['LAT2'])




    # 计算 两个经纬度之间的距离（KM）
    def dist(r):
        return geodesic((r.LAT1, r.LON1), (r.LAT2, r.LON2)).km
    
    df['dist_km'] = df.apply(dist, axis=1)

    # 判断 两个经纬度之间的距离（KM）是否 小于 某个阈值
    df['co_flag'] = df['dist_km'] <= cfg['D_km']

    # 连续片段
    df['grp'] = (df['co_flag'] != df['co_flag'].shift()).cumsum()

    co = (df[df['co_flag']]
          .groupby('grp')
          .agg(start=('timeUnix', 'min'), end=('timeUnix', 'max')))
    
    co['duration'] = co['end'] - co['start']
    co = co[co['duration'] >= pd.Timedelta(cfg['min_dur'])]
    
    # 判断 co DataFrame 是否 为空，为空直接返回
    if co.empty:
        return co

    def judge_type(r):
        return '补给' if r['duration'] >= pd.Timedelta(cfg['threshold']) else '协同'

    co['type'] = co.apply(judge_type,axis=1)

    return co.reset_index(drop=True)


# 4-3 并行计算
def worker(pair, df_all):
    a, b = pair
    co = co_occurrence(
        df_all[df_all.MMSI == a],
        df_all[df_all.MMSI == b]
    )

    if co.empty:
        return co

    co['mmsi_A'] = a
    co['mmsi_B'] = b
    return co

# ---------------- 4. 并行主流程 ----------------
def CalcCoTraj_main(df_all):

    '''
    df 要包含 MMSI、timeUnix、LAT、LON 这些列
    '''

    # 4-1 加网格
    df_all = add_h3(df_all, cfg['cell_res'])

    # 4-2 候选船舶对
    pairs = candidate_pairs(df_all)


    with Pool(cfg['n_workers']) as p:
        chunks = p.starmap(worker, [(pair, df_all) for pair in pairs], chunksize=1)

    # 4-4 汇总
    result = pd.concat(chunks, ignore_index=True)

    if result.empty:
        return result
    return result.sort_values(['start', 'mmsi_A', 'mmsi_B'])

# # ---------------- 运行示例 ----------------
# if __name__ == '__main__':
#     df_all = fake_fleet(n_ship=20, n_point=200)
#     co_df = CalcCoTraj_main(df_all)
#     co_df = co_df.reset_index(drop=True)
#     print(co_df.head())