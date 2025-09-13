'''
本文件 根据输入的经纬度 计算 速度 与 航向
'''


import pandas as pd
from geopy.distance import Geodesic
from pandas.core.frame import DataFrame

def add_speed_heading_geopy(data:DataFrame, speed_unit='km/h'):
    """
    df 必须含 MMSI, LAT, LON, timeUnix（pd.Timestamp）
    返回新增 SOG, COG 两列
    """

    # 根据 MMSI 进行分组
    df_group = data.groupby('MMSI')

    # 将 所有组的组名（MMSI） 变成一个 列表
    df_group_MMSI_list = list(df_group.groups.keys())

    # 将 每组 处理后的 DataFrame 放进一个list
    df_group_list = []

    for cur_MMSI in df_group_MMSI_list:
        
        df = df_group.get_group(cur_MMSI)

        df = df.sort_values('timeUnix')

        # 移前一行坐标
        prev = df.shift(1)

        # 计算 distance & bearing（一次性矢量计算）
        def calc(row):
            if pd.isna(row.lat_prev):
                return pd.Series({'dist_km': None, 'heading': None})
            g = Geodesic.WGS84.Inverse(row.lat_prev, row.lon_prev,
                                    row.LAT, row.LON)
            dist_km = g['s12'] / 1000.0        # 米 -> 公里
            heading = g['azi1'] % 360
            return pd.Series({'dist_km': dist_km, 'heading': heading})

        tmp = df.assign(lat_prev=prev['LAT'],
                        lon_prev=prev['LON']).apply(calc, axis=1)
        dist_km = tmp['dist_km']
        heading = tmp['heading']

        # 时间间隔（小时）
        delta_h = (df['timeUnix'] - prev['timeUnix']).dt.total_seconds() / 3600.0

        # 速度
        if speed_unit == 'knots':
            df['SOG'] = (dist_km * 0.539957) / delta_h
        else:
            df['SOG'] = dist_km / delta_h

        df['COG'] = heading
        df = df.fillna(method='bfill')

        df_group_list.append(df)

    # 将多个 DataFrame 进行 合并
    df_all = pd.concat(df_group_list)
    df_all = df_all.reset_index(drop=True)

    return df_all