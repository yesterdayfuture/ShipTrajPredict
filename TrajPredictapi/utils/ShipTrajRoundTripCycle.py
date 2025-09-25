#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单文件完成：生成测试 AIS → 分析往返周期 → 可视化
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import folium
import io, datetime, random

# ========== 1. 生成测试 AIS 数据（一条船在 A、B 两港之间来回） ==========
def make_test_ais(mmsi=123456789, days=60, interval_min=5,
                  portA=(31.2, 121.5), portB=(30.8, 122.0),
                  sail_speed_kn=14, port_stay_h=(8, 18)):
    """返回 DataFrame，含 MMSI LAT LON SOG COG timeUnix"""
    records = []
    ts = int((datetime.datetime.utcnow() - datetime.timedelta(days=days)).timestamp())
    end_ts = ts + days*24*3600
    loc = portA
    while ts < end_ts:
        # 在港：SOG≈0，随机停留
        stay_h = random.uniform(*port_stay_h)
        stay_sec = int(stay_h*3600)
        for _ in range(0, stay_sec, interval_min*60):
            records.append([mmsi, loc[0], loc[1],
                            max(0, random.gauss(0.3, 0.2)),
                            random.uniform(0,360), ts])
            ts += interval_min*60
            if ts >= end_ts: break
        if ts >= end_ts: break
        # 出航
        other_port = portB if loc==portA else portA
        dist_nm = geodesic(loc, other_port).nm
        sail_h = dist_nm / sail_speed_kn
        sail_sec = int(sail_h*3600)
        steps = max(1, sail_sec // (interval_min*60))
        lats = np.linspace(loc[0], other_port[0], steps)
        lons = np.linspace(loc[1], other_port[1], steps)
        for i in range(steps):
            records.append([mmsi, lats[i], lons[i],
                            max(0, random.gauss(sail_speed_kn, 1)),
                            random.uniform(0,360), ts])
            ts += interval_min*60
            if ts >= end_ts: break
        loc = other_port
    df = pd.DataFrame(records, columns=['MMSI','LAT','LON','SOG','COG','timeUnix'])
    df['datetime'] = pd.to_datetime(df.timeUnix, unit='s')
    return df

# ========== 2. 往返周期分析函数 ==========
def roundtrip_cycle(df, stop_sog=1.0, stop_min=20, port_eps_km=2.0, min_port_samples=5):
    df = df.sort_values('timeUnix').reset_index(drop=True)
    # 停船块
    df['stop'] = (df.SOG <= stop_sog).astype(int)
    df['block'] = (df.stop.diff()!=0).cumsum()
    stop_summary = (df[df.stop==1]
                    .groupby('block')
                    .agg(t_start=('datetime','min'),
                         t_end  =('datetime','max'),
                         lat    =('LAT','median'),
                         lon    =('LON','median'),
                         n      =('MMSI','count'))
                    .query('n>=@stop_min'))
    # DBSCAN 找港口
    coords = np.radians(stop_summary[['lat','lon']].values)
    db = DBSCAN(eps=port_eps_km/6371.0088, min_samples=min_port_samples, metric='haversine')
    stop_summary['port'] = db.fit_predict(coords)
    top2 = stop_summary.port.value_counts().head(2).index
    stop_summary = stop_summary[stop_summary.port.isin(top2)]
    # 标记每行属于哪个港口
    port_df = stop_summary[['t_start','t_end','lat','lon','port']].copy()
    df['port'] = -1
    for _,r in port_df.iterrows():
        mask = (df.datetime>=r.t_start) & (df.datetime<=r.t_end)
        df.loc[mask, 'port'] = r.port
    # 提取 A→B→A
    changes = df[df.port>=0][['datetime','port']].drop_duplicates()
    changes['prev'] = changes.port.shift(1)
    changes = changes[changes.port != changes.prev].reset_index(drop=True)
    cycles = []
    for i in range(2, len(changes)):
        a,b,c = changes.iloc[i-2:i+1].port.tolist()
        if a==c and a!=b:
            dep = changes.iloc[i-2].datetime
            arr = changes.iloc[i].datetime
            cycles.append({'departure': dep, 'arrival': arr,
                           'cycle_h': (arr-dep).total_seconds()/3600})
    return pd.DataFrame(cycles), stop_summary

# ========== 3. 主流程 ==========
if __name__ == '__main__':
    # 3.1 生成测试数据
    ais = make_test_ais()
    ais.to_csv('test_ais.csv', index=False)
    print('已生成测试 AIS：test_ais.csv  （{} 行）'.format(len(ais)))

    # 3.2 往返周期分析
    cycles, ports = roundtrip_cycle(ais)
    print(cycles)
    print(ports)
    print('\n往返周期样本数：', len(cycles))
    print('平均周期：{:.1f} h  ≈ {:.1f} 天'.format(cycles.cycle_h.mean(), cycles.cycle_h.mean()/24))
    print('标准差：{:.1f} h'.format(cycles.cycle_h.std()))

    # # 3.3 分布图
    # plt.figure()
    # sns.histplot(cycles.cycle_h, bins=15, kde=True)
    # plt.xlabel('Round-trip period (h)')
    # plt.title('Distribution of round-trip cycle')
    # plt.tight_layout()
    # plt.savefig('cycle_dist.png')
    # print('已保存周期分布图：cycle_dist.png')
    #
    # # 3.4 港口地图
    # m = folium.Map(location=[ais.LAT.mean(), ais.LON.mean()], zoom_start=8)
    # folium.PolyLine(ais[['LAT','LON']].values, color='gray', weight=1.5, opacity=0.4).add_to(m)
    # for _,r in ports.iterrows():
    #     folium.CircleMarker(location=(r.lat, r.lon), radius=10,
    #                         popup='Port {}'.format(int(r.port)),
    #                         color='red', fill=True).add_to(m)
    # m.save('ports_map.html')
    # print('已保存港口地图：ports_map.html')