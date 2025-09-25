# '''
#
# 根据输入的 ais 数据，生成典型的轨迹（关键路径）
# pip install pandas numpy geopandas scikit-learn shapely folium seaborn
# '''
#
# import pandas as pd, geopandas as gpd, shapely.geometry as geom
# from shapely.geometry import Point, LineString
# from pandas.core.frame import DataFrame
# from shapely.ops import unary_union
# import numpy as np
# from sklearn.cluster import DBSCAN
#
# def getTypicalTraj(df:DataFrame):
#     gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.LON, df.LAT)])
#     gdf = gdf.set_crs(4326).to_crs(3857)          # Web墨卡托，单位米
#
#     print(f" gdf = {gdf}")
#
#     # 2.1 去重、去野点
#     gdf = gdf.drop_duplicates(subset=['timeUnix'])
#     gdf = gdf[gdf.SOG <= 30]  # 超速异常
#
#     # 使用凸包中心替代几何中位数
#     centroid = gdf.geometry.convex_hull.centroid
#     gdf = gdf[gdf.geometry.distance(centroid) < 50000]
#
#     # gdf = gdf[gdf.geometry.distance(gdf.geometry.median()) < 50000]  # 50 km 离群
#
#     # 2.2 按航次（voyage）切片：停船 >2 h 就分段
#     gdf = gdf.sort_values('timeUnix')
#     gdf['dt'] = gdf.timeUnix.diff().dt.total_seconds() / 3600
#
#     print(f" gdf = {gdf}")
#
#     gdf['voy_id'] = (gdf['dt'] > 2).cumsum()
#
#     voyages = [gdf[gdf.voy_id == vid] for vid in gdf.voy_id.unique()]
#
#     print(f"voyages: {voyages}")
#
#     # 轨迹压缩（Douglas-Peucker）
#     # 减少 90 % 点，保留几何形状，后续聚类飞快
#     def dp_compress(voy, eps=500):  # 500 m 容忍
#         line = LineString(voy.geometry.tolist())
#         return line.simplify(eps, preserve_topology=False)
#
#     simple_lines = [dp_compress(v) for v in voyages]
#
#     print(f"simple_lines: {simple_lines}")
#
#
#     #把“所有航次”变成一张图 → 聚类走廊
#     # 生成等距“小段”
#     # 用 DBSCAN 聚类小段 → 同一走廊的小段聚一类
#     # 每类拟合中心线
#     segments, centers = [], []
#     for line in simple_lines:
#         if line.length < 1000:
#             continue
#         n = int(line.length / 500)  # 500 m 一段
#         pts = [line.interpolate(i / n, normalized=True) for i in range(n)]
#         for p in pts:
#             segments.append([p.x, p.y])
#     segments = np.array(segments)
#
#     print(f"segments: {segments}")
#     # 添加调试打印语句
#     print(f"Segments shape: {segments.shape}")  # 预期输出应为(n_samples, 2)
#     print(f"Segments dtype: {segments.dtype}")  # 确认数值类型
#     print(f"Segments sample: {segments[:5]}")  # 检查前5个样本
#
#     # DBSCAN 聚类
#     clust = DBSCAN(eps=1500, min_samples=20).fit(segments)
#     labels = clust.labels_
#
#     # 每类求中心线
#     center_lines = {}
#     for l in np.unique(labels):
#         if l == -1: continue
#         pts = segments[labels == l]
#         center_lines[l] = LineString(pts).simplify(1000)
#
#
#     # 把同一走廊的小段再挂上原始速度，统计分位数。
#     speed_df = []
#     for line in simple_lines:
#         n = int(line.length / 500)
#         for i in range(n):
#             p = line.interpolate(i / n, normalized=True)
#             # 找原始点里最近 1 点
#             dist = gdf.distance(p)
#             idx = dist.idxmin()
#             speed_df.append({'x': p.x, 'y': p.y, 'sog': gdf.loc[idx].sog,
#                              'label': clust.predict([[p.x, p.y]])[0]})
#     speed_df = pd.DataFrame(speed_df)
#
#     sig = (speed_df.groupby('label')['sog']
#            .agg(['median', 'q25', 'q75'])
#            .rename(columns={'median': 'v50', 'q25': 'v25', 'q75': 'v75'}))
#
#     print(f" sig = {sig}")


# import pandas as pd
# import numpy as np
# import geopandas as gpd
# from shapely.geometry import Point, LineString
# from datetime import datetime, timedelta
# import random
#
#
# def generate_test_data():
#     """生成测试用的船舶轨迹数据"""
#     np.random.seed(42)  # 保证结果可重现
#
#     # 生成基础数据
#     n_points = 200
#     base_time = datetime(2024, 1, 1, 0, 0, 0)
#
#     # 创建两条主要航线（模拟真实船舶轨迹）
#     # 航线1：从上海到东京
#     route1_lons = np.linspace(121.47, 139.69, n_points // 2)
#     route1_lats = np.linspace(31.23, 35.68, n_points // 2)
#
#     # 航线2：从香港到新加坡
#     route2_lons = np.linspace(114.16, 103.82, n_points // 2)
#     route2_lats = np.linspace(22.28, 1.35, n_points // 2)
#
#     # 合并两条航线
#     all_lons = np.concatenate([route1_lons, route2_lons])
#     all_lats = np.concatenate([route1_lats, route2_lats])
#
#     # 添加一些噪声使轨迹更真实
#     all_lons += np.random.normal(0, 0.1, n_points)
#     all_lats += np.random.normal(0, 0.05, n_points)
#
#     # 生成时间序列（模拟真实的时间间隔）
#     time_unix = []
#     current_time = base_time
#     for i in range(n_points):
#         time_unix.append(current_time)
#         # 模拟不规则的时间间隔（10分钟到2小时）
#         interval = timedelta(minutes=random.randint(10, 120))
#         current_time += interval
#
#     # 生成速度数据（SOG）
#     sog_values = []
#     for i in range(n_points):
#         # 大部分时间在10-25节之间，偶尔有低速或高速
#         if i % 50 == 0:  # 模拟停泊
#             sog_values.append(random.uniform(0, 2))
#         elif i % 20 == 0:  # 模拟高速
#             sog_values.append(random.uniform(25, 30))
#         else:  # 正常航行
#             sog_values.append(random.uniform(10, 25))
#
#     # 创建DataFrame
#     df = pd.DataFrame({
#         'timeUnix': time_unix,
#         'LON': all_lons,
#         'LAT': all_lats,
#         'SOG': sog_values,
#         'voyage_id': ['voyage_1'] * (n_points // 2) + ['voyage_2'] * (n_points // 2)
#     })
#
#     # 添加一些重复的时间点（测试去重功能）
#     df = pd.concat([df, df.iloc[[10, 50, 100]]], ignore_index=True)
#
#     # 添加一些异常点（测试离群点检测）
#     outlier_df = pd.DataFrame({
#         'timeUnix': [base_time + timedelta(days=1)],
#         'LON': [150.0],  # 远离正常轨迹
#         'LAT': [40.0],
#         'SOG': [35.0],  # 超速
#         'voyage_id': ['outlier']
#     })
#
#     df = pd.concat([df, outlier_df], ignore_index=True)
#
#     return df
#
#
# def test_pipeline():
#     """测试完整的处理流程"""
#     # 1. 生成测试数据
#     df = generate_test_data()
#     print(f"原始数据形状: {df.shape}")
#     print(f"数据预览:\n{df.head()}")
#
#     # 2. 创建GeoDataFrame（你的原始代码）
#     gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.LON, df.LAT)])
#     gdf = gdf.set_crs(4326).to_crs(3857)  # Web墨卡托，单位米
#
#     print(f"转换后的gdf形状: {gdf.shape}")
#     print(f" gdf.head() = {gdf.head()}")
#
#     # 3. 数据清洗（你的原始代码）
#     # 2.1 去重、去野点
#     gdf = gdf.drop_duplicates(subset=['timeUnix'])
#     gdf = gdf[gdf.SOG <= 30]  # 超速异常
#
#     print(f"去重去异常后的gdf形状: {gdf.shape}")
#
#     # 使用凸包中心替代几何中位数
#     centroid = gdf.geometry.convex_hull.centroid
#     # print(f" centroid = {centroid}")
#     gdf = gdf[gdf.geometry.distance(centroid) < 50000]  # 50 km 离群
#
#     print(f"去除离群点后的gdf形状: {gdf.shape}")
#     print(f" gdf.head() = {gdf.head()}")
#
#     # 2.2 按航次（voyage）切片：停船 >2 h 就分段
#     gdf = gdf.sort_values('timeUnix')
#     gdf['dt'] = (gdf.timeUnix.diff().dt.total_seconds() / 3600).fillna(0)
#
#     # 创建航次ID（基于时间间隔）
#     gdf['voy_id'] = (gdf['dt'] > 2).cumsum()
#     print(f" gdf.head() = {gdf.head()}")
#
#     voyages = [gdf[gdf.voy_id == vid] for vid in gdf.voy_id.unique()]
#     print(f"分割后的航次数: {len(voyages)}")
#     print(f" voyages = {voyages}")
#
#     # 轨迹压缩（Douglas-Peucker）
#     def dp_compress(voy, eps=500):  # 500 m 容忍
#         if len(voy) < 2:
#             return LineString([])
#         line = LineString(voy.geometry.tolist())
#         return line.simplify(eps, preserve_topology=False)
#
#     simple_lines = [dp_compress(v) for v in voyages if len(v) >= 2]
#     simple_lines = [line for line in simple_lines if not line.is_empty]
#
#     print(f"压缩后的轨迹数: {len(simple_lines)}")
#
#     # 生成等距小段用于聚类
#     segments, centers = [], []
#     for line in simple_lines:
#         if line.length < 1000:
#             continue
#         n = max(2, int(line.length / 500))  # 500 m 一段
#         pts = [line.interpolate(i / n, normalized=True) for i in range(n)]
#         for p in pts:
#             segments.append([p.x, p.y])
#
#     if len(segments) == 0:
#         print("没有足够长的轨迹用于聚类")
#         return
#
#     segments = np.array(segments)
#     print(f"生成的小段数: {len(segments)}")
#     print(f"Segments shape: {segments.shape}")
#
#     # DBSCAN 聚类（需要sklearn）
#     try:
#         from sklearn.cluster import DBSCAN
#         clust = DBSCAN(eps=1500, min_samples=5).fit(segments)  # 调整参数适应测试数据
#         labels = clust.labels_
#
#         # 每类求中心线
#         center_lines = {}
#         for l in np.unique(labels):
#             if l == -1:
#                 continue
#             pts = segments[labels == l]
#             if len(pts) >= 2:  # 至少需要两个点来创建线
#                 center_lines[l] = LineString(pts).simplify(1000)
#
#         print(f"发现的走廊数: {len(center_lines)}")
#
#         # 速度统计分析
#         speed_data = []
#         for line in simple_lines:
#             if line.length < 1000:
#                 continue
#             n = max(2, int(line.length / 500))
#             for i in range(n):
#                 p = line.interpolate(i / n, normalized=True)
#                 # 找原始点里最近的点
#                 distances = gdf.geometry.distance(p)
#                 if len(distances) > 0:
#                     idx = distances.idxmin()
#                     try:
#                         label = clust.predict([[p.x, p.y]])[0]
#                         speed_data.append({
#                             'x': p.x, 'y': p.y,
#                             'sog': gdf.loc[idx].SOG,
#                             'label': label
#                         })
#                     except:
#                         continue
#
#         if speed_data:
#             speed_df = pd.DataFrame(speed_data)
#             sig = (speed_df.groupby('label')['sog']
#                    .agg(['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
#                    .rename(columns={'median': 'v50', '<lambda_0>': 'v25', '<lambda_1>': 'v75'}))
#
#             print(f"速度统计:\n{sig}")
#         else:
#             print("没有足够的速度数据进行统计")
#
#     except ImportError:
#         print("sklearn未安装，跳过聚类步骤")
#
#     return gdf, simple_lines
#
#
# if __name__ == "__main__":
#     # 运行测试
#     gdf, simple_lines = test_pipeline()
#     print(f" gdf.head() = {gdf.head()}")
#     print(f" simple_lines = {simple_lines}")
#
#     # 保存测试数据供后续使用
#     df = generate_test_data()
#     df.to_csv('test_ship_data.csv', index=False)
#     print("测试数据已保存到 test_ship_data.csv")


import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from datetime import datetime, timedelta
import random
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

def generate_test_data():
    """生成测试用的船舶轨迹数据"""
    np.random.seed(42)  # 保证结果可重现

    # 生成基础数据
    n_points = 200
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    # 创建两条主要航线（模拟真实船舶轨迹）
    # 航线1：从上海到东京
    route1_lons = np.linspace(121.47, 139.69, n_points // 2)
    route1_lats = np.linspace(31.23, 35.68, n_points // 2)

    # 航线2：从香港到新加坡
    route2_lons = np.linspace(114.16, 103.82, n_points // 2)
    route2_lats = np.linspace(22.28, 1.35, n_points // 2)

    # 合并两条航线
    all_lons = np.concatenate([route1_lons, route2_lons])
    all_lats = np.concatenate([route1_lats, route2_lats])

    # 添加一些噪声使轨迹更真实
    all_lons += np.random.normal(0, 0.1, n_points)
    all_lats += np.random.normal(0, 0.05, n_points)

    # 生成时间序列（模拟真实的时间间隔）
    time_unix = []
    current_time = base_time
    for i in range(n_points):
        time_unix.append(current_time)
        # 模拟不规则的时间间隔（10分钟到2小时）
        interval = timedelta(minutes=random.randint(10, 120))
        current_time += interval

    # 生成速度数据（SOG） - 添加更多的停泊段
    sog_values = []
    for i in range(n_points):
        # 模拟停泊段（低速持续一段时间）
        if 20 <= i < 30:  # 第一个停泊段
            sog_values.append(random.uniform(0, 1))
        elif 70 <= i < 80:  # 第二个停泊段
            sog_values.append(random.uniform(0, 1))
        elif 140 <= i < 150:  # 第三个停泊段
            sog_values.append(random.uniform(0, 1))
        elif i % 20 == 0:  # 模拟高速
            sog_values.append(random.uniform(25, 30))
        else:  # 正常航行
            sog_values.append(random.uniform(10, 25))

    # 创建DataFrame
    df = pd.DataFrame({
        'timeUnix': time_unix,
        'LON': all_lons,
        'LAT': all_lats,
        'SOG': sog_values,
        'voyage_id': ['voyage_1'] * (n_points // 2) + ['voyage_2'] * (n_points // 2)
    })

    # 添加一些重复的时间点（测试去重功能）
    df = pd.concat([df, df.iloc[[10, 50, 100]]], ignore_index=True)

    return df


def segment_by_speed_duration(gdf, speed_threshold=2, duration_threshold=2):
    """
    使用速度阈值+持续时间法进行轨迹分段

    参数:
    - gdf: GeoDataFrame，包含轨迹点数据
    - speed_threshold: 速度阈值（节），低于此值认为船舶停泊
    - duration_threshold: 持续时间阈值（小时），停泊超过此时间则分段

    返回:
    - 分段后的轨迹列表，每个元素是一个GeoDataFrame
    """
    # 确保按时间排序
    gdf = gdf.sort_values('timeUnix')
    # 按时间戳对数据进行排序，确保轨迹点按时间顺序排列

    # 标记低速点
    gdf['low_speed'] = gdf['SOG'] < speed_threshold
    # 创建一个新列'low_speed'，标记速度低于阈值的点（True表示低速，False表示正常速度）

    # 计算时间差（小时）
    gdf['time_diff'] = gdf['timeUnix'].diff().dt.total_seconds() / 3600
    # 计算相邻点之间的时间差（单位：小时）
    # diff()计算当前行与前一行的时间差
    # dt.total_seconds()将时间差转换为秒数
    # /3600将秒转换为小时

    gdf['time_diff'] = gdf['time_diff'].fillna(0)
    # 将第一行的NaN值（因为没有前一行）填充为0

    # 识别停泊段
    gdf['stop_segment'] = 0
    # 创建一个新列'stop_segment'，初始化为0，用于标记停泊段的ID

    stop_id = 0
    # 初始化停泊段ID计数器

    in_stop = False
    # 标志变量，表示当前是否处于停泊段中

    stop_start_idx = None
    # 记录当前停泊段的起始索引

    # 遍历所有数据点
    for i in range(len(gdf)):
        # 检查当前点是否为低速点
        if gdf.iloc[i]['low_speed']:
            # 如果当前不是处于停泊段，则开始新的停泊段
            if not in_stop:
                # 开始新的停泊段
                in_stop = True  # 设置标志为"处于停泊段"
                stop_start_idx = i  # 记录停泊段起始索引
                stop_id += 1  # 停泊段ID加1
            # 将当前点标记为当前停泊段ID
            gdf.iloc[i, gdf.columns.get_loc('stop_segment')] = stop_id
        else:
            # 当前点不是低速点
            if in_stop:
                # 如果之前处于停泊段，则现在结束停泊段
                in_stop = False  # 设置标志为"不处于停泊段"
                # 计算停泊持续时间
                stop_duration = (gdf.iloc[i - 1]['timeUnix'] - gdf.iloc[stop_start_idx][
                    'timeUnix']).total_seconds() / 3600
                # 计算从停泊段开始到结束的时间差（小时）

                # 如果停泊时间不足阈值，则取消该停泊段标记
                if stop_duration < duration_threshold:
                    # 将这一停泊段的所有标记重置为0（不视为有效停泊段）
                    gdf.loc[gdf['stop_segment'] == stop_id, 'stop_segment'] = 0

            # 将非低速点标记为0（不属于任何停泊段）
            gdf.iloc[i, gdf.columns.get_loc('stop_segment')] = 0

    # 处理最后一个点可能是停泊点的情况
    if in_stop:
        # 如果遍历结束后仍处于停泊段，计算整个停泊段的持续时间
        stop_duration = (gdf.iloc[-1]['timeUnix'] - gdf.iloc[stop_start_idx]['timeUnix']).total_seconds() / 3600
        # 如果持续时间不足阈值，取消该停泊段标记
        if stop_duration < duration_threshold:
            gdf.loc[gdf['stop_segment'] == stop_id, 'stop_segment'] = 0

    # 根据停泊段进行轨迹分段
    segments = []  # 初始化分段结果列表
    current_segment = []  # 当前正在构建的段的索引列表
    current_voyage_id = 0  # 当前航次ID

    # 再次遍历所有数据点，进行分段
    for i in range(len(gdf)):
        current_segment.append(i)  # 将当前点索引添加到当前段

        # 检查是否需要分段（不是最后一个点时）
        if i < len(gdf) - 1:
            # 如果下一个点属于不同的停泊段（且不是0），则分段
            if gdf.iloc[i]['stop_segment'] != gdf.iloc[i + 1]['stop_segment'] and gdf.iloc[i + 1]['stop_segment'] != 0:
                # 当前点与下一个点的停泊段ID不同，且下一个点属于有效停泊段
                if len(current_segment) > 1:  # 确保段内有足够点（至少2个）
                    # 将当前段添加到分段结果中
                    segments.append(gdf.iloc[current_segment].copy())
                    current_segment = []  # 重置当前段
                    current_voyage_id += 1  # 航次ID加1

    # 添加最后一个段（遍历结束后剩余的段）
    if len(current_segment) > 1:  # 确保段内有足够点
        segments.append(gdf.iloc[current_segment].copy())

    # 为每个段分配航次ID
    for i, segment in enumerate(segments):
        segment['voy_id'] = i  # 为每个段添加航次ID列

    return segments  # 返回分段结果

def extract_typical_trajectory_points(segments, n_points=10):
    """
    从每个轨迹段中提取典型轨迹点

    参数:
    - segments: 轨迹段列表
    - n_points: 每个轨迹段提取的点数

    返回:
    - 典型轨迹点字典，键为航次ID，值为经纬度点列表
    """
    typical_points = {}

    for segment in segments:
        voy_id = segment['voy_id'].iloc[0]

        # 如果轨迹点太少，直接使用所有点
        if len(segment) <= n_points:
            points = [(row['LON'], row['LAT']) for _, row in segment.iterrows()]
        else:
            # 等间隔采样
            indices = np.linspace(0, len(segment) - 1, n_points, dtype=int)
            points = [(segment.iloc[i]['LON'], segment.iloc[i]['LAT']) for i in indices]

        typical_points[voy_id] = points

    return typical_points


def visualize_trajectories(gdf, segments, typical_points):
    """可视化轨迹数据"""

    # 创建子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('船舶轨迹分析与可视化', fontsize=16, fontweight='bold')

    # 1. 原始轨迹可视化
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(gdf['LON'], gdf['LAT'], c=gdf['SOG'], cmap='viridis',
                           s=30, alpha=0.7)
    ax1.set_title('原始轨迹点（颜色表示速度）')
    ax1.set_xlabel('经度')
    ax1.set_ylabel('纬度')
    plt.colorbar(scatter1, ax=ax1, label='速度 (节)')

    # 2. 分段轨迹可视化
    ax2 = axes[0, 1]
    colors = list(mcolors.TABLEAU_COLORS.values())
    for i, segment in enumerate(segments):
        color = colors[i % len(colors)]
        ax2.plot(segment['LON'], segment['LAT'], 'o-', color=color,
                 markersize=4, linewidth=2, label=f'航次 {i}')
    ax2.set_title('分段轨迹（不同颜色表示不同航次）')
    ax2.set_xlabel('经度')
    ax2.set_ylabel('纬度')
    ax2.legend()

    # 3. 典型轨迹点可视化
    ax3 = axes[1, 0]
    for voy_id, points in typical_points.items():
        color = colors[voy_id % len(colors)]
        lons, lats = zip(*points)
        ax3.plot(lons, lats, 's-', color=color, markersize=6,
                 linewidth=2, label=f'航次 {voy_id} 典型点')
    ax3.set_title('典型轨迹点')
    ax3.set_xlabel('经度')
    ax3.set_ylabel('纬度')
    ax3.legend()

    # 4. 速度时间序列可视化
    ax4 = axes[1, 1]
    for i, segment in enumerate(segments):
        color = colors[i % len(colors)]
        ax4.plot(segment['timeUnix'], segment['SOG'], 'o-', color=color,
                 markersize=3, linewidth=1, label=f'航次 {i}')
    ax4.set_title('速度时间序列')
    ax4.set_xlabel('时间')
    ax4.set_ylabel('速度 (节)')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('trajectory_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # 5. 单独绘制轨迹分段详情图
    plt.figure(figsize=(12, 8))
    for i, segment in enumerate(segments):
        color = colors[i % len(colors)]
        # 绘制轨迹线
        plt.plot(segment['LON'], segment['LAT'], 'o-', color=color,
                 markersize=5, linewidth=2, label=f'航次 {i}')

        # 标记起点和终点
        start_point = segment.iloc[0]
        end_point = segment.iloc[-1]
        plt.plot(start_point['LON'], start_point['LAT'], '^', color=color,
                 markersize=10, markeredgecolor='black')
        plt.plot(end_point['LON'], end_point['LAT'], 's', color=color,
                 markersize=10, markeredgecolor='black')

        # 添加航次标签
        mid_idx = len(segment) // 2
        mid_point = segment.iloc[mid_idx]
        plt.annotate(f'航次 {i}',
                     xy=(mid_point['LON'], mid_point['LAT']),
                     xytext=(5, 5), textcoords='offset points',
                     bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.7),
                     fontsize=9)

    plt.title('船舶轨迹分段详情', fontsize=14, fontweight='bold')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('trajectory_segments.png', dpi=300, bbox_inches='tight')
    # plt.show()


def test_pipeline():
    """测试完整的处理流程"""
    # 1. 生成测试数据
    df = generate_test_data()
    print(f"原始数据形状: {df.shape}")
    print(f"数据预览:\n{df.head()}")

    # 使用第一条船舶的轨迹进行分析
    df = df.groupby(by=['voyage_id']).get_group("voyage_1")

    # 2. 创建GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.LON, df.LAT)])
    gdf = gdf.set_crs(4326)  # 使用WGS84坐标系

    print(f"转换后的gdf形状: {gdf.shape}")

    # 3. 数据清洗
    gdf = gdf.drop_duplicates(subset=['timeUnix'])
    gdf = gdf[gdf.SOG <= 30]  # 超速异常

    print(f"去重去异常后的gdf形状: {gdf.shape}")

    # 4. 使用速度阈值+持续时间法进行分段
    segments = segment_by_speed_duration(gdf, speed_threshold=2, duration_threshold=2)
    print(f"分段后的航次数: {len(segments)}")
    # print(segments[0])

    # 5. 提取典型轨迹点
    typical_points = extract_typical_trajectory_points(segments, n_points=10)

    # 6. 输出典型轨迹点
    print("\n=== 典型轨迹点 ===")
    for voy_id, points in typical_points.items():
        print(f"\n航次 {voy_id} 的典型轨迹点:")
        for i, (lon, lat) in enumerate(points):
            print(f"  点 {i + 1}: 经度={lon:.4f}, 纬度={lat:.4f}")

    # 7. 可视化轨迹段（可选）
    print("\n=== 轨迹段统计 ===")
    for i, segment in enumerate(segments):
        print(f"航次 {i}: {len(segment)} 个点, "
              f"持续时间: {(segment['timeUnix'].max() - segment['timeUnix'].min()).total_seconds() / 3600:.2f} 小时, "
              f"平均速度: {segment['SOG'].mean():.2f} 节")

    # 8. 可视化
    visualize_trajectories(gdf, segments, typical_points)

    return gdf, segments, typical_points


if __name__ == "__main__":
    # 运行测试
    gdf, segments, typical_points = test_pipeline()

    # 保存测试数据
    df = generate_test_data()
    df.to_csv('test_ship_data.csv', index=False)
    print("\n测试数据已保存到 test_ship_data.csv")

    # 保存典型轨迹点
    with open('typical_trajectory_points.txt', 'w') as f:
        f.write("典型轨迹点\n")
        f.write("==========\n\n")
        for voy_id, points in typical_points.items():
            f.write(f"航次 {voy_id}:\n")
            for i, (lon, lat) in enumerate(points):
                f.write(f"  点 {i + 1}: 经度={lon:.4f}, 纬度={lat:.4f}\n")
            f.write("\n")
    print("典型轨迹点已保存到 typical_trajectory_points.txt")