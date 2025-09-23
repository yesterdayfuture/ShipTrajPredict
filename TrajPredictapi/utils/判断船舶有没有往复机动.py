
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from shapely.geometry import LineString

def detect_reciprocating_movement(ais_data):
    """
    检测船舶往复机动
    :param ais_data: DataFrame需包含列['timestamp','speed','course','lon','lat']
    :return: (bool, DataFrame) 是否往复机动, 异常轨迹点
    """
    # 预处理
    ais_data = ais_data.sort_values('timestamp').reset_index(drop=True)
    ais_data['delta_speed'] = ais_data['speed'].diff().abs()
    ais_data['delta_course'] = ais_data['course'].diff().abs()
    
    # 航速变化检测（速度正负交替）
    speed_changes = ((ais_data['speed'] * ais_data['speed'].shift(1)) < 0).sum()
    
    # 航向突变检测（30秒内航向变化>90度）
    course_changes = (ais_data['delta_course'].rolling(window=3).max() > 90).any()
    
    # 轨迹压缩后检测局部密度（Douglas-Peucker简化）
    coords = ais_data[['lon','lat']].values
    line = LineString(coords)
    simplified = line.simplify(tolerance=0.0001)
    density_ratio = len(coords) / max(1, len(simplified.coords))
    
    # 综合判断（满足任意两个条件即视为往复机动）
    conditions_met = sum([speed_changes>=2, course_changes, density_ratio>5])
    anomalies = ais_data[(ais_data['delta_course']>90) | (ais_data['delta_speed']>2)]
    
    return conditions_met>=2, anomalies




from geopy.distance import geodesic
# 计算 两个经纬度 之间的距离
def calcTwoPointDistance(a,b):

    d = geodesic(a, b).km
    return d