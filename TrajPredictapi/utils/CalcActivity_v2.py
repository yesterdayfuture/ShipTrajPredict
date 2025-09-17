'''
判断一个经纬度点是否在一个由经纬度点围成的封闭多边形内（支持任意形状、支持有孔洞）。
计算该点到多边形边界的最短球面距离（单位：km）。

pip install geopy shapely
'''

from typing import List, Tuple, Optional
from shapely.geometry import Point, Polygon, MultiPolygon
from geopy.distance import geodesic
import pandas as pd
from pandas.core.frame import DataFrame

def create_polygon_with_holes(
        exterior_ring: List[Tuple[float, float]],
        interior_rings: Optional[List[List[Tuple[float, float]]]] = None
) -> Polygon:
    """
    创建带孔洞的多边形

    参数:
        exterior_ring: 多边形外环的顶点列表
        interior_rings: 多边形内环（孔洞）的列表

    返回:
        Shapely Polygon 对象
    """
    if interior_rings:
        return Polygon(exterior_ring, interior_rings)
    else:
        return Polygon(exterior_ring)


def point_in_polygon(
        point_lon: float,
        point_lat: float,
        polygon: Polygon
) -> bool:
    """
    判断点是否在多边形内（支持带孔洞的多边形）

    参数:
        point_lon, point_lat: 要判断的点的经纬度
        polygon: Shapely Polygon 对象

    返回:
        点是否在多边形内（布尔值）
    """
    point = Point(point_lon, point_lat)
    return polygon.contains(point)


def distance_to_polygon(
        point_lon: float,
        point_lat: float,
        polygon: Polygon
) -> float:
    """
    计算点到多边形边界的最短球面距离（单位：千米）

    参数:
        point_lon, point_lat: 点的经纬度
        polygon: Shapely Polygon 对象

    返回:
        点到多边形边界的最短距离（千米）
    """
    point = Point(point_lon, point_lat)

    # 如果点在多边形内，距离为0
    if polygon.contains(point):
        return 0.0

    # 计算点到多边形边界的最短距离
    # 使用geodesic方法计算球面距离
    boundary = polygon.boundary

    # 获取边界上最近的点
    nearest_point = boundary.interpolate(boundary.project(point))

    # 计算两点之间的球面距离
    return geodesic(
        (point_lat, point_lon),
        (nearest_point.y, nearest_point.x)
    ).kilometers


activtity_dict = {
        "台湾":[(118,23), (122,23), (122,27), (118,27), (118,23)],
        "钓鱼岛":[(123,25.667), (124.5667,25.667), (124.5667,26), (123,26), (123,25.667)],
        "黄岩岛": [(117.8483, 15.1350), (117.8467, 15.1233), (117.8433, 15.1167), (117.8367, 15.1100), (117.825, 15.1017),
                   (117.7367, 15.105), (117.7183, 15.1217), (117.71, 15.2117), (117.7133, 15.2183), (117.7217, 15.2233),
                   (117.7317, 15.225), (117.74, 15.225), (117.8283, 15.16), (117.84, 15.15), (117.8467, 15.1417),
                   (117.8483, 15.1350)],
        "西沙群岛": [(111.183, 15.7667), (112.9, 15.7667), (112.9, 17.13), (111.183, 17.13), (111.183, 15.7667)],
        "南海岛礁": [(115.883, 7.93), (115.9667, 7.93), (115.9667, 8), (115.883, 8), (115.883, 7.93)],
        "台湾海峡": [(119, 22.5), (122, 22.5), (122, 25.5), (119, 25.5), (119, 22.5)],
    }


# 获取每个区域的 polygon
activtity_polygon = {}

for k,v in activtity_dict.items():
    activtity_polygon[k] = create_polygon_with_holes(v)


config_threshold = {
    "闯岛闯礁":24*1.8,
    "侦查":12*1.8,
    "抵近岛礁":50*1.8,
}

# 获取距离最近一个区域的距离
def getDistanceNbor(row:DataFrame, activtity_polygon: dict, config_threshold:dict):

    # 当前 经纬度距离最近的区域名称
    curRegion = None
    # 当前 经纬度距离最近的区域名称的距离
    curMinDistance = None

    for k,v in activtity_polygon.items():

        # 判断点是否在多边形内
        is_inside = point_in_polygon(row['LON'], row['LAT'], v)
        if is_inside:
            curRegion = k
            curMinDistance = 0
            break
        else:
            # 计算点到多边形的距离
            distance = distance_to_polygon(row['LON'], row['LAT'], v)
            if curMinDistance is None:
                curRegion = k
                curMinDistance = distance
            else:
                if curMinDistance > distance:
                    curRegion = k
                    curMinDistance = distance

    activityName = "正常航行"
    if curMinDistance == 0:
        if curRegion == "台湾海峡":
            activityName = "正在穿航"
        else:
            activityName = "登陆/演习"

    elif curMinDistance <= config_threshold['侦查']:

        if curRegion == "台湾海峡":
            activityName = f"预计要进行穿航，可能性 {1/(curMinDistance + 1) * 100 :.4f}%"
        else:
            activityName = "侦查"

    elif curMinDistance <= config_threshold['闯岛闯礁']:

        if curRegion == "南海岛礁":
            activityName = "闯礁"
        if curRegion in ["台湾", "钓鱼岛", "黄岩岛", "西沙群岛"]:
            activityName = "闯岛"

        if curRegion == "台湾海峡":
            activityName = f"预计要进行穿航，可能性 {1/(curMinDistance + 1) * 100 :.4f}%"

    elif curMinDistance <= config_threshold['抵近岛礁']:

        if curRegion in ["台湾", "钓鱼岛", "黄岩岛", "西沙群岛", "南海岛礁"]:
            activityName = "抵近岛礁"
        if curRegion == "台湾海峡":
            activityName = f"预计要进行穿航，可能性 {1/(curMinDistance + 1) * 100 :.4f}%"

    else:
        activityName = "正常航行"

    # print([curRegion, curMinDistance, activityName])
    return [curRegion, curMinDistance, activityName]

# 示例使用
if __name__ == "__main__":
    # 定义一个带孔洞的多边形
    # 外环（矩形）
    exterior = [
        (116.28, 39.98), (116.45, 39.98),
        (116.45, 40.05), (116.28, 40.05),
        (116.28, 39.98)  # 闭合多边形
    ]
    exterior2 = [(118,23), (122,23), (122,27), (118,27), (118,23)]

    # 内环（孔洞，小矩形）
    interior = [
        (116.32, 40.00), (116.38, 40.00),
        (116.38, 40.03), (116.32, 40.03),
        (116.32, 40.00)  # 闭合孔洞
    ]

    # 创建多边形对象
    # polygon = create_polygon_with_holes(exterior, [interior])

    polygon2 = create_polygon_with_holes(exterior2)
    # 测试点
    test_points = [
        (118, 23),
        (116.35, 40.02),  # 在多边形内但在孔洞外
        (116.35, 40.01),  # 在孔洞内
        (116.20, 40.02),  # 在多边形外
        (116.30, 40.00)  # 在多边形边界附近
    ]

    for i, (lon, lat) in enumerate(test_points):
        # 判断点是否在多边形内
        is_inside = point_in_polygon(lon, lat, polygon2)
        print(f"点 {i + 1} ({lon}, {lat}) 是否在多边形内: {is_inside}")

        # 计算点到多边形的距离
        distance = distance_to_polygon(lon, lat, polygon2)
        print(f"点 {i + 1} 到多边形边界的距离: {distance:.4f} 公里")
        print()

    df = pd.DataFrame({
        "LAT":[23, 25, 15, 15, 7.9, 22, 5],
        "LON":[118, 123, 117, 111, 115, 119, 100]
    })

    # print(df)

    newcols = df.apply(
        lambda row:getDistanceNbor(row,activtity_polygon,config_threshold), axis=1, result_type='expand')

    # print(newcols)
    df[["距离最近区域", "最近距离", "活动场景"]] = newcols
    print(df)





