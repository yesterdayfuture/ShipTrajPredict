# from shapely.geometry import Point, LineString
# from shapely.ops import transform
# from pyproj import CRS, Transformer
# from geopy.distance import geodesic

# def create_utm_projection(lon, lat):
#     """根据中心点坐标创建UTM投影"""
#     utm_zone = int((lon + 180) / 6) + 1
#     hemisphere = 'north' if lat >= 0 else 'south'
#     return CRS(f"+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84")

# def point_to_line_distance(point_coords, line_coords):
#     """
#     计算经纬度点到线串的最近距离（单位：米）
    
#     参数:
#         point_coords: (经度, 纬度)
#         line_coords: [(经度1,纬度1), (经度2,纬度2)...]
    
#     返回:
#         最近距离（米）
#     """
#     # 创建几何对象
#     geom_point = Point(point_coords)
#     geom_line = LineString(line_coords)
    
#     # 获取UTM投影
#     center_lon = sum(p[0] for p in line_coords)/len(line_coords)
#     center_lat = sum(p[1] for p in line_coords)/len(line_coords)
#     utm_crs = create_utm_projection(center_lon, center_lat)
    
#     # 定义坐标转换器
#     transformer = Transformer.from_crs(4326, utm_crs, always_xy=True)
    
#     # 投影转换
#     projected_point = transform(transformer.transform, geom_point)
#     projected_line = transform(transformer.transform, geom_line)
    
#     # 计算投影后距离
#     return projected_point.distance(projected_line)

# # 示例用法
# if __name__ == "__main__":
#     test_point = (116.404, 39.915)  # 北京天安门坐标
#     test_line = [
#         (116.41, 39.92), 
#         (116.42, 39.93),
#         (116.43, 39.91)
#     ]  # 模拟的路径坐标
    
#     distance = point_to_line_distance(test_point, test_line)
#     print(f"最近距离: {distance:.2f} 米")





from geopy.distance import geodesic
from shapely.geometry import Point, LineString
import numpy as np

def find_nearest_line(target_point, lines_dict):
    """
    计算目标点到多条线段的最短距离及对应线段ID
    :param target_point: 目标点坐标 (lat, lon)
    :param lines_dict: 线段字典 {id: [(lat1,lon1), (lat2,lon2), ...]}
    :return: (最近线段ID, 最短距离_米)
    """
    target = Point(target_point[1], target_point[0])  # Shapely使用(x=lon,y=lat)
    min_distance = float('inf')
    nearest_id = None

    for line_id, coords in lines_dict.items():
        # 将经纬度坐标转换为(lon,lat)格式
        line_coords = [(lon, lat) for lat, lon in coords]
        line = LineString(line_coords)
        distance = target.distance(line) * 111319.9  # 转换为米(1度≈111km)
        
        if distance < min_distance:
            min_distance = distance
            nearest_id = line_id

    return nearest_id, min_distance

# 示例用法
if __name__ == "__main__":
    # 测试点(北京天安门)
    test_point = (39.9042, 116.4074)
    
    # 模拟多条线段(格式: {ID: [点1,点2,...]})
    sample_lines = {
        "line1": [(31.2304, 121.4737), (30.2741, 120.1551)],  # 上海-杭州
        "line2": [(22.5431, 114.0579), (23.1291, 113.2644)],  # 深圳-广州
        "line3": [(34.0522, -118.2437), (40.7128, -74.0060)]  # 洛杉矶-纽约
    }
    
    nearest_id, distance = find_nearest_line(test_point, sample_lines)
    print(f"最近线段: {nearest_id}, 距离: {distance:.2f}米")
