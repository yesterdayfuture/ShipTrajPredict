# main.py
from fastapi import FastAPI
from response import R, response_success, response_fail
import uvicorn
from typing import List, Dict, Union
import pandas as pd
import numpy as np

from utils.util import *
from utils.CalcCoTraj import CalcCoTraj_main
from utils.CalcTrajActivity import CalcTrajActivity_main
from utils.CalcShipSogCog import add_speed_heading_geopy
from utils.UnionTraj import UnionTraj_main
from kimi.util_modelPredict import UtilModelPredic_main
from utils.CalcActivity_v2 import getDistanceNbor, config_threshold, activtity_polygon

app = FastAPI(
    title="Demo API",
    version="1.0.0",
    docs_url="/docs"
)

@app.get("/api/v1/hello", response_model=R[str])
def hello():
    return response_success(data="Hello FastAPI")

@app.get("/api/v1/users", response_model=R[List[Dict]])
def users():
    return response_success(data=[{"id": 1, "name": "Alice"}])

@app.get("/api/v1/fail", response_model=R[str])
def users():
    return response_fail(data="调用失败")


# 轨迹预测 api
@app.post("/api/v1/traj_predict", response_model=Union[R[Dict], R[str]], description="根据输入的轨迹点，预测未来时间点的经纬度，每条船至少存在32个轨迹点")
def traj_predict(data:Dict):

    # 获取 输入数据的 key 值
    clounms = list(data.keys())
    # 输入数据的 key 值 必须包含下方内容
    required_keys = ['MMSI','LAT','LON','timeUnix']
    # 检查字典是否包含所有指定的键
    if  not contains_keys(clounms, required_keys):
        return response_fail(data="输入数据不规范，字典缺少某些键")
    
    df = pd.DataFrame(data)
    df['timeUnix'] = pd.to_datetime(df["timeUnix"])

    # 计算 速度与航向
    df = add_speed_heading_geopy(df)

    df = pd.read_csv(r'data/AIS_2023_12_28_test.csv')

    # 获取预测结果
    df = UtilModelPredic_main(df)

    print(type(df))
    # print(df)

    # 返回 时间类型 的列名
    timeCols = getTimeColNmae(df)

    if len(timeCols) >= 1 :
        for col in timeCols:
            df[col] = df[col].astype(str)

    return response_success(data=convert_df_to_dict(df))


# 轨迹共现判断 api
@app.post("/api/v1/co_traj", response_model=Union[R[Dict], R[str]], description="根据输入的船舶轨迹，判断是否在某个时空片段进行 同时出现 (输入数据的 key 值 必须包含 'MMSI','LAT','LON','timeUnix')")
def co_traj(data:Dict):
    # 获取 输入数据的 key 值
    clounms = list(data.keys())
    # 输入数据的 key 值 必须包含下方内容
    required_keys = ['MMSI','LAT','LON','timeUnix']
    # 检查字典是否包含所有指定的键
    if  not contains_keys(clounms, required_keys):
        return response_fail(data="输入数据不规范，字典缺少某些键")
    
    df = pd.DataFrame(data)
    df['timeUnix'] = pd.to_datetime(df["timeUnix"])

    Co_result = CalcCoTraj_main(df)

    Co_result = UnionTraj_main(Co_result)

    # 返回 时间类型 的列名
    timeCols = getTimeColNmae(Co_result)
    print(timeCols)

    if len(timeCols) >= 1 :
        for col in timeCols:
            Co_result[col] = Co_result[col].astype(str)
    
    # print(Co_result['duration'].dtype)

    # Co_result['duration'] = Co_result['duration'].astype(str)

    return response_success(data=convert_df_to_dict(Co_result))


# 活动场景 分析 api
@app.post("/api/v1/traj_activity", response_model=Union[R[Dict], R[str]], description="根据输入的船舶轨迹，判断当前的轨迹所属 任务场景(输入数据的 key 值 必须包含 'MMSI','LAT','LON','timeUnix','lat_center','lon_center')")
def traj_activity(data:Dict):
    # 获取 输入数据的 key 值
    clounms = list(data.keys())
    # 输入数据的 key 值 必须包含下方内容
    required_keys = ['MMSI','LAT','LON','timeUnix','lat_center','lon_center']
    # 检查字典是否包含所有指定的键
    if  not contains_keys(clounms, required_keys):
        return response_fail(data="输入数据不规范，字典缺少某些键")
    
    new_data = {}
    for key in ['MMSI','LAT','LON','timeUnix']:
        new_data[key] = data[key]
    
    lat_center = data['lat_center']
    lon_center = data['lon_center']

    df = pd.DataFrame(new_data)
    df['timeUnix'] = pd.to_datetime(df["timeUnix"])

    result_data = CalcTrajActivity_main(df, lat_center, lon_center)


    result_data['start'] = result_data['start'].astype(str)
    result_data['end'] = result_data['end'].astype(str)
    result_data['duration'] = result_data['duration'].astype(str)

    return response_success(data=convert_df_to_dict(result_data))


# 活动场景 分析 api
@app.post("/api/v2/traj_activity", response_model=Union[R[Dict], R[str]],
          description="根据输入的船舶轨迹，判断当前的轨迹所属 任务场景(输入数据的 key 值 必须包含 'MMSI','LAT','LON','timeUnix')")
def traj_activity2(data: Dict):
    # 获取 输入数据的 key 值
    clounms = list(data.keys())
    # 输入数据的 key 值 必须包含下方内容
    required_keys = ['MMSI', 'LAT', 'LON', 'timeUnix']
    # 检查字典是否包含所有指定的键
    if not contains_keys(clounms, required_keys):
        return response_fail(data="输入数据不规范，字典缺少某些键")

    new_data = {}
    for key in ['MMSI', 'LAT', 'LON', 'timeUnix']:
        new_data[key] = data[key]

    df = pd.DataFrame(new_data)
    df['timeUnix'] = pd.to_datetime(df["timeUnix"])

    newcols = df.apply(
        lambda row: getDistanceNbor(row, activtity_polygon, config_threshold), axis=1, result_type='expand')

    # print(newcols)
    df[["NborRegion", "NborDistance", "Activity"]] = newcols

    return response_success(data=convert_df_to_json(df))



# 主函数入口
if __name__ == '__main__':
# 启动FastAPI应用，用6006端口映射到本地，从而在本地使用api
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)  # 在指定端口和主机上启动应用