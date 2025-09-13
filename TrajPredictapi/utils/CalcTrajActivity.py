
'''
本文件 根据 轨迹点 到 指定经纬度 之间的 距离来进行判断 当前活动场景
'''
from pandas.core.frame import DataFrame
from geopy.distance import geodesic


# 计算距离
def getDist(r,lat_center=31.23,lon_center=121.471):
    return geodesic((r.LAT, r.LON), (lat_center, lon_center)).km


#判定 活动场景
def judge_activity(r):
    if r.dist_km > 20:
        return '正常'
    if r.dist_km <= 20 and r.dist_km > 10:
        return '穿航'
    if r.dist_km <= 10 and r.dist_km > 8:
        return 'z察'
    if r.dist_km <= 8 and r.dist_km > 6:
        return '闯礁'
    if r.dist_km <= 6 and r.dist_km > 4:
        return '闯岛'
    else:
        return '登陆/演习'


def CalcTrajActivity_main(df:DataFrame, lat_center = None, lon_center = None):
    '''
    df 要包含 MMSI、timeUnix、LAT、LON 这些列
    '''

    # 添加 距离 列
    df['dist_km'] = df.apply(lambda r:getDist(r,lat_center,lon_center),axis=1)

    # 添加 活动场景 列
    df['activity'] = df.apply(judge_activity,axis=1)

    # 连续 活动场景 片段
    df['grp'] = (df['activity'] != df['activity'].shift()).cumsum()

    activity_traj = (df.groupby(['MMSI','activity']).agg(
                mmsi=('MMSI', 'first'),
                start=('timeUnix', 'min'),
                end=('timeUnix', 'max'),
                activity=('activity', 'first')
            ))

    activity_traj['duration'] = activity_traj['end'] - activity_traj['start']

    return activity_traj.reset_index(drop=True)




