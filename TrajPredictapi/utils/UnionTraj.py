
'''
本文件将 CalcCoTraj.py 文件计算的 共现结果进行组合
如：
    已知组1中a和组2中b是一行，组1中b和组2中c是一行，组1中c和组2中d是一行，且开始时间最大差距不大于30min，结束时间最大差距不大于30min，那么认为abcd是一个大组的

'''

import pandas as pd
from pandas.core.frame import DataFrame
from datetime import datetime, timedelta
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

window_timedelta = os.getenv('windowTimedelta')
print(f'从 .env 读取配置信息 windowTimedelta 为 {window_timedelta}')



#使用并查集（Union-Find）数据结构来分组元素。给定配对关系（如a和b一组、b和c一组、c和d一组），程序会将所有相连的元素合并到同一组中。
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1



def getTimeGroup(df:DataFrame, windowsTime = '30 min') -> List[DataFrame]:
    # 2. 把行按“时间窗口”聚类
    # 思路：只要某行与前一个窗口的最早/最晚时间差都在 30 min 内就归到同一窗口，否则新开窗口
    # max_gap = timedelta(minutes=30)
    max_gap = pd.Timedelta(windowsTime)

    # 先按开始时间排序
    df_sorted = df.sort_values('start').reset_index(drop=True)

    window_id = 0
    windows = []

    current_window = [(0,df_sorted.iloc[0])]
    for index, row in df_sorted.iloc[1:].iterrows():
        # 当前窗口的边界
        min_start = min(r['start'] for _, r in current_window)
        max_end   = max(r['end'] for _, r in current_window)
        
        # 判断是否可以合并
        if (abs(row['start'] - min_start) <= max_gap and
            abs(row['end'] - max_end)   <= max_gap):
            current_window.append((index, row))
        else:
            windows.append(current_window)
            current_window = [(index, row)]

    # 最后一组 添加
    windows.append(current_window)



    df_groupByTime = []
    # 3. 输出结果
    for idx, win in enumerate(windows, 1):
        all_members = sorted(set(sum([[r['mmsi_A'], r['mmsi_B']] for _,r in win], [])))
        min_start = min(r['start'] for _, r in win)
        max_end   = max(r['end'] for _, r in win)
        index_list = [i for i,_ in win]
        print(f"大组 {idx}: 成员 {all_members}")
        print(f'   成员函数对应的原始索引为：{index_list}')
        print(f"   最小开始时间: {min_start}")
        print(f"   最大结束时间: {max_end}\n")

        cur_group = df_sorted.iloc[index_list]
        cur_group['group'] = idx
        # print(cur_group)
        df_groupByTime.append(cur_group)

    return df_groupByTime


# 对 每个时间窗口 进行 详细分组
def group_elements(df :DataFrame) -> List:
    uf = UnionFind()
    for _, r in df.iterrows():
        uf.union(r['mmsi_A'], r['mmsi_B'])
    
    groups = {}
    for element in uf.parent:
        root = uf.find(element)
        if root not in groups:
            groups[root] = []
        groups[root].append(element)
    
    return list(groups.values())




# 主函数
def UnionTraj_main(df :DataFrame, windowsTime = None):
    '''
    本文件的主函数
    df 要包含 mmsi_A、mmsi_B、start、end、type 这些列
    '''

    if not windowsTime:
        windowsTime = window_timedelta

    # 获取 根据时间窗口进行划分后的结果，是一个 DataFrame 列表
    df_groupByTime = getTimeGroup(df,windowsTime)

    # 用来保存 每一个时间分组下的详细分组，是一个三维列表
    # 第一维长度 表示 共有几个 时间窗口分组，每一个元素代表 一个时间窗口下的详细分组信息
    # 第二维长度 表示 当前时间窗口分组 下共分为 几个大组，每一个元素代表 当前大组下的 组名
    # 第三维长度 表示 当前大组共有 几个组， 每一个元素代表 组名
    group_list = []

    # 循环判断 每个时间窗口分组 下的详细分组信息
    # 添加 child_group 列，表示在当前时间窗口分组中，属于哪个大组
    for idx, df_group in enumerate(df_groupByTime, 1):
        cur_group_list = group_elements(df_group)
        print(f'当前组序号为：{idx}')
        print(f'当前组的数据为：{df_group}')
        print(f'当前组细分为：{len(cur_group_list)} 组')
        print(f'当前组的细分组为：{cur_group_list} \n')
        
        for child_idx, cur_child_list in enumerate(cur_group_list, 1):
            #新增 child_group 列，表示在当前分组（指的是连接关系分组）下，属于那一组
            # 创建条件：组1或组2在cur_child_list中
            condition = df_group['mmsi_A'].isin(cur_child_list) | df_group['mmsi_B'].isin(cur_child_list)
            
            # 初始化child_group列（如果不存在）
            if 'child_group' not in df_group.columns:
                df_group['child_group'] = None
            
            # 使用loc进行安全赋值
            df_group.loc[condition, 'child_group'] = child_idx
        group_list.append(cur_group_list)

    
    # 存放分离后的 DataFrame
    df_list = []
    # 原始的 df_groupByTime 中，一行数据表示两个组（即两条船），现在进行拆分，一行只表示一个组（即一条船）
    for cur_df in df_groupByTime:
        columns = list(cur_df.columns)
        colunms1 = columns.copy()
        colunms1.remove('mmsi_B')
        colunms2 = columns.copy()
        colunms2.remove('mmsi_A')
        df_list.append(cur_df.loc[:,colunms1])

        df_tep = cur_df.loc[:,colunms2]
        # print(df_tep)

        df_list.append(df_tep.rename(columns={'mmsi_B':'mmsi_A'}))   # 原地修改

    # 将分离后的 DataFrame 进行合并，并 根据 ['group', 'child_group'] 两列 进行排序
    df_all = pd.concat(df_list)
    df_all = df_all.reset_index(drop=True)
    df_all.sort_values(['group', 'child_group'])

    # 以 'group', 'child_group' 进行分组，找到每组的 最小开始时间 和 最大结束时间
    result = df_all.groupby(['group', 'child_group']).agg({
        'start': 'min',
        'end': 'max',
        'type': lambda x: x.iloc[0]  # 取组内首个type值
    }).reset_index()


    # 多列关联（指定左右列名）
    df_result = pd.merge(df_all, result, left_on=['group','child_group'], right_on=['group','child_group'], how='outer')

    #去除 重复行
    df_result = df_result.drop_duplicates(subset=['mmsi_A', 'start_y', 'end_y'])
    #移除 某些列
    df_result = df_result.drop(columns=['start_x', 'end_x', 'type_x'])
    df_result.reset_index(drop=True)

    df_result = df_result.rename(columns={'mmsi_A':'MMSI', 'start_y':'start', 'end_y':'end', 'type_y':'type'})

    return df_result


