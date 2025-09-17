
import pandas as pd
from pandas.core.frame import DataFrame

def contains_keys(dictionary, keys):
    """
    检查字典是否包含所有指定的键
    :param dictionary: 要检查的字典的 keys 集合
    :param keys: 需要检查的键列表/集合
    :return: bool 是否包含所有键
    """
    return all(key in dictionary for key in keys)


def convert_df_to_dict(df):
    """
    将DataFrame转换为指定格式的字典
    格式示例:
    {
        "date_str": ["2023-01-01", "2023-02-15"],
        "value": [100, 200],
        "other_col": ["A", "B"]
    }
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是pandas DataFrame")
    
    result_dict = {}
    for col in df.columns:
        result_dict[col] = df[col].tolist()
    return result_dict


def convert_df_to_json(df):
    """
    将DataFrame转换为指定格式的字典
    格式示例:
    {
        "mmsi编号1": [
                    {"LON":121, "LAT":121,"timeUnix":"2025-01-01"},
                    {"LON":121, "LAT":121,"timeUnix":"2025-01-01"},
                    ],
        "mmsi编号2": [
                    {"LON":121, "LAT":121,"timeUnix":"2025-01-01"},
                    {"LON":121, "LAT":121,"timeUnix":"2025-01-01"},
                    ],
    }
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是pandas DataFrame")

    result_dict = {}
    df_groups = df.groupby("MMSI")
    for curGroupName in list(df_groups.groups.keys()):
        cur_group = df_groups.get_group(curGroupName).sort_values(by="timeUnix")
        cur_group_cols = cur_group.columns.tolist()
        cur_data = []
        # print(cur_group_cols)
        for _, row in cur_group.iterrows():
            # print(row)
            cur_row = {k:row[k] for k in cur_group_cols if k!="MMSI"}
            cur_data.append(cur_row)

        result_dict[curGroupName] = cur_data

    return result_dict

def getTimeColNmae(df:DataFrame):
    """
    获取 类型属于 时间的 列的列名
    """

    cols = df.columns

    timeCols = []
    for col in cols:
        if df[col].dtype == 'datetime64[ns]' or df[col].dtype == "timedelta64[ns]":
            timeCols.append(col)
    
    return timeCols

# 将时间类型 转为 字符串时，规范 时间格式
def format_timedelta(td):
    if pd.isna(td):
        return None
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"