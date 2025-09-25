'''

测试 datatime 时间模块

'''

import datetime as dt


cur_time = dt.datetime.now()
print(f" 当前时间 = {cur_time}")
print(f" 当前 UTC 时间 = {dt.datetime.utcnow()}")
print(f" 当前时间戳 = {dt.datetime.now().timestamp()}")
print(f" 当前日期 = {dt.date.today()}")
print(f" 今日时间 = {dt.datetime.now().strftime('%H:%M:%S')}")

