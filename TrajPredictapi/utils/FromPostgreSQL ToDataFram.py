
# ================= method 1 ====================
import psycopg2
import pandas as pd

# 创建数据库连接
conn = psycopg2.connect(
    host='localhost',
    user='username',
    password='password',
    database='db_name',
    port=5432
)

# 使用 pandas 读取数据
sql_query = "SELECT * FROM your_table_name"
df = pd.read_sql(sql_query, conn)

# 关闭连接
conn.close()

print(df.head())




# ================= method 2 ====================
from sqlalchemy import create_engine
import pandas as pd

# 创建引擎
engine = create_engine('postgresql+psycopg2://username:password@localhost:5432/db_name')

# 读取数据
df = pd.read_sql_table('your_table_name', engine)  # 读取整张表
# 或者使用 SQL 查询
# df = pd.read_sql_query("SELECT * FROM your_table_name", engine)

print(df.head())



# ================= 大表数据，分块读取 ====================

# 使用 chunksize 参数分块读取
for chunk in pd.read_sql_query("SELECT * FROM large_table", engine, chunksize=1000):
    process(chunk)  # 处理每个数据块