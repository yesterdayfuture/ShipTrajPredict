'''

使用代码操作数据库表进行增删改查
pip install sqlalchemy pymysql psycopg2-binary

'''

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据库连接配置
# MySQL
DATABASE_URL = "mysql+pymysql://username:password@localhost:3306/mydatabase"
# PostgreSQL
# DATABASE_URL = "postgresql+psycopg2://username:password@localhost:5432/mydatabase"

# 创建引擎
engine = create_engine(DATABASE_URL, echo=True)  # echo=True 显示执行的SQL

# 表结构假设（用于示例）
"""
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    age INT,
    salary DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active TINYINT DEFAULT 1
);

CREATE TABLE products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    category VARCHAR(50),
    stock INT DEFAULT 0
);
"""

def execute_query(sql, params=None):
    """执行查询语句，返回结果列表"""
    try:
        with engine.connect() as connection:
            # 使用 text() 包装 SQL 语句，支持参数化查询
            result = connection.execute(text(sql), params or {})
            # 获取所有结果
            return result.fetchall()
    except SQLAlchemyError as e:
        logger.error(f"查询执行失败: {e}")
        return []


def execute_query_single(sql, params=None):
    """执行查询语句，返回单条结果"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql), params or {})
            return result.fetchone()
    except SQLAlchemyError as e:
        logger.error(f"查询执行失败: {e}")
        return None

def execute_scalar(sql, params=None):
    """执行查询语句，返回单个值"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql), params or {})
            row = result.fetchone()
            return row[0] if row else None
    except SQLAlchemyError as e:
        logger.error(f"标量查询失败: {e}")
        return None


def execute_update(sql, params=None):
    """执行增删改语句，返回影响的行数"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql), params or {})
            connection.commit()  # 手动提交事务
            return result.rowcount
    except SQLAlchemyError as e:
        logger.error(f"更新操作失败: {e}")
        return 0

def execute_many(sql, params_list):
    """批量执行相同的SQL语句"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql), params_list)
            connection.commit()
            return result.rowcount
    except SQLAlchemyError as e:
        logger.error(f"批量操作失败: {e}")
        return 0


def create_user(username, email, age=None, salary=None):
    """创建新用户"""
    sql = """
    INSERT INTO users (username, email, age, salary) 
    VALUES (:username, :email, :age, :salary)
    """

    params = {
        'username': username,
        'email': email,
        'age': age,
        'salary': salary
    }

    affected_rows = execute_update(sql, params)
    if affected_rows > 0:
        logger.info(f"用户 {username} 创建成功")
        return True
    return False


def batch_create_users(users_data):
    """批量创建用户"""
    sql = """
    INSERT INTO users (username, email, age, salary) 
    VALUES (:username, :email, :age, :salary)
    """

    affected_rows = execute_many(sql, users_data)
    logger.info(f"批量创建了 {affected_rows} 个用户")
    return affected_rows


def get_all_users():
    """获取所有用户"""
    sql = "SELECT * FROM users WHERE is_active = 1 ORDER BY created_at DESC"
    return execute_query(sql)

def get_user_by_id(user_id):
    """根据ID获取用户"""
    sql = "SELECT * FROM users WHERE id = :user_id AND is_active = 1"
    return execute_query_single(sql, {'user_id': user_id})

def get_users_by_age_range(min_age, max_age):
    """根据年龄范围查询用户"""
    sql = """
    SELECT id, username, email, age, salary, created_at 
    FROM users 
    WHERE age BETWEEN :min_age AND :max_age AND is_active = 1
    ORDER BY age
    """
    return execute_query(sql, {'min_age': min_age, 'max_age': max_age})

def search_users(keyword):
    """搜索用户"""
    sql = """
    SELECT * FROM users 
    WHERE (username LIKE :keyword OR email LIKE :keyword) 
    AND is_active = 1
    """
    return execute_query(sql, {'keyword': f'%{keyword}%'})

def get_user_count():
    """获取用户数量"""
    sql = "SELECT COUNT(*) FROM users WHERE is_active = 1"
    return execute_scalar(sql)


def update_user(user_id, **kwargs):
    """更新用户信息"""
    if not kwargs:
        return 0

    set_clauses = []
    params = {'user_id': user_id}

    for field, value in kwargs.items():
        if value is not None:  # 只更新非None的值
            set_clauses.append(f"{field} = :{field}")
            params[field] = value

    if not set_clauses:
        return 0

    sql = f"UPDATE users SET {', '.join(set_clauses)} WHERE id = :user_id"

    affected_rows = execute_update(sql, params)
    logger.info(f"更新了 {affected_rows} 个用户")
    return affected_rows


def increase_salary(percentage):
    """给所有用户涨薪"""
    sql = "UPDATE users SET salary = salary * (1 + :percentage) WHERE is_active = 1"
    affected_rows = execute_update(sql, {'percentage': percentage / 100})
    logger.info(f"为 {affected_rows} 个用户涨薪 {percentage}%")
    return affected_rows


def delete_user(user_id):
    """删除用户（物理删除）"""
    sql = "DELETE FROM users WHERE id = :user_id"
    affected_rows = execute_update(sql, {'user_id': user_id})
    logger.info(f"删除了 {affected_rows} 个用户")
    return affected_rows

def soft_delete_user(user_id):
    """软删除用户"""
    sql = "UPDATE users SET is_active = 0 WHERE id = :user_id"
    affected_rows = execute_update(sql, {'user_id': user_id})
    logger.info(f"软删除了 {affected_rows} 个用户")
    return affected_rows

def delete_inactive_users():
    """删除所有非活跃用户"""
    sql = "DELETE FROM users WHERE is_active = 0"
    affected_rows = execute_update(sql)
    logger.info(f"删除了 {affected_rows} 个非活跃用户")
    return affected_rows


def get_user_statistics():
    """获取用户统计信息"""
    sql = """
    SELECT 
        COUNT(*) as total_users,
        AVG(age) as avg_age,
        AVG(salary) as avg_salary,
        SUM(salary) as total_salary,
        MAX(salary) as max_salary,
        MIN(salary) as min_salary
    FROM users 
    WHERE is_active = 1
    """
    return execute_query_single(sql)


def get_users_with_pagination(page=1, page_size=10):
    """分页查询用户"""
    offset = (page - 1) * page_size
    sql = """
    SELECT * FROM users 
    WHERE is_active = 1 
    ORDER BY created_at DESC 
    LIMIT :limit OFFSET :offset
    """
    return execute_query(sql, {'limit': page_size, 'offset': offset})


def transactional_operations():
    """事务操作示例：要么全部成功，要么全部回滚"""
    try:
        with engine.connect() as connection:
            # 开始事务
            with connection.begin():
                # 插入用户
                user_sql = """
                INSERT INTO users (username, email, age) 
                VALUES ('transaction_user', 'transaction@example.com', 30)
                """
                connection.execute(text(user_sql))

                # 插入产品
                product_sql = """
                INSERT INTO products (name, price, category) 
                VALUES ('Transaction Product', 100.0, 'Test')
                """
                connection.execute(text(product_sql))

                logger.info("事务操作成功完成")
                return True

    except SQLAlchemyError as e:
        logger.error(f"事务操作失败: {e}")
        return False




def query_to_dataframe(sql, params=None):
    """将SQL查询结果转换为DataFrame"""
    try:
        with engine.connect() as connection:
            df = pd.read_sql_query(text(sql), connection, params=params)
            return df
    except Exception as e:
        logger.error(f"DataFrame转换失败: {e}")
        return pd.DataFrame()

def get_users_dataframe():
    """获取用户DataFrame"""
    sql = "SELECT * FROM users WHERE is_active = 1"
    return query_to_dataframe(sql)

def get_user_stats_dataframe():
    """获取用户统计信息的DataFrame"""
    sql = """
    SELECT 
        category,
        COUNT(*) as user_count,
        AVG(age) as avg_age,
        AVG(salary) as avg_salary
    FROM users 
    WHERE is_active = 1 
    GROUP BY category
    """
    return query_to_dataframe(sql)


def main():
    """主函数示例"""

    # 1. 创建用户
    create_user("john_doe", "john@example.com", 25, 50000.0)
    create_user("jane_smith", "jane@example.com", 30, 60000.0)

    # 2. 批量创建用户
    users_data = [
        {'username': 'user1', 'email': 'user1@example.com', 'age': 22, 'salary': 45000.0},
        {'username': 'user2', 'email': 'user2@example.com', 'age': 28, 'salary': 55000.0},
        {'username': 'user3', 'email': 'user3@example.com', 'age': 35, 'salary': 70000.0}
    ]
    batch_create_users(users_data)

    # 3. 查询用户
    users = get_all_users()
    print(f"所有用户数量: {len(users)}")

    # 4. 条件查询
    young_users = get_users_by_age_range(20, 30)
    print(f"20-30岁用户: {len(young_users)}")

    # 5. 更新用户
    update_user(1, salary=55000.0, age=26)

    # 6. 获取统计信息
    stats = get_user_statistics()
    if stats:
        print(f"平均年龄: {stats.avg_age:.1f}, 平均薪资: {stats.avg_salary:.2f}")

    # 7. 使用DataFrame
    df = get_users_dataframe()
    print(f"DataFrame形状: {df.shape}")
    print(df.head())

    # 8. 分页查询
    page1 = get_users_with_pagination(page=1, page_size=5)
    print(f"第1页用户: {len(page1)}")

    # 9. 事务操作
    transactional_operations()


if __name__ == "__main__":
    main()






