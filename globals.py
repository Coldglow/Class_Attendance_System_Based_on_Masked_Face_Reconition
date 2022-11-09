# 连接数据库操作
import pymysql


def connect_to_sql():
    db = pymysql.connect(host="localhost", user="root", password="wsw101733", database="class_info")
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    return db, cursor


def prepare_data(*args):
    data = []
    for value in args:
        if isinstance(value, tuple):
            for item in value:
                data.append(item)
        else:
            data.append(value)

    return data
