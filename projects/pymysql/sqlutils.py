import pymysql
import pandas as pd
import numpy as np


# database setting
PARAMS = {
    'host': "192.168.1.12",
    "port": 3306,
    'user': 'user',
    'password': 'password',
    'charset': 'utf8',
    'database': 'database',
}




def desc_table(cur, table, justcolumns=False):
    cur.execute(f"desc {table}")
    res = cur.fetchall()
    
    if justcolumns:
        res = [r[0] for r in res]

    return res


def show_tables(cur):
    cur.execute('show tables')
    res = cur.fetchall()
    return [r[0] for r in res]


def head_table(cur, table, limit=5):
    return select_table(cur, table, limit)


def select_table(cur, table, limit=None):
    sqls = f"select * from {table} limit {limit}"

    if limit is None or -1:
        sqls = f"select * from {table}"

    cur.execute(sqls)
    res = cur.fetchall()
    columns = desc_table(cur, table, justcolumns=True)
    data = np.array(res)
    df = pd.DataFrame(data, columns=columns)
    return df


def dump_table(cur, table, rows=None, dir="."):
    """
    Args:
        cur: server.cursor
        table: sql database table name
        rows: a.k.a limit
    """
    df = select_table(cur, table)
    df.to_csv(f"{dir}/{table}.csv")


def dump_database(server, dir='.'):
    """
    Args:
        server: a.k.a conn
    """
    cur = server.cursor()
    tables = show_tables(cur)
    for table in tables:
        dump_table(cur, table, dir=dir)
    cur.close()


conn = pymysql.connect(**PARAMS)
conn.close()