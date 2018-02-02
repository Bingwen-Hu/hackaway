# -*- coding: utf-8 -*-
"""163.com mail example
Created on Fri Feb  2 09:50:24 2018

@author: Mory
"""
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import smtplib
import pymysql
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--host', help='mysql host')
parser.add_argument('--user', help='mysql username')
parser.add_argument('--passwd', help='mysql password')
parser.add_argument('--port', help='mysql port')
parser.add_argument('--receiver', help='receiver username')
parser.add_argument('--sender', help='sender username')
parser.add_argument('--sender_pass', help='sender password')
args = parser.parse_args()

HOST = args.host
USER = args.user
PASSWORD = args.passwd
PORT = int(args.port)
DATABASE = "urun_statistic"
TABLE = 'wechat_info'


mailto_list = [args.receiver]
mail_host = 'smtp.163.com'
mail_user = args.sender
mail_pass = args.sender_pass




def connect_sql(host, user, password, db, port=3306):
    """ 数据库连接 """
    conn = pymysql.connect(host=host, user=user, port=port,
                           password=password, charset='utf8')
    conn.select_db(db)
    return conn


def gen_date():
    """ 返回当天的日期，以及一个包含一天，一周，一月的字符串列表

    ret: 元组，(访问接口日期，（今天日期，一周前日期，一个月前日期）)
    """
    def helper(day):
        day = str(day.date())
        day = ''.join(day.split('-'))
        return day

    today = datetime.now()
    week = today - timedelta(7)
    month = today - timedelta(30)
    fetch_date = str(today.date())
    lst = list(map(helper, [today, week, month]))
    return fetch_date, lst


def fetch_mysql(conn, date):
    """从数据库中获取当天数据，如果有数据，说明程序正常运行"""
    sql = "select * from wechat_info where fetchdate=%s and infotype=%s"
    params = [date, 3]
    cur = conn.cursor()
    result = cur.execute(sql, params)
    cur.close()
    return result



def send_mail(to_list, subject, content):
    me = f"LogServer<{mail_user}>"
    msg = MIMEText(content, _subtype='plain', _charset='utf-8')
    msg['Subject'] = subject
    msg['From'] = me
    msg['To'] = ";".join(to_list)

    try:
        server = smtplib.SMTP()
        server.connect(mail_host)
        server.login(mail_user, mail_pass)
        server.sendmail(me, to_list, msg.as_string())
        server.close()
        return True
    except Exception as e:
        print(e)
        return False

if __name__ == '__main__':
    conn = connect_sql(HOST, USER, PASSWORD, DATABASE, PORT)
    date, _ = gen_date()
    result = fetch_mysql(conn, date)
    if result > 0:
        content = "Mory, 程序运行正常"
    else:
        content = "Mory，程序可能运行不正常，得马上检查一下！"
    send_mail(mailto_list, '程序监控报告', content)