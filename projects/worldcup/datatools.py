import pymysql
import json
import re
import os
import time
from email.mime.text import MIMEText
import smtplib

mailto_list = [os.environ.get('RECEIVER')]
mail_host = 'smtp.163.com'
mail_user = os.environ.get('SENDER')
mail_pass = os.environ.get('SENDER_PASS')

SERVER_PARAMS = {
    'host': "localhost",
    'port': 27702,
    'user': 'like_jian',
    'password': 'worldcup2018',
    'database': 'worldcup',
    'charset': 'utf8'
}

server = pymysql.Connection(**SERVER_PARAMS)

teams = [
    [1, '俄罗斯', '沙特'],
    [2, '埃及', '乌拉圭'],
    [3, '葡萄牙', '西班牙'],
    [4, '摩洛哥', '伊朗'],
    [5, '法国', '澳大利亚'],
    [6, '秘鲁', '丹麦'],
    [7, '阿根廷', '冰岛'],
    [8, '克罗地亚', '尼日利亚'],
    [9, '巴西', '瑞士'],
    [10, '哥斯达黎加', '塞尔维亚'],
    [11, '德国', '墨西哥'],
    [12, '瑞典', '韩国'],
    [13, '比利时', '巴拿马'],
    [14, '突尼斯', '英格兰'],
    [15, '哥伦比亚', '日本'],
    [16, '波兰', '塞内加尔'],
    [17, '俄罗斯', '埃及'],
    [18, '乌拉圭', '沙特'],
    [19, '葡萄牙', '摩洛哥'],
    [20, '法国', '秘鲁'],
    [21, '伊朗', '西班牙'],
    [22, '丹麦', '澳大利亚'],
    [23, '阿根廷', '克罗地亚'],
    [24, '尼日利亚', '冰岛'],
    [25, '巴西', '哥斯达黎加'],
    [26, '塞尔维亚', '瑞士'],
    [27, '韩国', '墨西哥'],
    [28, '德国', '瑞典'],
    [29, '比利时', '突尼斯'],
    [30, '英格兰', '巴拿马'],
    [31, '波兰', '哥伦比亚'],
    [32, '日本', '塞内加尔'],
    [33, '乌拉圭', '俄罗斯'],
    [34, '沙特', '埃及'],
    [35, '伊朗', '葡萄牙'],
    [36, '西班牙', '摩洛哥'],
    [37, '丹麦', '法国'],
    [38, '澳大利亚', '秘鲁'],
    [39, '尼日利亚', '阿根廷'],
    [40, '冰岛', '克罗地亚'],
    [41, '塞尔维亚', '巴西'],
    [42, '瑞士', '哥斯达黎加'],
    [43, '韩国', '德国'],
    [44, '墨西哥', '瑞典'],
    [45, '英格兰', '比利时'],
    [46, '巴拿马', '突尼斯'],
    [47, '日本', '波兰'],
    [48, '塞内加尔', '哥伦比亚'],
]

def get_contents(num=12):
    sqls = f"select Blogerid, Content from Data where  Addon >= NOW() - interval {num} hour"
    with server.cursor() as cursor:
        cursor.execute(sqls)
        content = cursor.fetchall()
    return content

def get_host_and_guest(gpid):
    team = teams[gpid-1]
    return team[1], team[2]


def insert_newdata(gpid, host, guest, dpoints):
    sqls = "insert into Predict (GPid, Host, Guest, DPoints) values (%s, %s, %s, %s)"
    with server.cursor() as cursor:
        params = [gpid, host, guest, dpoints]
        cursor.execute(sqls, params)
    server.commit()
    print("Insert GPid: %s, Host: %s, Guest: %s, DPoints: %s" % (gpid, host, guest, dpoints))


def match_data_helper(contents, pattern):
    for (blogerid, content) in contents:
        res = pattern.search(content)
        if res is not None:
            start, end = res.start(), res.end()
            cup = content[start-10: end+10]
            yield cup


def match_data(contents, pattern):
    lst = list(match_data_helper(contents, pattern))
    lst = list(set(lst))
    return lst




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


pattern = re.compile(r'\D\d[：:\-比]\d\D')

if __name__ == '__main__':
    while(True):
        server.connect()
        contents = get_contents(12)
        data = match_data(contents, pattern)
        text = "\n".join(data)
        send_mail(mailto_list, "worldcup data explore", text)
        server.close()
        time.sleep(12 * 3600)