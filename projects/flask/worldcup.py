# coding: utf-8
"""
python:  3.6
flask:   1.0.2

"""
from flask import Flask
from flask import request
import pymysql
import json
import random

app = Flask(__name__)

SERVER_PARAMS = {
    'host': "localhost",
    'port': 27702,
    'user': 'like_jian',
    'password': 'worldcup2018',
    'database': 'worldcup',
    'charset': 'utf8'
}

server = pymysql.Connection(**SERVER_PARAMS)
@app.route('/predict/<gpid>')
def predict(gpid):
    if not server.open:
        server.connect()
    with server.cursor() as cursor:
        sqls = "select * from Predict where GPid=%s" % gpid
        cursor.execute(sqls)
        result = cursor.fetchone()
        if result is not None:
            sqldict = predict_parser(result)
        else:
            sqldict = {'code': 1, "msg": "场次必须为1-62（包括62）之间的数"}
    jsondata = json.dumps(sqldict)
    return jsondata


def predict_parser(result):
    temp = {
        'code': 0,
        'GPid': result[0],
        'host': result[1],
        'guest': result[2],
        'points': result[3] if result[3] else result[6],
        'rate': result[4] if result[4] else result[5],
    }
    score_h, score_g = temp['points'].split('-')
    res = {
        "code": 0,
        "msg": "获取成功",
        "basedata":{
            "teams":[
                {
                    "teamname": temp['guest'],
                    "score": int(score_g),
                },
                {
                    "teamname": temp["host"],
                    "score": int(score_h)
                }
            ]
        }
    }
    return res

@app.route("/reply/<gpid>", methods=['POST', 'GET'])
def reply(gpid):
    """
    1. 根据场次检索出预测值，比赛双方 -- reply_search
    2. 如果有用户输入，则分析用户输入，返回一句回答；否则随机返回回答
       reply_msg_post | reply_msg_get
    3. 构造出前端需要的格式，头像采用随机头像，昵称均用专家
       reply_format
    """
    sqldict = reply_search(gpid)
    if request.method == 'POST':
        # msg = request.form.get("msg")
        # print(parameters)
        data = json.loads(request.get_data())
        msg = data.get('msg')
        answer = reply_msg_post(msg, sqldict)
    else:
        answer = reply_msg_get(sqldict)

    res = reply_format(answer)
    return json.dumps(res)



def reply_search(gpid):
    """单次检索数据库，取得该场次的预测值"""
    sqls = "select Host, Guest, Points, DPoints from Predict where GPid=%s"
    params = [gpid]
    if not server.open:
        server.connect()
    with server.cursor() as cursor:
        cursor.execute(sqls, params)
        res = cursor.fetchone()
    sqldict = {
        'host': res[0],
        'guest': res[1],
        'points': res[2] if res[2] else res[3],
        'gpid': gpid,
    }
    return sqldict

def reply_format(answer):
    sqls = "select ImgPath from Bloger"
    with server.cursor() as cursor:
        cursor.execute(sqls)
        portraits = cursor.fetchall()
        portraits = [p[0] for p in portraits]
    res = {
        "code": 0,
        "msg": "获取成功",
        "list": [
            {
                "headimgurl": random.choices(portraits),
                "nickname": "专家",
                "image": "",
                "text": answer,
                "video": "",
                "videoposter": "",
                "isreverse": False,
            },
            {
                "headimgurl": random.choices(portraits),
                "nickname": "专家",
                "image": "",
                "text": "我不认同你的观点",
                "video": "",
                "videoposter": "",
                "isreverse": True,
            },
            {
                "headimgurl": random.choices(portraits),
                "nickname": "专家",
                "image": "",
                "text": "我觉得俄罗斯必赢",
                "video": "",
                "videoposter": "",
                "isreverse": True,
            }
        ]
    }
    return res


def reply_msg_post(msg, sqldict):
    host = sqldict['host']
    guest = sqldict['guest']
    points = sqldict['points']
    if host in msg:
        msg = msg.replace(host, guest)
    elif guest in msg:
        msg = msg.replace(guest, host)
    else:
        msg = f"我认为{host}会赢"
    return msg

def reply_msg_get(sqldict):
    host = sqldict['host']
    guest = sqldict['guest']
    score_h, score_g = sqldict['points'].split('-')
    score_h, score_g = int(score_h), int(score_g)
    if score_h > score_g:
        msg = f"我认为{host}会羸，比分{score_h}:{score_g}"
    else:
        msg = f"我认为{guest}会羸多1~2分"
    return msg

if __name__ == '__main__':
    from werkzeug.contrib.fixers import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(host='localhost', port=8013, debug=True)
