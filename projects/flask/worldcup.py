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

from answer_generative import complete_answer, vote_answer, analysis_stand

app = Flask(__name__)

SERVER_PARAMS = {
    'host': "119.84.122.135",
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
    2. 如果有用户输入，则分析用户输入，作出相应的回答；否则随机返回回答
       reply_msg_post | reply_msg_get
    3. 构造出前端需要的格式，头像采用随机头像，昵称均用专家
       reply_format

    ps: 名字长于4的作为反方，小于4的作为正方，根据实际的情况调整
    也可以在库中的stand字段中标识，0为正方，1为反方
    """
    sqldict = reply_search(gpid)
    msg = None
    if request.method == 'POST':
        # msg = request.form.get("msg")
        # print(parameters)
        data = json.loads(request.get_data())
        msg = data.get('msg')
    res = reply_format(sqldict, msg)
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

def reply_format(sqldict, msg):
    # FIXME: divide me
    host = sqldict['host']
    guest = sqldict['guest']
    score_h, score_g = sqldict['points'].split('-')
    score_h, score_g = int(score_h), int(score_g)
    origin_stand = score_h < score_g # 0 正 1 负

    sqls = "select ImgPath, Stand from Bloger where Stand = 1 or Stand = 0"
    with server.cursor() as cursor:
        cursor.execute(sqls)
        authors = cursor.fetchall()
        # note that authors[i][0] => ImgPath, [1] => Stand
        
    user_stand = 2 # 无正负
    count = 6

    # 使与原观点一致的多一些
    same = [author for author in authors if author[1] == origin_stand]
    index = list(range(len(same)))
    random.shuffle(index)
    index = index[:4]
    same = same[:4]

    diff = [author for author in authors if author[1] != origin_stand]
    index = list(range(len(diff)))
    random.shuffle(index)
    index = index[:2]
    diff = diff[:2]

    same.extend(diff)
    index = list(range(count))
    random.shuffle(index)
    authors = [same[i] for i in index]
    


    if msg is not None:
        user_stand = analysis_stand(msg, host, guest)
    # closure
    def reply_format_helper():
        resultlist = []
        for i, (portrait, stand) in enumerate(authors):
            resultlist.append({
                "headimgurl": portrait,
                "nickname": "专家",
                "image": "",
                "text": complete_answer(host, guest, stand),
                "video": "",
                "videoposter": "",
                "isreverse": stand != origin_stand,
            })
        # image
        # import base64
        i = 2
        portrait, stand = "wait", origin_stand
        # imgdata = open("test.jpg", "rb").read()
        # imgdata = base64.b64encode(imgdata)
        resultlist[i] = {
                "headimgurl": portrait,
                "nickname": "小新",
                "image": 'http://119.84.122.134:10058/wxapp/12.png',
                "text": "",
                "video": "",
                "videoposter": "",
                "isreverse": stand != origin_stand,
        }
        resultlist[i+1] = {
                "headimgurl": portrait,
                "nickname": "小新",
                "image": "",
                "text": complete_answer(host, guest, stand),
                "video": "",
                "videoposter": "",
                "isreverse": stand != origin_stand,
        }
        return resultlist

    res = {
        "code": 0,
        "msg": "获取成功",
        "list": reply_format_helper()
    }
    return res

if __name__ == '__main__':
    from werkzeug.contrib.fixers import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(host='localhost', port=8013, debug=True)
