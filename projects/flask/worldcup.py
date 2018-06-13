# coding: utf-8
"""
python:  2.7.6
pymongo: 2.7.1
flask:   0.10.x

"""
from flask import Flask
from flask import request
import pymysql
import json

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
        sqldict = predict_parser(result)
    jsondata = json.dumps(sqldict)
    return jsondata


def predict_parser(result):
    return {
        'code': 0,
        'GPid': result[0],
        'host': result[1],
        'guest': result[2],
        'points': result[3] if result[3] else result[6],
        'rate': result[4] if result[4] else result[5],
    }

@app.route("/reply/<gpid>", methods=['POST', 'GET'])
def reply(gpid):
    host_guest_points = reply_search(gpid)
    if request.method == 'POST':
        msg = request.form.get("msg")
        # print(parameters)
        ans = reply_msg_post(msg, host_guest_points)
    else:
        ans = reply_msg_get(host_guest_points)

    res = {
        "code": 0,
        "msg": "获取成功",
        "list": [
            {
                "headingurl": "",
                "nickname": "",
                "image": "",
                "text": ans,
                "video": "",
                "videoposter": "",
            }
        ]
    }    
    return json.dumps(res)



def reply_search(gpid):
    sqls = "select Host, Guest, Points, DPoints from Predict where GPid=%s"
    params = [gpid]
    if not server.open:
        server.connect()
    with server.cursor() as cursor:
        cursor.execute(sqls, params)
        res = cursor.fetchone()
    return {
        'host': res[0],
        'guest': res[1],
        'points': res[2] if res[2] else res[3],
    }


def reply_msg_post(msg, host_guest_points):
    host = host_guest_points['host']
    guest = host_guest_points['guest']
    points = host_guest_points['points']
    if host in msg:
        msg = msg.replace(host, guest)
    elif guest in msg:
        msg = msg.replace(guest, host)
    else:
        msg = f"我认为{host}会赢"
    return msg

def reply_msg_get(host_guest_points):
    host = host_guest_points['host']
    guest = host_guest_points['guest']
    score_h, score_g = host_guest_points['points'].split('-')
    score_h, score_g = int(score_h), int(score_g)
    if score_h > score_g:
        msg = f"我认为{host}会羸，比分{score_h}:{score_g}"
    else:
        msg = f"我认为{guest}会羸多1~2分"
    return msg

if __name__ == '__main__':
    from werkzeug.contrib.fixers import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(host='localhost', port=38016, debug=True)
