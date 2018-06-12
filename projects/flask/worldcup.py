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
    


def predict_parser(gpid, args=None):
    pass




if __name__ == '__main__':
    app.run(host='localhost', port=38016, debug=True)
