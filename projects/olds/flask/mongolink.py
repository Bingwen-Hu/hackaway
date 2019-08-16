# coding: utf-8
"""
python:  2.7.6
pymongo: 2.7.1
flask:   0.10.x

"""
from flask import Flask
from flask import request
from pymongo import Connection
import json

app = Flask(__name__)


@app.route('/<db>/<collection>')
def hello_world(db, collection):
    args = request.args
    if len(args) == 0:
        return "Only support keywords query"
    args_ = {k.decode('utf-8'): v for (k, v) in args.items()}
    jsonlist = mongodb_parser(db, collection, args_)
    return jsonlist


def mongodb_parser(db, collection, args=None):
    conn = Connection()
    db = conn[db]
    collection = db[collection]
    res = collection.find(args)
    reslist = list(res)
    for r in reslist:
        r.pop('_id')
    jsonlist = json.dumps({'result': reslist})
    conn.close()
    return jsonlist


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=38016, debug=False)
