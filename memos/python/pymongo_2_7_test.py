# pymongo 2.7.1

from pymongo import Connection
import json

conn = Connection(host='localhost', port='27017')
db = conn['test']
collection = db['passport']
cursor = collection.find({"addon": "xxxx-xxxx-xxxx"})  # multi result

cursor_list = list(cursor)
for d in cursor:
    d.pop('_id')

jsonlist = [json.dumps(c) for c in cursor_list]
