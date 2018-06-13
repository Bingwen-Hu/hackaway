import requests
import json


def test_reply():
    url = "http://localhost:8013/reply/1"
    # data = {"msg": "我觉得俄罗斯会赢"}
    # resp = requests.post(url, data=data)
    data_json = json.dumps({"msg": "我觉得俄罗斯会赢"})
    resp_json = requests.post(url, data=data_json)
    print(json.loads(resp_json.text))


def test_predict():
    url = "http://localhost:8013/predict/1"
    # data = {"msg": "我觉得俄罗斯会赢"}
    # resp = requests.post(url, data=data)
    resp_json = requests.get(url)
    print(json.loads(resp_json.text))

if __name__ == '__main__':
    test_predict()
    test_reply()