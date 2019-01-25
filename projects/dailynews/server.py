# coding: utf-8
"""
python:  3.6
flask:   1.0.2

"""
from flask import Flask
from flask import request, jsonify, render_template, redirect
import requests

from datetime import datetime

app = Flask(__name__)



def get_weather():
    url = 'http://wthrcdn.etouch.cn/weather_mini?city=广州'
    resp = requests.get(url)
    result = resp.json()
    today_weather = result['data']['forecast'][0]

    def format_helper(weather):
        date = datetime.now().strftime('%Y/%m/%d')
        weekday = weather['date'][-3:]
        high = weather['high'][-3:]
        low = weather['low'][-3:]
        temperature = '{}~{}'.format(low, high)
        wtype = '天气{}'.format(weather['type'])
        return '{} {} {} {}'.format(date, weekday, wtype, temperature)

    return format_helper(today_weather)


@app.route('/dailynews', methods=["GET"])
def locate():
    data = [['天气真好', 'http://url']]
    weather = get_weather()
    return render_template('dailynews.html', news=data, weather=weather)

if __name__ == '__main__':
    app.run(host='localhost', port=38016, debug=True)
