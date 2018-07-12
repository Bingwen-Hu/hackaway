from flask import Flask
from flask import request



app = Flask(__name__)

# level 0
@app.route('/')
def home_page():
    html = """
    <head>
        <title>我的网站</title>
    </head>
    <body>
        这是我的网站
    </body>
    """
    return html

# level 1
@app.route('/<name>')
def sayhello(name):
    html = f"""
    <h3>hello, {name}</h3>
    """
    return html

from flask import redirect
@app.route('/baidu')
def tobaidu():
    return redirect('http://www.baidu.com')

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)