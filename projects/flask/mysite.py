from flask import Flask
from flask import request



app = Flask(__name__)

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


if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)