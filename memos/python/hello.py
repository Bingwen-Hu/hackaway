"""
A very gentle beginning of Python Web Development
"""

from flask import (Flask, 
                   make_response, 
                   redirect,
                   abort)

# using __name__ to define a root of the service
app = Flask(__name__)


@app.route('/')
def index():
    response = make_response("<h1>This document carries a cookie!</h1>")
    response.set_cookie('answer', '42')
    response.status = '200 not oK' # must be str
    return response
    
# name is used as variable, a status code is accepted
@app.route('/hello/<name>')
def hello(name):
    return "<h2>Hello %s<h2>" % name, 500


@app.route('/r')
def r():
    return redirect("https://www.baidu.com")

# abort will turn the control to web server
@app.route("/abort/<id>")
def get_user(id):
    if int(id) > 6:
        abort(404)
    return "<h2>your id is %s<h2>" % id

# the third party extension of flask is used as this
from flask_script import Manager
manager = Manager(app)

if __name__ == "__main__":
    manager.run()
