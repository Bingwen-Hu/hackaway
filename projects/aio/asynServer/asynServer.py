from aiohttp import web
import aiohttp_jinja2
import jinja2

async def hello(request):
    return web.Response(text='Hello world!')


async def passvar(request):
    name = request.match_info.get('name', 'unknown')
    return web.json_response({'name': name})


async def login(request):
    data = await request.post()
    login = data['login']
    password = data['password']
    return web.json_response({'login': login, 'password': password})

@aiohttp_jinja2.template('login.html')
async def get_login(request):
    return {}



# run with gunicorn
async def face_server():
    app = web.Application()
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('templates'))
    app.router.add_get('/', hello)
    # app.router.add_get('/{name}/{ann}', passvar)
    app.router.add_get('/login', get_login)
    app.router.add_post('/login', login)
    return app