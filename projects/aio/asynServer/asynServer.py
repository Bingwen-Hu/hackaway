from aiohttp import web
import aiohttp_jinja2
import jinja2


routes = web.RouteTableDef()

@routes.get('/')
async def hello(request):
    return web.Response(text='Hello world!')

# @routes.get('/{name}')
async def passvar(request):
    name = request.match_info.get('name', 'unknown')
    return web.json_response({'name': name})

@routes.view('/login')
class LoginView(web.View):
        
    async def post(self):
        data = await self.request.post()
        login = data['login']
        password = data['password']
        return web.json_response({'login': login, 'password': password})

    @aiohttp_jinja2.template('login.html')
    async def get(self):
        return {}



# run with gunicorn
async def face_server():
    app = web.Application()
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('templates'))
    app.router.add_routes(routes)
    return app