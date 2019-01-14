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


@routes.view('/upload')
class UploadView(web.View):
    async def post(self):
        reader = await self.request.multipart()
        # verify inputs
        field = await reader.next()
        assert field.name == 'fileContent'
        
        with open(field.filename, 'wb') as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)
        return web.Response(text="haha")

    @aiohttp_jinja2.template('upload.html')
    async def get(self):
        return {}

@routes.view('/image')
class ShowImage(web.View):
    @aiohttp_jinja2.template('image.html')
    async def get(self):
        return {}

@routes.view('/audio')
class ShowImage(web.View):
    @aiohttp_jinja2.template('audio.html')
    async def get(self):
        return {}

routes.static('/statics', './statics')

# run with gunicorn
async def run_server():
    app = web.Application()
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('templates'))
    app.router.add_routes(routes)
    return app

# gunicorn asynServer:run_server --worker-class aiohttp.GunicornWebWorker