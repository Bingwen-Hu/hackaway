from aiohttp import web


async def hello(request):
    return web.Response(text='Hello world!')


async def passvar(request):
    name = request.match_info.get('name', 'unknown')
    return web.json_response({'name': name})


# run with gunicorn
async def face_server():
    app = web.Application()
    app.router.add_get('/', hello)
    app.router.add_get('/{name}/{ann}', passvar)
    return app