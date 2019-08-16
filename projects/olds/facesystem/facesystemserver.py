import os
import jinja2
import aiohttp_jinja2
from aiohttp import web
from uuid import uuid1

import facesystem


# routes对象提供修饰器，用于配置路由
routes = web.RouteTableDef()

@routes.view('/facesystem/recognize')
class RecognizeView(web.View):
    async def post(self):
        # 读取post请求中上传的文件，可以是图片、视频等
        reader = await self.request.multipart()
        field = await reader.next()
        # 验证字段名为`image`， 跟`get`方法的html文件字段对应
        assert field.name == 'image'

        # 将post过来的文件保存在当前路径下
        _, ext = os.path.splitext(field.filename)
        filename = f"{uuid1()}{ext}"
        with open(filename, 'wb') as f:
            while True:
                chunk = await field.read_chunk() # 每次读一小块
                # chunk为空说明已经读写完成，退出循环 
                if not chunk:
                    break
                f.write(chunk)

        # 将文件路径传给api功能函数
        try:
            face, points = facesystem.face_detect(filename)
        except:
            face = None
        if face is None:
            result = {
                "state": 10011, 
                "message": 'no faces detected',
                "data": [],
            }
        else:
            info = facesystem.face_recognize(face)
            if info: 
                info['points'] = points
                result = {
                    'state': 10000,
                    'message': 'success',
                    'data': [info],
                }
            else:
                result = {
                    'state': 10010,
                    'message': "face unregistered",
                    'data': [],
                }

        # (可选)移除所保存的文件
        os.remove(filename)
        # 以json的格式返回结果
        return web.json_response(result)

    # get方法，因为不需要传递变量给模板进行渲染，所以直接返回空字典
    @aiohttp_jinja2.template('recognize.html')
    async def get(self):
        return {}

@routes.view('/facesystem/register')
class RegisterView(web.View):
    async def post(self):
        reader = await self.request.multipart()
        # verify inputs
        field = await reader.next()
        assert field.name == 'image'

        filename = os.path.basename(field.filename)
        with open(filename, 'wb') as f:
            while True:
                chunk = await field.read_chunk() 
                if not chunk: 
                    break
                f.write(chunk)
        # parse other information        
        try:
            face, points = facesystem.face_detect(filename)
        except:
            face = None
        if face is None:
            result = {
                "state": 10011, 
                "message": 'no faces detected',
                'data': []
            }
        else:
            jsoninfo = {} 
            while True:
                field = await reader.next()
                if field is None:
                    break
                content = await field.read()
                jsoninfo[field.name] = content.decode()
            
            # perform recognization
            if facesystem.face_register(face, jsoninfo):
                result = {
                    "state": 10000, 
                    "message": 'success',
                    'data': [],
                }
            else:
                result = {
                    "state": 10010, 
                    "message": 'face already exists',
                    'data': []
                }
        # remove it
        os.remove(filename)        
        return web.json_response(result)

    @aiohttp_jinja2.template('register.html')
    async def get(self):
        return {}


# 通过gunicorn来运行：gunicorn facesystemserver:facesystem_server --worker-class aiohttp.GunicornWebWorker
async def facesystem_server():
    """配置服务端应用，包括创建`app`对象，配置模板和路由。"""
    app = web.Application()
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('templates')) # 配置jinja2所渲染的模板文件
    app.router.add_routes(routes) # 添加路由配置
    return app

