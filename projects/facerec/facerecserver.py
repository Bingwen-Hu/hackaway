import os
import jinja2
import aiohttp_jinja2
from aiohttp import web

import facerec


# routes对象提供修饰器，用于配置路由
routes = web.RouteTableDef()

# 通过url指定不同的检测模式
@routes.view('/facerec/{mode}/detect')
class RecognizeView(web.View):
    async def post(self):
        # 读取post请求中上传的文件，可以是图片、视频等
        reader = await self.request.multipart()
        field = await reader.next()
        # 验证字段名为`image`， 跟`get`方法的html文件字段对应
        assert field.name == 'image'

        # 将post过来的文件保存在当前路径下
        filename = os.path.basename(field.filename)
        with open(filename, 'wb') as f:
            while True:
                chunk = await field.read_chunk() # 每次读一小块
                # chunk为空说明已经读写完成，退出循环 
                if not chunk:
                    break
                f.write(chunk)

        # 将文件路径传给api功能函数
        result = {
            'state': 10010,
            'message': "not found",
            'data': [],
        }
        info = facerec.detect(filename, mode=self.request.match_info['mode'])
        if info:
            result = {
                'state': 10000,
                'message': 'succeed',
                'data': info,
            }
        # (可选)移除所保存的文件
        os.remove(filename)
        # 以json的格式返回结果
        return web.json_response(result)

    # get方法，因为不需要传递变量给模板进行渲染，所以直接返回空字典
    @aiohttp_jinja2.template('detect.html')
    async def get(self):
        return {'mode': self.request.match_info['mode']}


# 通过gunicorn来运行：gunicorn facerecserver:facerec_server --worker-class aiohttp.GunicornWebWorker
async def facerec_server():
    """配置服务端应用，包括创建`app`对象，配置模板和路由。"""
    app = web.Application()
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('templates')) # 配置jinja2所渲染的模板文件
    app.router.add_routes(routes) # 添加路由配置
    return app

