import aiohttp
import asyncio
import requests




# ok
url = "http://"
files = {'image': open('/home/mory/data/face_test/10.jpg', 'rb')}

# this is ok too
form = aiohttp.FormData()
form.add_field('image',
               open('/home/mory/data/face_test/10.jpg', 'rb'),
               filename='image.png',
               content_type='image/jpeg')

async def download(url, files):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=files) as resp:
            content = await resp.read()
            print(content)

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(download(url, files))]
loop.run_until_complete(asyncio.wait(tasks))
