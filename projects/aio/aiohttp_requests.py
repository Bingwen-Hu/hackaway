import aiohttp
import asyncio
import json



# global dict settings
urls = {
    "shezheng": "http://",
    "shehuang": 'http://',
}


# NOTE: can be remove in productive code
# this is ok too
# form = aiohttp.FormData()
# form.add_field('image',
#               open('/home/mory/data/face_test/10.jpg', 'rb'),
#               filename='image.png',
#               content_type='image/jpeg')
def requests_all():
    # files = [{'image': open('/home/mory/data/face_test/10.jpg', 'rb')} for _ in urls]
    files = [{'image': open('/home/mory/Downloads/1.jpg', 'rb')} for _ in urls]
    results = {}
    async def requests(key, files):
        async with aiohttp.ClientSession() as session:
            async with session.post(urls[key], data=files) as resp:
                content = await resp.read()
                results.update({key: json.loads(content)})
    loop = asyncio.get_event_loop()
    tasks = [asyncio.ensure_future(requests(key, files_)) for key, files_ in zip(urls, files)]
    loop.run_until_complete(asyncio.wait(tasks))
    return results

def face_parse(result):
    result_ = result['results']
    if not result_:
        return {'infotype': 'shezheng', 'rate': 0.0, 'content': ''}
    positive = [r for r in result_ if r['label'] != 'unknown']
    positive.sort(key=lambda x: x['distance'])
    rate = min(1, 1 - positive[0]['distance'] / 2 + (len(positive) - 1) * 0.1)
    content = " ".join(p['label'] for p in positive)
    return {'infotype':'shezheng', 'rate': rate, 'content': content} 



def construct(results):
    rets = []
    for key, result in results.items():
        if key == 'shezheng':
            ret = face_parse(result)
        elif key == 'shehuang':
            ret = {'infotype':'shehuang', 'rate': result['pos'], 'content': ''} 
        rets.append(ret)
    return json.dumps(rets)

if __name__ == '__main__':
    results = requests_all()
    results = construct(results)