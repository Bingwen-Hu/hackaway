import requests
import os.path

urls = {
    "shezheng": 'http://',
    "shehuang": 'http://',
}


def requests_all(filepath):
    files = [{'image': (os.path.basename(filepath),open(filepath, 'rb'), 'image/jpeg')} for _ in urls]
    def request_(key, files):
        resp = requests.post(urls[key], files=files)
        return key, resp.json()
    result = list(map(request_, urls, files))
    return result


def face_parse(result):
    result_ = result['results']
    if not result_:
        return {'infotype': 'shezheng', 'rate': 0.0, 'content': ''}
    positive = [r for r in result_ if r['label'] != 'unknown']
    if not positive:
        return {'infotype': 'shezheng', 'rate': 0.0, 'content': ''}
    positive.sort(key=lambda x: x['distance'])
    rate = min(1, 1 - positive[0]['distance'] / 2 + (len(positive) - 1) * 0.1)
    content = " ".join(p['label'] for p in positive)
    return {'infotype':'shezheng', 'rate': rate, 'content': content} 

def construct(results):
    rets = []
    for key, result in results:
        if key == 'shezheng':
            ret = face_parse(result)
        elif key == 'shehuang':
            ret = {'infotype':'shehuang', 'rate': result['pos'], 'content': ''} 
        rets.append(ret)
    return rets

def multitask(filepath):
    results = requests_all(filepath)
    results = construct(results)
    return results


if __name__ == "__main__":
    filepath = 'zhou.jpg'
    rets = multitask(filepath)
    print(rets)