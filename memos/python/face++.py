import sys
import requests
from pprint import pprint
print = pprint

api_key = '7zuC2S1YbMWr4Tcs7l1Igk6QB41mpQwj'
api_secret = 'TOP4TM1X-ITyjXTQBbzGHB0ByG_StWBc'


url = f'https://api-cn.faceplusplus.com/facepp/v3/detect?api_key={api_key}&api_secret={api_secret}'
url = f'https://api-cn.faceplusplus.com/imagepp/v1/recognizetext?api_key={api_key}&api_secret={api_secret}'


jpg = sys.argv[1]
file = {"image_file": (jpg, open(jpg, 'rb'), 'image/jpg')}
r = requests.post(url, files=file)
print(r.json())
