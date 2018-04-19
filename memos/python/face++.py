import requests
from pprint import pprint
print = pprint

api_key = 'api_key'
api_secret = 'api_secret'

url = f'https://api-cn.faceplusplus.com/facepp/v3/detect?api_key={api_key}&api_secret={api_secret}'
jpg = "4.jpg"
file = {"image_file": (jpg, open(jpg, 'rb'), 'image/jpg')}
r = requests.post(url, files=file)
print(r.json())