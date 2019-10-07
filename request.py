import requests
import json


URL = 'http://127.0.0.1:5000/predict'
PARAMS = {'exp': 1.8}

r = requests.get(url=URL, params=PARAMS)

try:
    print(r.json())
except(json.decoder.JSONDecodeError, ValueError):
    print("N'est pas JSON")
























