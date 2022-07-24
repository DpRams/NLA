import requests

url = 'http://127.0.0.1:8000/pipeline/data/upload'
file = {'file': open(r'C:\Users\ZZ01GI858\Desktop\research\project\static\img\acc.png', 'rb')}
resp = requests.post(url=url, files=file) 
print(resp.json())