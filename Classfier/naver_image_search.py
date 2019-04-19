import requests
import urllib.request as req
import urllib.parse as parse
import json
import os
from os import makedirs

headers = {
    "X-Naver-Client-Id" : "gWE1ouEdgfhh_7gwTIym",
    "X-Naver-Client-Secret" : "UcMdskj0Kj"
}

url = "https://openapi.naver.com/v1/search/image"

params = {
    "query" : "아이린",
    "start" : 1002,
    "display" : 100
}

res = requests.get(url, headers=headers, params=params)
print(res.status_code)

list = json.loads(res.text)
print(list)

for ix, item in enumerate(list["items"]):
    title = item["title"]
    link = item["link"]
    info = parse.urlparse(link)
    fileName = os.path.split(info.path)[1]
    
    print(ix, title, fileName)
    print("링크",link)

    savepath = "./img/irin/"

    if not os.path.exists(savepath):
        makedirs(savepath)

    if not fileName.endswith(".jpg"):
       continue
    
    # 다운로드
    mem = req.urlopen(link).read()

    # 파일로 저장하기
    with open(savepath+fileName, mode="wb") as f:
        f.write(mem)
        print("저장되었습니다.!")