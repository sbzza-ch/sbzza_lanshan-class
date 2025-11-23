import socket
client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client.connect(('127.0.0.1',8080))
client.send("hello".encode('utf-8'))
data=client.recv(1024).decode('utf-8')
print(f"收到{data}")
client.close()
'''import requests
resp=requests.get("http://httpbin.org/get")
print("状态码：",resp.status_code)
print("响应体",resp.text)
data={"name":"Alice","age":50}
resp=requests.post("http://httpbin.org/post",json=data)
print("响应JSON",resp.json())'''
'''import requests
url = "https://www.bilibili.com/video/BV1GJ411x7h7/"
params = {
    "spm_id_from": "333.337.search-card.all.click",
    "vd_source": "c91109028f20fc7ac11b64fad50e8706"
}
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.bilibili.com/",
}
resp = requests.get(url, params=params, headers=headers)
print("请求的完整 URL:", resp.url)
print("状态码:", resp.status_code)
print("返回内容片段:", resp.text[:200]'''
