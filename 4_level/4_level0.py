'''import threading
import time
count=1
def print_time(threadID,counter):
    while counter:
        time.sleep(1)
        print(f"线程{threadID}:{time.ctime(time.time())}")
        counter -= 1
def print_count(threadID):
    print (f"线程{threadID}：count={count}")
class MyThread(threading.Thread):
    def __init__(self,thread_ID,name):
        threading.Thread.__init__(self)
        self.thread_ID = thread_ID
        self.name = name
    def run(self):
        global count
        print(f"开始线程{self.name}")
        for i in range(5):
            with lock:
                print_count(self.thread_ID)
                count+=1
            time.sleep(1)
        print(f"{self.name}结束")
thread1=MyThread(1,"thread1")
thread2=MyThread(2,"thread2")
lock=threading.Lock()
#thread=threading.Thread(target=MyThread().run)
thread1.start()
thread2.start()
thread1.join()
thread2.join()
print("住线程结束")'''
'''import threading
import time
import queue
work_queue=queue.Queue(maxsize=10)
def producer():
    for i in range(5):
        work_queue.put(i)
        print(f"生产:{i}")
        time.sleep(0.4)
    print("生产结束")
def consumer():
    while True:
        item=work_queue.get()
        print(f"消费:{item}")
        time.sleep(1)
        work_queue.task_done()
producer_thread=threading.Thread(target=producer)
consumer_thread=threading.Thread(target=consumer,daemon=True)
producer_thread.start()
consumer_thread.start()
producer_thread.join()
work_queue.join()
print("任务结束")'''
'''from concurrent.futures import ThreadPoolExecutor
import time
def task(name):
    print(f"任务{name}开始")
    time.sleep(1)
    print(f"任务{name}结束")
    return f"结果-{name}"
with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(task,i) for i in range(3)]
    for future in futures:
        print(future.result())'''
'''from concurrent.futures import ProcessPoolExecutor,as_completed
import time
import os
def fibonacci(n):
    if n<=1:
        return n
    return fibonacci(n-1)+fibonacci(n-2)
if __name__ == '__main__':
    numbers=[30,31,32,33,34]
    print(f"CPU核心数：{os.cpu_count()}")
    start=time.time()
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures ={executor.submit(fibonacci,num):num for num in numbers}
        for future in as_completed(futures):
            num=futures[future]
            result=future.result()
            print(f"fibonacci({num})={result}")
            end=time.time()
            print(f"耗时{end-start:.2f}秒")'''
'''import asyncio
import time
async def say_after(delay,what):
    await asyncio.sleep(delay)
    print(what)
async def main():
    print("开始")
    await asyncio.gather(
        say_after(2,"hello"),
        say_after(1,"world"),
    )
    print("结束")
if __name__ == '__main__':
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"耗时：{end-start:.2f}秒")'''

'''
import asyncio
import time
async def say(name,delay):
    await asyncio.sleep(delay)
    print(f"hello {name}")
    return f"{name}完成"
''''''async def main():
    task=asyncio.create_task(say(name="Alice",delay=1))
    task2=asyncio.create_task(say(name="Bob",delay=2))
    print("等待，，")
#    result = await asyncio.gather(task,task2)
#    print(result)  
    for result in asyncio.as_completed([task,task2]):
        result = await result
        print(result)''''''
async def main():
    await say(name="Alice",delay=1)
    await say(name="Bob",delay=2)
if __name__ == '__main__':
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(end-start)
'''
'''import asyncio
async def producer(queue):
    for i in range(10):
        print(f"生产{i}")
        await queue.put(i)
        await asyncio.sleep(1)
    await queue.put(None)
async def consumer(queue):
    while True:
        item=await queue.get()
        if item is None:
            break
        print(f"消费{item}")
        await asyncio.sleep(2)
async def main():
    queue = asyncio.Queue()
    await asyncio.gather(producer(queue),consumer(queue))
asyncio.run(main())'''
'''import asyncio
import time
def blocking_io(task_id):
    print(f"{task_id}开始")
    time.sleep(1)
    print(f"{task_id}完成")
    return f"result-{task_id}"
async def main():
    start = time.time()
    tasks = [asyncio.to_thread(blocking_io,i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    duration = time.time() - start
    print(f"over spent {duration:.2f}s")
    print("结果：",results)
asyncio.run(main())'''
'''
import time
import asyncio
from concurrent.futures import ProcessPoolExecutor
def fibonacci(n):
    if n<=1:
        return n
    return fibonacci(n-1)+fibonacci(n-2)
async def main():
    start=time.perf_counter()
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=16) as pool:
        tasks=[loop.run_in_executor(pool,fibonacci,35) for _ in range(10)]
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start
        print(f"over spent {duration:.2f}s")
        print("结果：",results)
if __name__ == '__main__':
    asyncio.run(main())
'''
import socket
server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(('127.0.0.1',8080))
server.listen(1)
print("服务器启动")
conn,addr=server.accept()
print(f"连接{addr}")
data=conn.recv(1024).decode('utf-8')
conn.close()
server.close()



























