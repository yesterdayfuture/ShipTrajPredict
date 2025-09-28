'''

学习异步库的使用

'''

import asyncio
import time
import threading

cur_progress = 0

async def task():
    print(" task start ...")
    await asyncio.sleep(5)
    print(" task finish ...")


# 回调函数
def collback(future):
    # 使用全局变量
    global cur_progress

    s = future.get_name().split(" ")

    with threading.Lock() as cur_lock:
        cur_progress = cur_progress + int(s[1])

    print(f"任务 {s[0]} 已完成！当前任务总进度={cur_progress}/{s[2]}")

async def main():

    strat_time = time.time()
    print(" main start ...")
    tasks = []
    for i in range(5):
        cur_task = asyncio.create_task(task(),name=f'task{i} 20 100')
        #添加回调
        cur_task.add_done_callback(collback)
        tasks.append(cur_task)
        await asyncio.sleep(1)

    await asyncio.gather(*tasks)

    print(" main finish ...")
    print(f' 运行时间 = {time.time() - strat_time}s')

    # 判断所有任务是否完成
    done, pending = await asyncio.wait(tasks)
    for cur_task in done:
        print(f"任务 {cur_task.get_name()} 已完成，结果: {cur_task.result()}")


if __name__ == "__main__":
    asyncio.run(main())