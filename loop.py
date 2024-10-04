import threading
import subprocess

# 定义一个线程任务
def run_test(index):
    print(f"Run #{index}")
    # 运行shell命令并获取输出
    result = subprocess.run(['python', 'testingmask.py'], capture_output=True, text=True)
    output = result.stdout + result.stderr  # 获取标准输出和错误输出
    
    # 检查输出中是否包含 "detection"
    if "detection" in output:
        print(f"Detection found in run #{index}")
    else:
        print(f"No detection found in run #{index}")
    
    print("------------------------")

# 创建线程列表
threads = []
num_runs = 10

# 创建并启动线程
for i in range(num_runs):
    t = threading.Thread(target=run_test, args=(i + 1,))
    threads.append(t)
    t.start()

# 等待所有线程完成
for t in threads:
    t.join()

print("All tests completed.")
