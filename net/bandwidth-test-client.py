import socket
import time
import argparse
import json
import os

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="客户端 - 发送不同大小的数据包并记录延迟时间")
parser.add_argument('-ip', '--ip_address', type=str, required=True, help="服务器的IP地址")
parser.add_argument('-p', '--port', type=int, default=12345, help="服务器的端口号（默认12345）")
parser.add_argument('-o', '--output', type=str, default="latency_results.json", help="结果输出文件名（默认latency_results.json）")
parser.add_argument('-r', '--rate', type=str, required=True, help="测试的带宽限制（例如 '50mbit'）")

args = parser.parseArgs()
# 定义要测试的不同数据包大小（以字节为单位）
data_sizes = [
    0,             # 0字节，测量基础延迟
    1 * 1024,      # 1KB
    10 * 1024,     # 10KB
    100 * 1024,    # 100KB
    1 * 1024 * 1024,   # 1MB
    5 * 1024 * 1024,   # 5MB
    10 * 1024 * 1024,  # 10MB
    50 * 1024 * 1024,  # 50MB
    100 * 1024 * 1024  # 100MB
]

# 初始化结果列表
results = []

# 进行测试并收集结果
for size in data_sizes:
    # 创建客户端 socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.ip_address, args.port))  # 连接到服务器的IP和端口

    # 生成指定大小的数据
    data = b'0' * size

    # 记录发送开始时间
    start_time = time.time()

    # 使用 sendall 确保数据全部发送
    client_socket.sendall(data)

    # 记录发送结束时间
    end_time = time.time()

    # 计算延迟时间
    latency = end_time - start_time

    # 计算带宽 (Mbps)
    if latency > 0:
        bandwidth_mbps = (size * 8) / (latency * 1_000_000)
    else:
        bandwidth_mbps = float('inf')  # 在极端情况下避免除以零

    # 添加结果到列表
    results.append({
        "size_bytes": size,
        "latency_seconds": latency,
        "bandwidth_mbps": bandwidth_mbps
    })

    # 打印结果到控制台
    print(f"发送 {size / 1024} KB 数据包所需的时间: {latency:.6f} 秒，带宽: {bandwidth_mbps:.2f} Mbps")

    # 关闭连接
    client_socket.close()

# 如果输出文件存在，加载已有内容
if os.path.exists(args.output):
    with open(args.output, 'r') as file:
        try:
            existing_data = json.load(file)
            if not isinstance(existing_data, dict):
                existing_data = {}
        except json.JSONDecodeError:
            existing_data = {}
else:
    existing_data = {}

# 将新结果添加到以 `rate` 命名的键中
existing_data[args.rate] = results

# 将更新后的数据写回文件
with open(args.output, 'w') as file:
    json.dump(existing_data, file, indent=4)

# 打印完成信息
print(f"所有测试结果已保存到 {args.output} 的 '{args.rate}' 键中")
