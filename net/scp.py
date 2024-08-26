import argparse
import subprocess
import sys

# 定义主机名到IP地址的映射
HOST_MAP = {
    "edge-1": "115.146.92.106",
    "edge-2": "115.146.94.60",
    "cloud-1": "45.113.232.150",
    # 在此处添加更多主机名和对应的IP地址
}

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="通过 SCP 发送文件到指定主机")
parser.add_argument('filename', type=str, help="要发送的文件名")
parser.add_argument('hostname', type=str, help="目标主机名（例如 edge-1, edge-2, cloud-1 等）")

args = parser.parse_args()

# 检查主机名是否在映射中
if args.hostname not in HOST_MAP:
    print(f"Error: Unknown hostname '{args.hostname}'. Please define it in the script.")
    sys.exit(1)

# 获取对应的IP地址
target_ip = HOST_MAP[args.hostname]

# 定义SSH密钥路径
ssh_key_path = "~/.ssh/id_rsa"

# 构建scp命令
scp_command = ["scp", "-i", ssh_key_path, args.filename, f"ubuntu@{target_ip}:/home/ubuntu/"]

print(scp_command)
# 执行scp命令并检查结果
try:
    subprocess.run(scp_command, check=True)
    print(f"File '{args.filename}' successfully sent to {args.hostname} ({target_ip})")
except subprocess.CalledProcessError as e:
    print(f"Failed to send file '{args.filename}' to {args.hostname} ({target_ip})")
    print(f"Error: {e}")
