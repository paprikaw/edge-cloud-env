import argparse
import subprocess
import sys

# 定义主机名到IP地址的映射
HOST_MAP = {
    "tb-client-vm-2-1": "172.26.133.107",
    "tb-cloud-vm-8-1": "172.26.129.236",
    "tb-cloud-vm-8-2": "172.26.133.27",
    "tb-edge-vm-4-1": "172.26.133.218",
    "tb-edge-vm-4-2": "172.26.133.227",
    "tb-edge-vm-2-1": "172.26.132.14",
    "tb-edge-vm-2-2": "172.26.132.147",
}
ssh_key_path = "~/.ssh/id_rsa"

def send_file(filename, hostname, ip):
    scp_command = ["scp", "-i", ssh_key_path, filename, f"ubuntu@{ip}:/home/ubuntu/"]
    print(scp_command)
    try:
        subprocess.run(scp_command, check=True)
        print(f"文件 '{filename}' 成功发送到 {hostname} ({ip})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"发送文件 '{filename}' 到 {hostname} ({ip}) 失败")
        print(f"错误: {e}")
        return False

def send_to_multiple_hosts(filenames, host_filter=None):
    for hostname, ip in HOST_MAP.items():
        if host_filter is None or host_filter(hostname):
            for filename in filenames:
                send_file(filename, hostname, ip)

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="通过 SCP 发送文件到指定主机")
parser.add_argument('filenames', type=str, nargs='+', help="要发送的文件名（可以是多个）")
parser.add_argument('--host', type=str, default='default', help="目标主机名（例如 edge-1, edge-2, cloud-1 等）")
parser.add_argument('--all', action='store_true', help="发送到所有主机")
parser.add_argument('--all-edge', action='store_true', help="发送到所有边缘节点")
parser.add_argument('--all-cloud', action='store_true', help="发送到所有云节点")

args = parser.parse_args()

if args.all:
    send_to_multiple_hosts(args.filenames)
elif args.all_edge:
    send_to_multiple_hosts(args.filenames, lambda h: h.startswith("tb-edge-"))
elif args.all_cloud:
    send_to_multiple_hosts(args.filenames, lambda h: h.startswith("tb-cloud-"))
elif args.host in HOST_MAP:
    for filename in args.filenames:
        send_file(filename, args.host, HOST_MAP[args.host])
elif args.host == 'default':
    print("错误: 未指定目标主机。请使用 --host 参数指定目标主机或使用 --all, --all-edge, --all-cloud 标志。")
    sys.exit(1)
else:
    print(f"错误: 未知的主机名 '{args.host}'。请在脚本中定义它。")
    sys.exit(1)
