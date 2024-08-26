#!/bin/bash

# 检查输入参数是否正确
if [ $# -ne 2 ]; then
  echo "Usage: $0 <filename> <hostname>"
  echo "Example: $0 traffic.sh edge-1"
  exit 1
fi

# 获取输入参数
FILE=$1
HOSTNAME=$2

# 定义主机名到IP地址的映射
declare -A HOST_MAP
HOST_MAP=(
  ["edge-1"]="115.146.92.106"
  ["edge-2"]="115.146.94.60"
  ["cloud-1"]="45.113.232.150"
  # 在此处添加更多主机名和对应的IP地址
)

# 检查主机名是否在映射中
if [ -z "${HOST_MAP[$HOSTNAME]}" ]; then
  echo "Error: Unknown hostname '$HOSTNAME'. Please define it in the script."
  exit 1
fi

# 获取对应的IP地址
TARGET_IP=${HOST_MAP[$HOSTNAME]}

# 定义SSH密钥路径
SSH_KEY_PATH="~/.ssh/id_rsa"

echo $TARGET_IP
echo $SSH_KEY_PATH
# # 执行scp命令，将文件发送到目标主机
# scp -i $SSH_KEY_PATH $FILE ubuntu@$TARGET_IP:/home/ubuntu/
# 
# # 检查scp命令是否成功
# if [ $? -eq 0 ]; then
#   echo "File '$FILE' successfully sent to $HOSTNAME ($TARGET_IP)"
# else
#   echo "Failed to send file '$FILE' to $HOSTNAME ($TARGET_IP)"
# fi
