import socket
import threading

# 定义处理客户端连接的函数
def handle_client(conn, addr):
    print(f"已连接到客户端: {addr}")
    try:
        while True:
            # 期望接收的数据大小
            expected_data_size = 5 * 1024 * 1024  # 5MB

            # 使用 MSG_WAITALL 确保接收到完整的数据
            data = conn.recv(expected_data_size, socket.MSG_WAITALL)

            if not data:
                # 如果接收到空数据，则表示客户端已关闭连接
                print(f"客户端 {addr} 已关闭连接。")
                break

            print(f"从客户端 {addr} 接收到的数据大小: {len(data)} 字节")

    finally:
        conn.close()

# 创建服务器端 socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 12345))  # 绑定到所有接口的端口12345
server_socket.listen(5)  # 允许最多5个待处理连接

print("服务器正在等待连接...")

# 持续接受新的客户端连接
try:
    while True:
        conn, addr = server_socket.accept()
        # 每个客户端连接都创建一个新的线程进行处理
        client_thread = threading.Thread(target=handle_client, args=(conn, addr))
        client_thread.start()

except KeyboardInterrupt:
    print("服务器手动中止。")

# 关闭服务器 socket
server_socket.close()