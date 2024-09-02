
import json
import random
from typing import Dict, List
from call import Call


class Pod:
    def __init__(self, name: str, id: int, original_name: str, cpu_requests: float, memory_requests: float, num_replicas: int, type: str):
        self.name = name
        self.id = id
        self.original_name = original_name
        self.cpu_requests = cpu_requests
        self.memory_requests = memory_requests
        self.num_replicas = num_replicas
        self.node_id = -1
        self.total_bandwidth = 0
        self.type = type # service / client

    def is_instance_of(self, ms_name):
        return ms_name == self.original_name
    def get_node_id(self):
        return self.node_id
    def get_name(self):
        return self.name
    def get_type(self):
        return self.type
    def __repr__(self):
        return f"<Microservice {self.name}: {self.cpu_requests} CPU, {self.memory_requests} Memory, {self.num_replicas} replicas>"
        
# if __name__ == "__main__":
#     # 假设文件路径如下：
#     microservices_config_path = 'microservices.json'
#     calls_config_path = 'calls.json'

#     # 创建 Microservices 对象
#     microservices = Application(microservices_config_path, calls_config_path)

#     # 测试 1: 打印所有微服务实例
#     print("Test 1: 打印所有微服务实例")
#     for ms_instance in microservices.instances.values():
#         print(ms_instance)
#     print("\n")

#     # 测试 2: 调度微服务到特定节点
#     print("Test 2: 调度微服务到特定节点")
#     nodes = ["node1", "node2", "node3"]
#     for ms_instance in microservices.get_instances():
#         # 随机将微服务调度到节点
#         node = random.choice(nodes)
#         microservices.schedule_to_node(ms_instance, node)
#         print(f"Scheduled {ms_instance} to {node}")
#     microservices.calculate_microservice_bandwidth()

#     # 测试 3: 验证微服务是否正确调度到节点
#     print("\nTest 3: 验证微服务是否正确调度到节点")
#     for ms_instance in microservices.instances.values():
#         print(f"{ms_instance.name} is scheduled to {ms_instance.node_id}")
#     print("\n")

#     # 测试 4: 获取并打印邻接列表
#     print("Test 4: 获取并打印邻接列表")
#     for ms_name, adjacency in microservices.bandwidth_adj_list.items():
#         print(f"Microservice {ms_name}: {adjacency}")
#     print("\n")

#     # 测试 5: 计算每个微服务的总带宽使用量
#     print("Test 5: 计算每个微服务的总带宽使用量")
#     for ms_instance in microservices.instances:
#         print(f"{ms_instance.name}: Total Bandwidth = {ms_instance.total_bandwidth} MB/s")
    