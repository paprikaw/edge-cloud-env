import json
import random
from scipy.interpolate import interp1d
from microservice import Application
import parser
from typing import Dict

class MicroserviceEnvironment:
    def __init__(self, profiling_path, node_path, microservices_config_path, calls_config_path):
        self.profiling_data = self._load_profiling_data(profiling_path)
        # self.cluster = self._load_cluster_data(cluster_path)
        self.nodes = self._initialize_nodes(node_path) # 初始化节点信息
        self.ms_apps: Dict[str,Application] = {}

    def _load_profiling_data(self, path):
        with open(path, 'r') as json_file:
            return json.load(json_file)

    def _load_dependency_data(self, path):
        with open(path, 'r') as json_file:
            return json.load(json_file)

    def _load_cluster_data(self, path):
        with open(path, 'r') as json_file:
            return json.load(json_file)

    # def _initialize_nodes(self):
    #     nodes = {}
    #     for node_type, latency_info in self.profiling_data["nodes"].items():
    #         bandwidth_usages = list(map(int, latency_info.keys()))
    #         latencies = list(latency_info.values())
    #         latency_func = interp1d(bandwidth_usages, latencies, kind='linear', fill_value="extrapolate")
    #         nodes[node_type] = latency_func
    #     return nodes

    def _initialize_nodes(self, nodes_config_path):
        """从配置文件中初始化节点资源"""
        with open(nodes_config_path, 'r') as json_file:
            nodes_config = json.load(json_file)

        nodes = {}
        node_id = 1
        # 处理 cluster_setup 中的节点生成配置
        for layer, node_types in nodes_config["cluster_setup"].items():
            for node_type, config in node_types.items():
                # 确定要生成的节点数量
                node_count = random.randint(config["count_range"][0], config["count_range"][1])

                for _ in range(node_count):
                    cpu_availability = parser.parse_cpu_requests(random.choice(config["cpu_availability"]))
                    memory_availability = parser.parse_memory(random.choice(config["memory_availability"]))
                    bandwidth_utilization = parser.parse_percentage(random.choice(config["bandwidth_utilization"]))
                    bandwidth = parser.parse_bandwidth(nodes_config["node_type"][node_type]["bandwidth"])
                    cpu_type = nodes_config["node_type"][node_type]["cpu_type"]

                    # 创建节点并存入字典
                    node_name = f"{node_type}_{node_id}"
                    nodes[node_name] = {
                        "cpu_type": cpu_type,
                        "cpu_availability": cpu_availability,
                        "memory_availability": memory_availability,
                        "bandwidth_utilization": bandwidth_utilization,
                        "bandwidth": bandwidth
                    }

                    print(f"Initialized node: {node_name} with CPU Type: {cpu_type}, CPU: {cpu_availability}, Memory: {memory_availability}, Bandwidth: {bandwidth}, Bandwidth Utilization: {bandwidth_utilization}")

                    node_id += 1

        return nodes 


    def _claim_resource(self, node_id: str, cpu_requests: int, memory_requests: int):
        """在节点上申请资源"""
        node_info = self.nodes[node_id]
        if node_info["cpu_availability"] < cpu_requests or node_info["memory_availability"] < memory_requests:
            raise Exception("Insufficient resources on the node")
        node_info["cpu_availability"] -= cpu_requests
        node_info["memory_availability"] -= memory_requests

    def _release_resource(self, node_id: str, cpu_requests: int, memory_requests: int):
        """释放节点上的资源"""
        node_info = self.nodes[node_id]
        node_info["cpu_availability"] += cpu_requests
        node_info["memory_availability"] += memory_requests

    def load_ms_app(self, ms_config_path: str, calls_config_path: str, ms_name):
        """加载微服务配置"""

        # handle name conflict
        if ms_name in self.ms_apps:
            raise Exception(f"Microservice {ms_name} already exists")
        self.ms_apps[ms_name] = Application(ms_config_path, calls_config_path, ms_name)

    def deploy(self, ms_app_name:str) -> bool:
        """
        随机部署微服务到集群中的节点上，返回是否部署成功
        这个部署过程应该是一个transaction，如果没有找到合适的节点，应该回滚
        """

        # 当前的microservice必须要
        ms_app = self.ms_apps[ms_app_name]
        assert(ms_app.deployState == "Undeployed") # 只有没有被部署的应用才能被部署

        # commit log用于记录没有提交的部署
        commit_log = []

        for ms_instance in ms_app.instances.values():
            # 寻找cpu和memory都满足的节点
            found = False
            for node_id, node_info in self.nodes.items():
                if node_info["cpu_availability"] >= ms_instance.cpu_requests and node_info["memory_availability"] >= ms_instance.memory_requests:
                    commit_log.append((ms_instance, node_id))
                    self._claim_resource(node_id, ms_instance.cpu_requests, ms_instance.memory_requests)
                    found = True
                    break
            # 没有找到合适的节点，使用commit_log回滚
            if not found:
                for ms_instance, node_id in commit_log:
                    self._release_resource(node_id, ms_instance.cpu_requests, ms_instance.memory_requests)
                return False

        # 所有instance都找到的节点，commit
        for ms_instance, node_id in commit_log:
            ms_app.schedule_instance_to_node(ms_instance, node_id)

        assert(ms_app.deployState == "Deployed") # 此时所有instance都应该被部署
        return True

    def start_traffic(self, ms_app_name:str):
        """开始模拟流量，主要是计算对应的带宽压力"""

    def schedule_microservices(self, instance_id: str, node_id: str):
        """将微服务调度到集群中的节点"""
        node_ids = list(self.nodes.keys())
        for ms_instance in self.microservices.get_instances():
            node_id = random.choice(node_ids)
            self.microservices.schedule_to_node(ms_instance, node_id)
            print(f"Scheduled {ms_instance} to {node_id}")



if __name__ == "__main__":
    # 假设文件路径如下：
    profiling_path = 'default_profile.json'
    microservices_config_path = 'microservices.json'
    calls_config_path = 'calls.json'
    node_config = 'nodes.json'

    # 创建 MicroserviceEnvironment 对象，并且新建一个微服务应用
    env = MicroserviceEnvironment(profiling_path, node_config, microservices_config_path, calls_config_path)
    env.load_ms_app(microservices_config_path, calls_config_path, "iot-ms-app")
    env.random_deploy("iot-ms-app")

    # 测试 1: 测试random deploy
    print("Test 1: 随机部署microservice到节点上")
    is_deployed = environment.random_deploy()

    print("\n")

    # 测试 2: 计算并打印每个微服务的带宽使用量
    # print("Test 2: 计算并打印每个微服务的带宽使用量")
    # environment.calculate_bandwidth()
    # print("\n")

    # # 测试 3: 打印微服务的邻接列表
    # print("Test 3: 打印微服务的邻接列表")
    # environment.print_adjacency_list()
    # print("\n")