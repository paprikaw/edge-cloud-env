import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import random
from scipy.interpolate import interp1d
from microservice import Microservice
from application import Application
from node import Node
import parser
from typing import Dict, cast, List
from calls import Call

class MicroserviceSimulator:
    def __init__(self, node_path):
        self.node_id = 1
        # self.profiling_data = self._load_profiling_data(profiling_path)
        self.nodes: Dict[str, Node] = {}
        self.latency_between_layer: Dict[str, Dict[str, float]] = {}
        self.node_layer_map: Dict[str, List[str]] = {}
        nodes_config = self._load_profiling_data(node_path)
        self._init_nodes(nodes_config)  # 初始化节点信息
        self._init_layer_latency(nodes_config["latency"]) # 初始化节点之间的延迟信息
        self.ms_apps: Dict[str, Application] = {}
        
    def _load_profiling_data(self, path):
        with open(path, 'r') as json_file:
            return json.load(json_file)

    def _init_layer_latency(self, latency_config):
        """初始化节点之间的延迟信息"""
        for layer, latencies in latency_config.items():
            for target_layer, latency in latencies.items():
                self.latency_between_layer.setdefault(layer, {})[target_layer] = parser.parse_time(latency)

    def _init_nodes(self, nodes_config):
        """从配置文件中初始化节点资源"""
        for layer, node_types in nodes_config["cluster_setup"].items():
            for node_type, config in node_types.items():
                node_count = random.randint(config["count_range"][0], config["count_range"][1])
                for _ in range(node_count):
                    cpu_availability = parser.parse_cpu_requests(random.choice(config["cpu_availability"]))
                    memory_availability = parser.parse_memory(random.choice(config["memory_availability"]))
                    bandwidth = parser.parse_bandwidth(nodes_config["node_type"][node_type]["bandwidth"])
                    bandwidth_usage = parser.parse_percentage(random.choice(config["bandwidth_utilization"])) * float(bandwidth)
                    cpu_type = nodes_config["node_type"][node_type]["cpu_type"]

                    node_name = f"{node_type}_{self.node_id}"
                    self.nodes[node_name] = Node(
                        node_id=self.node_id,
                        node_name=node_name,
                        node_type=node_type,
                        cpu_type=cpu_type,
                        cpu_availability=cpu_availability,
                        memory_availability=memory_availability,
                        bandwidth_usage=bandwidth_usage,
                        bandwidth=bandwidth,
                        layer=layer
                    )
                    self.node_layer_map.setdefault(layer, []).append(node_name)
                    print(f"Initialized node: {node_name} with CPU Type: {cpu_type}, CPU: {cpu_availability}, Memory: {memory_availability}, Bandwidth: {bandwidth}, Bandwidth Usage: {bandwidth_usage}")
                    self.node_id += 1

    def _find_available_nodes(self, ms: Microservice) -> List[str]:
        """查找可用的节点"""
        if ms.type == "service":
            nodes = self.node_layer_map["cloud"] + self.node_layer_map["edge"]
        else:
            nodes = self.node_layer_map["client"]
    
        # 过滤出可用节点
        available_nodes = [node_id for node_id in nodes if self.nodes[node_id].check_resource(ms.cpu_requests, ms.memory_requests)]
        return available_nodes

    def get_node(self, node_id: str)->Node:
        """获取节点"""
        return self.nodes[node_id]

    def load_ms_app(self, ms_config_path: str, calls_config_path: str, ms_app_name):
        """加载微服务应用配置"""
        if ms_app_name in self.ms_apps:
            raise Exception(f"Microservice {ms_app_name} already exists")
        self.ms_apps[ms_app_name] = Application(ms_config_path, calls_config_path, ms_app_name)

    def deploy_ms_app(self, ms_app_name: str) -> bool:
        """将整个微服务部署到集群当中"""
        ms_app = self.ms_apps[ms_app_name]
        assert(ms_app.deployState == "Undeployed")

        commit_log = []
        for ms in ms_app.instances.values():
            nodes = self._find_available_nodes(ms)

            if len(nodes) == 0:
                for ms_id, node_id in commit_log:
                    ms = ms_app.get_instance(ms_id)
                    self.nodes[node_id].release_resource(ms.cpu_requests, ms.memory_requests)
                return False

            idx = random.randint(0, len(nodes) - 1)
            commit_log.append((ms.name, nodes[idx]))
            node = self.nodes[nodes[idx]]
            node.claim_resource(ms.cpu_requests, ms.memory_requests)

        for ms, node_id in commit_log:
            ms_app.schedule_instance_to_node(ms, node_id)

        assert(ms_app.deployState == "Deployed")
        return True

    def undeploy_ms(self, ms_app_name: str, instance_id: str):
        """将单个微服务从集群中撤销"""
        ms_app = self.ms_apps[ms_app_name]
        assert(ms_app.deployState == "Deployed")

        ms = ms_app.get_instance(instance_id)
        node = self.nodes[ms.node_id]
        node.release_resource(ms.cpu_requests, ms.memory_requests)
        ms_app.unschedule_instance(instance_id)
        assert(ms_app.deployState == "Ongoing")
        return

    def reset_ms(self, ms_app_name: str):
        """重置微服务实例"""
        ms_app = self.ms_apps[ms_app_name]
        for ms_instance in ms_app.instances.values():
            self.undeploy_ms(ms_app_name, ms_instance.id)
        ms_app.deployState = "Undeployed"
        return

    def deploy_ms(self, ms_app_name: str, instance_id: str, node_id: str):
        """将单个微服务部署到集群中"""
        ms_app = self.ms_apps[ms_app_name]
        assert(ms_app.deployState == "Ongoing")

        ms = ms_app.get_instance(instance_id)
        node = self.nodes[node_id]
        if not node.check_resource(ms.cpu_requests, ms.memory_requests):
            raise Exception(f"Insufficient resources on node {node_id}")

        ms_app.schedule_instance_to_node(instance_id, node_id)
        assert(ms.node_id != -1)

        node.claim_resource(ms.cpu_requests, ms.memory_requests)
        assert(ms_app.deployState == "Deployed")
        return

    def start_traffic(self, ms_app_name: str):
        """开始模拟流量，主要是计算对应的带宽压力"""
        ms_app = self.ms_apps[ms_app_name]
        assert(ms_app.traffic_started == False)
        assert(ms_app.deployState == "Deployed")
        ms_app.calculate_microservice_bandwidth()
        for ms_instance in ms_app.instances.values():
            # Add total bandwidth of a instance to node bandwidth availability
            self.nodes[ms_instance.node_id].bandwidth_usage += ms_instance.total_bandwidth
        ms_app.traffic_started = True
        return

    def stop_traffic(self, ms_app_name: str):
        """停止模拟流量"""
        ms_app = self.ms_apps[ms_app_name]
        assert(ms_app.deployState == "Deployed")
        assert(ms_app.traffic_started == True)
        for ms_instance in ms_app.instances.values():
            # Subtract total bandwidth of a instance to node bandwidth availability
            self.nodes[ms_instance.node_id].bandwidth_usage -= ms_instance.total_bandwidth
        ms_app.traffic_started = False
        return

    def migrate_microservices(self, ms_app_id: str, instance_id: str, node_id: str):
        """
        将微服务调度到迁移到指定节点，这个接口给我们的sheduler使用
        在当前的语境下，只能schedule一个已经部署的微服务
        只能在traffic已经停止的情况下调用
        """
        env.stop_traffic(ms_name)
        self.undeploy_ms(ms_app_id, instance_id)
        self.deploy_ms(ms_app_id, instance_id, node_id)
        env.start_traffic(ms_name)

    #----- 关于计算latency的逻辑
    def predict_bandwidth(self, ms_app_id: str, instance_id: str) -> int:
        """预测微服务实例的带宽, 这个function十分重要，是我们的bandwidth预测模型"""
        ms_app = self.ms_apps[ms_app_id]
        instance = ms_app.get_instance(instance_id)
        node = self.nodes[instance.node_id]

        bandwidth_usage_without_instance = node.bandwidth_usage - instance.total_bandwidth
        if bandwidth_usage_without_instance / node.bandwidth < 0.9:
            return node.bandwidth - bandwidth_usage_without_instance
        else:
            return node.bandwidth / 10

    #--- Calculate network latency between instances
    def _latency_between_nodes(self, node_id1: str, node_id2: str):
        """计算两个节点之间的延迟"""
        node1 = self.nodes[node_id1]
        node2 = self.nodes[node_id2]
        return self.latency_between_layer[node1.layer][node2.layer]

    def _bandwidth_latency_between_instances(self, ms_app_id: str, instance_id1: str, instance_id2: str, data_size: int):
        bandwidth_between_instance = min(self.predict_bandwidth(ms_app_id, instance_id1), self.predict_bandwidth(ms_app_id, instance_id2))
        return data_size / bandwidth_between_instance

    def network_latency_between_instances(self, ms_app_id: str, instance_id1: str, instance_id2: str, data_size: int):
        """Calculate network latency between two instances
        Network latency = bandwidth latency + propogation latency between two nodes 
        """
        ms_app = self.ms_apps[ms_app_id]
        instance_1, instance_2 = ms_app.get_instance(instance_id1), ms_app.get_instance(instance_id2)

        return self._bandwidth_latency_between_instances(ms_app_id, instance_id1, instance_id2, data_size) + self._latency_between_nodes(instance_1.node_id, instance_2.node_id)

    def _get_instance_node(self, ms_app_id: str, instance_id: str) -> Node:
        """获取微服务实例所在的节点"""
        node_id = self.ms_apps[ms_app_id].get_instance(instance_id).node_id
        return self.nodes[node_id]

    def end_to_end_latency(self, ms_app_id: str, endpoint_id: str) -> float:
        """Calculate end-to-end latency of an endpoint"""
        endpoint = self.ms_apps[ms_app_id].get_endpoint(endpoint_id)
        ms_app = self.ms_apps[ms_app_id]
        def _calculate_call_latency(call: Call) -> float:
            """Helper function to recursively calculate latency for each call"""
            # 获取当前 Call 的执行时间            execution_time = call.execution_time.get(cpu_type, 0)

            # 计算当前call的平均执行时间
            replica_set = ms_app.get_replica_set(call.instance_id)
            avg_execution_latency = 0
            for instance_id in replica_set:
                cpu_type = self._get_instance_node(ms_app_id, instance_id).cpu_type
                execution_time = call.execution_time.get(cpu_type, 0)
                if execution_time == 0:
                    assert call.is_client == True 
                avg_execution_latency += execution_time 
            avg_execution_latency /= len(replica_set)

            # 计算并行调用中的最大延迟
            max_parallel_latency = 0
            for call_group in call.call_groups:
                latency = 0
                for next_call in call_group:
                    # 计算当前 Call 和下一个 Call 之间的平均网络延迟
                    avg_network_latency = self._network_latency_between_replica_set(next_call.data_size, ms_app_id, call.instance_id, next_call.instance_id)
                    # 计算下一跳的延迟，包括网络延迟和执行时间
                    next_call_latency = _calculate_call_latency(next_call)
                    
                    latency += (next_call_latency + avg_network_latency)
                max_parallel_latency = max(latency, max_parallel_latency)
            
            return avg_execution_latency + max_parallel_latency
        # 开始递归计算根 Call 对象的延迟
        total_latency = _calculate_call_latency(endpoint.call_groups)
        return total_latency

    def _network_latency_between_replica_set(self, data_size: int, ms_app_id: str, id1: str, id2: str)-> int:
        """计算两个replica set之间的延迟"""
        replica_set1 = self.ms_apps[ms_app_id].get_replica_set(id1)
        replica_set2 = self.ms_apps[ms_app_id].get_replica_set(id2)
        total_latency = 0
        for instance_id1 in replica_set1:
            for instance_id2 in replica_set2:
                total_latency += self.network_latency_between_instances(ms_app_id, instance_id1, instance_id2, data_size)
        return total_latency / (len(replica_set1) * len(replica_set2))

    def get_endpoints(self)->List[str]:
        """获取所有的endpoints"""
        endpoints = []
        for ms_app in self.ms_apps.values():
            endpoints.extend([name for name in ms_app.endpoints])
        return endpoints

    def check_node_deployable(self, ms_app_id: str, instance_id: str, node_id: str) -> bool:
        """Check if a node is deployable"""
        ms_app = self.ms_apps[ms_app_id]
        instance = ms_app.get_instance(instance_id)
        node = self.nodes[node_id]
        return node.check_resource(instance.cpu_requests, instance.memory_requests)
    def get_all_instances(self) -> List[int]:
        all_service_instances = []
        for app in self.ms_apps.values():
            service_instances = [instance.id for instance in app.instances.values() if instance.type == "service"]
            all_service_instances.extend(service_instances)
        return all_service_instances


if __name__ == "__main__":
    profiling_path = 'default_profile.json'
    microservices_config_path = 'microservices.json'
    calls_config_path = 'call_patterns.json'
    node_config = 'nodes.json'
    
    # 创建 MicroserviceEnvironment 实例
    env = MicroserviceSimulator(node_config)

    # 测试 1: 初始化节点
    print("Test 1: 初始化节点")
    print(f"Total nodes initialized: {len(env.nodes)}")
    for node_id, node in env.nodes.items():
        print(f"Node ID: {node_id}, CPU: {node.cpu_availability}, Memory: {node.memory_availability}, Bandwidth Usage: {node.bandwidth_usage}")
    
    # 测试 2: 加载微服务应用
    print("\nTest 2: 加载微服务应用")
    ms_name = "iot-ms-app"
    env.load_ms_app(microservices_config_path, calls_config_path, ms_name)
    print(f"Microservice {ms_name} loaded with {len(env.ms_apps[ms_name].get_instances())} instances")

    # 测试 3: 部署微服务
    print("\nTest 3: 部署微服务")
    is_deployed = env.deploy_ms_app(ms_name)
    if is_deployed:
        print(f"Microservice {ms_name} successfully deployed")
    else:
        print(f"Microservice {ms_name} deployment failed due to insufficient resources")

    # 测试 4: 开始流量模拟
    print("\nTest 4: 开始流量模拟")
    env.start_traffic(ms_name)
    for node_id, node in env.nodes.items():
        print(f"Node ID: {node_id}, Bandwidth Usage: {node.bandwidth_usage}")

    # 测试 5: 迁移微服务实例
    print("\nTest 5: 迁移微服务实例")
    ms_instance_id = env.ms_apps[ms_name].get_instances()[0]  # 选择一个实例
    ms = env.ms_apps[ms_name].get_instance(ms_instance_id)
    cur_node_id = ms.node_id
    target_node_id = -1
    
    # 选择另外一个available的节点
    for node_id, node in env.nodes.items():
        if node_id != cur_node_id and node.check_resource(ms.cpu_requests, ms.memory_requests):
            target_node_id = node_id
            break
    if target_node_id == -1:
        print(f"No available node to migrate instance {ms_instance_id}")
    else:    
        env.migrate_microservices(ms_name, ms_instance_id, target_node_id)
        print(f"Instance {ms_instance_id} migrated from {cur_node_id} to node {target_node_id}")

    for node_id, node in env.nodes.items():
        print(f"Node ID: {node_id}, Bandwidth Usage: {node.bandwidth_usage}")

    # 测试 6: 打印 Call Path
    print("\nTest 6: 打印 Call Path")
    env.ms_apps[ms_name].print_trace()

    # 测试 7：测试所有node之间的latency
    print("\nTest 7: 测试所有node之间的latency")
    for node_id1 in env.nodes:
        for node_id2 in env.nodes:
            print(f"Latency between {node_id1} and {node_id2}: {env._latency_between_nodes(node_id1, node_id2)}")

    # 测试 8: 计算端到端延迟
    print("\nTest 8: 计算端到端延迟")
    endpoint_id = "data-persistent"  # 假设这个 endpoint ID 存在于 calls.json 中
    end_to_end_latency = env.end_to_end_latency(ms_name, endpoint_id)
    print(f"End-to-End Latency for {endpoint_id}: {end_to_end_latency} ms")

    print("\nTest 9: 验证get_endpoint api")
    print(env.get_endpoints())

    print("\nTest 10: 验证get_all_instances api")
    print(env.get_all_instances())