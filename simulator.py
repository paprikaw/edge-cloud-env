import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import random
from scipy.interpolate import interp1d
from pod import Pod
from application import Application
from node import Node
import myparser
from typing import Dict, cast, List
from call import Call
import logging
from endpoint import Endpoint
logger = logging.getLogger(__name__)

class MicroserviceSimulator:
    def __init__(self, service_config_path = None, call_config_path = None, node_config_path = None, cloud_latency = None):
        if service_config_path is None or call_config_path is None or node_config_path is None:
            raise Exception("Please provide the config path")

        self.node_incre_id = 1 # 节点ID
        self.nodes: Dict[int, Node] = {} # 存储所有的节点
        self.latency_between_layer: Dict[str, Dict[str, float]] = {} # 存储不同层级之间的延迟
        self.node_layer_map: Dict[str, List[int]] = {} # 存储不同层级的节点
        self.cpu_types: List[str] = ["A", "B", "C", "D"]
        self.layers: List[str] = ["cloud", "edge", "client"]    
        self._init_nodes(node_config_path)  # 初始化节点信息
        self.cloud_latency = cloud_latency


        self.apps: Dict[str, Application] = {} # Dict[app_name, Application]
        self.load_app(service_config_path, call_config_path, "iot-ms-app")

    def _load_profiling_data(self, path):
        with open(path, 'r') as json_file:
            return json.load(json_file)
    def set_cloud_latency(self, latency):
        self.cloud_latency = latency
    def _init_layer_latency(self, latency_config):
        """初始化节点之间的延迟信息"""
        for layer, latencies in latency_config.items():
            for target_layer, latency_range in latencies.items():
                assert(isinstance(latency_range, list))
                assert(len(latency_range) == 2)
                latency = random.uniform(myparser.parse_time(latency_range[0]), myparser.parse_time(latency_range[1]))
                self.latency_between_layer.setdefault(layer, {})[target_layer] = latency

    def get_latency_between_layers(self, layer: str, target_layer: str) -> float:
        """获取两个层级之间的延迟"""
        if layer == target_layer:
            return 1.0
        if target_layer == "cloud":
            return self.cloud_latency
        if layer == "cloud":
            return self.cloud_latency
        return 1.0

    def _init_nodes(self, node_config_path):
        """从配置文件中初始化节点资源"""
        node_list = []
        nodes_config = self._load_profiling_data(node_config_path) # 加载节点配置
        for layer, node_types in nodes_config["cluster_setup"].items():
            for node_type, config in node_types.items():
                node_count = config["count"]
                for i in range(node_count):
                    cpu_availability = myparser.parse_cpu_requests(random.choice(config["cpu_availability"]))
                    memory_availability = myparser.parse_memory(random.choice(config["memory_availability"]))
                    bandwidth = myparser.parse_bandwidth(nodes_config["node_type"][node_type]["bandwidth"])
                    bandwidth_usage = myparser.parse_percentage(random.choice(config["bandwidth_utilization"])) * float(bandwidth)
                    cpu_type = nodes_config["node_type"][node_type]["cpu_type"]
                    node_name = f"{node_type}-{i+1}"
                    node_id = self.node_incre_id
                    self.nodes[node_id] = Node(
                        node_id=node_id,
                        node_name=node_name,
                        node_type=node_type,
                        cpu_type=cpu_type,
                        cpu_availability=cpu_availability,
                        memory_availability=memory_availability,
                        bandwidth_usage=bandwidth_usage,
                        bandwidth=bandwidth,
                        layer=layer
                    )
                    node_list.append(node_name)
                    self.node_layer_map.setdefault(layer, []).append(node_id)
                    logger.debug(f"Initialized node: {node_name} with CPU Type: {cpu_type}, CPU: {cpu_availability}, Memory: {memory_availability}, Bandwidth: {bandwidth}, Bandwidth Usage: {bandwidth_usage}")
                    self.node_incre_id += 1
        # 将node_name列表输出为JSON文件
        with open('node_name_order.json', 'w', encoding='utf-8') as f:
            json.dump(node_list, f, ensure_ascii=False, indent=4)
        self._init_layer_latency(nodes_config["latency"])

    def _find_available_nodes(self, app_name: str, pod_id: int) -> List[int]:
        """Find available nodes for a pod"""
        pod = self.get_app(app_name).get_pod(pod_id)
        if pod.layer == "all":
            nodes = self.node_layer_map["cloud"] + self.node_layer_map["edge"]
        else:
            nodes = self.node_layer_map[pod.layer]
        available_nodes = [node_id for node_id in nodes if self.get_node(node_id).check_resource(pod.cpu_requests, pod.memory_requests)]
        random.shuffle(available_nodes)
        return available_nodes

    def get_node(self, node_id: int)->Node:
        """获取节点"""
        return self.nodes[node_id]
    def get_node_ids(self) -> List[int]:
        return list(self.nodes.keys())
    def get_nodes(self) -> List[Node]:
        return list(self.nodes.values())

    def load_app(self, ms_config_path: str, calls_config_path: str, app_name: str):
        """加载微服务应用配置"""
        if app_name in self.apps:
            raise Exception(f"Microservice {app_name} already exists")
        self.apps[app_name] = Application(ms_config_path, calls_config_path, app_name)

    def deploy_app(self, app_name: str) -> bool:
        """Deploy pods of an application to the cluster"""
        app = self.apps[app_name]
        assert(app.deployState == "Undeployed")
         # Using a log to track the deployment
         # If in between the deployment, we find 
         # that there are no available nodes, we 
         # rollback the deployment
        log = []
        for pod_id in app.get_pods():
            nodes = self._find_available_nodes(app_name, pod_id)
            pod = app.get_pod(pod_id)
            if len(nodes) == 0:
                # Rollback the deployment if no available nodes
                for pod_id, node_id in log:
                    pod = app.get_pod(pod_id)
                    node = self.get_node(node_id)
                    node._release_resource(pod.cpu_requests, pod.memory_requests)
                return False

            idx = random.randint(0, len(nodes) - 1)
            log.append((pod_id, nodes[idx]))
            node = self.nodes[nodes[idx]]
            node.claim_resource(pod.cpu_requests, pod.memory_requests)
        
        for pod_id, node_id in log:
            app.schedule_pod_to_node(pod_id, node_id)
        
        assert(app.deployState == "Deployed")
        self.start_traffic(app_name)
        return True

    def undeploy_pod(self, app_name: str, pod_id: int):
        """将单个pod从集群中撤销"""
        app = self.get_app(app_name)
        assert app.traffic_started == False
        pod = app.get_pod(pod_id)
        node = self.get_node(pod.node_id)
        app._unschedule_pod(pod_id)
        node._release_resource(pod.cpu_requests, pod.memory_requests)
        return

    def reset_ms(self, ms_app_name: str):
        """重置微服务实例"""
        ms_app = self.apps[ms_app_name]
        self.stop_traffic(ms_app.name)
        for ms_instance in ms_app.pods.values():
            self.undeploy_pod(ms_app_name, ms_instance.id)
        assert ms_app.deployState == "Undeployed"
        return

    def deploy_pod(self, ms_app_name: str, pod_id: int, node_id: int):
        """将单个微服务部署到集群中"""
        ms_app = self.apps[ms_app_name]
        assert(ms_app.deployState == "Ongoing")

        ms = ms_app.get_pod(pod_id)
        node = self.nodes[node_id]
        if not node.check_resource(ms.cpu_requests, ms.memory_requests):
            raise Exception(f"Insufficient resources on node {node_id}")

        ms_app.schedule_pod_to_node(pod_id, node_id)
        assert(ms.node_id != -1)

        node.claim_resource(ms.cpu_requests, ms.memory_requests)
        assert(ms_app.deployState == "Deployed")
        return

    def start_traffic(self, ms_app_name: str):
        """开始模拟流量，主要是计算对应的带宽压力"""
        ms_app = self.apps[ms_app_name]
        assert(ms_app.traffic_started == False)
        assert(ms_app.deployState == "Deployed")
        ms_app.calculate_microservice_bandwidth()
        for ms_instance in ms_app.pods.values():
            # Add total bandwidth of a instance to node bandwidth availability
            self.nodes[ms_instance.node_id].bandwidth_usage += ms_instance.total_bandwidth
        ms_app.traffic_started = True
        return

    def stop_traffic(self, ms_app_name: str):
        """停止模拟流量"""
        ms_app = self.apps[ms_app_name]
        assert(ms_app.deployState == "Deployed")
        assert(ms_app.traffic_started == True)
        for ms_instance in ms_app.pods.values():
            # Subtract total bandwidth of a instance to node bandwidth availability
            self.nodes[ms_instance.node_id].bandwidth_usage -= ms_instance.total_bandwidth
        ms_app.traffic_started = False
        return

    def migrate_pods(self, ms_app_name: str, pod_id: int, node_id: int):
        """
        将微服务调度到迁移到指定节点，这个接口给我们的sheduler使用
        在当前的语境下，只能schedule一个已经部署的微服务
        只能在traffic已经停止的情况下调用
        """
        self.stop_traffic(ms_app_name)
        self.undeploy_pod(ms_app_name, pod_id)
        self.deploy_pod(ms_app_name, pod_id, node_id)
        self.start_traffic(ms_app_name)

    #----- 关于计算latency的逻辑
    def predict_bandwidth(self, ms_app_id: str, instance_id: str) -> int:
        """预测微服务实例的带宽, 这个function十分重要，是我们的bandwidth预测模型"""
        ms_app = self.apps[ms_app_id]
        pod = ms_app.get_pod(instance_id)
        node = self.nodes[pod.node_id]

        bandwidth_usage_without_instance = node.bandwidth_usage - pod.total_bandwidth
        if bandwidth_usage_without_instance / node.bandwidth < 0.9:
            return node.bandwidth - bandwidth_usage_without_instance
        else:
            return node.bandwidth / 10

    #--- Calculate network latency between instances
    def _latency_between_nodes(self, node_id1: str, node_id2: str):
        """计算两个节点之间的延迟"""
        node1 = self.nodes[node_id1]
        node2 = self.nodes[node_id2]
        return self.get_latency_between_layers(node1.layer, node2.layer)

    def _bandwidth_latency_between_instances(self, ms_app_id: str, instance_id1: str, instance_id2: str, data_size: int):
        bandwidth_between_instance = min(self.predict_bandwidth(ms_app_id, instance_id1), self.predict_bandwidth(ms_app_id, instance_id2))
        return data_size / bandwidth_between_instance

    def network_latency_between_instances(self, ms_app_id: str, instance_id1: str, instance_id2: str, data_size: int):
        """Calculate network latency between two instances
        Network latency = bandwidth latency + propogation latency between two nodes 
        """
        ms_app = self.apps[ms_app_id]
        instance_1, instance_2 = ms_app.get_pod(instance_id1), ms_app.get_pod(instance_id2)
        bandwidth_latency = self._bandwidth_latency_between_instances(ms_app_id, instance_id1, instance_id2, data_size)
        assert(bandwidth_latency < 1)
        return bandwidth_latency + self._latency_between_nodes(instance_1.node_id, instance_2.node_id)

    def _get_pod_node(self, ms_app_id: str, pod_id: int) -> Node:
        """获取微服务实例所在的节点"""
        node_id = self.apps[ms_app_id].get_pod(pod_id).node_id
        return self.nodes[node_id]

    def end_to_end_latency(self, ms_app_id: str, endpoint_id: str) -> float:
        """Calculate end-to-end latency of an endpoint"""
        endpoint = self.apps[ms_app_id].get_endpoint(endpoint_id)
        app = self.apps[ms_app_id]
        def _calculate_call_latency(call: Call) -> float:
            """Helper function to recursively calculate latency for each call"""
            # 获取当前 Call 的执行时间            execution_time = call.execution_time.get(cpu_type, 0)

            # 计算当前call的平均执行
            replica_set = app.get_replica_set(call.name)
            avg_execution_latency = 0
            for pod_id in replica_set:
                cpu_type = self._get_pod_node(ms_app_id, pod_id).cpu_type
                execution_time = call.execution_time.get(str(cpu_type), 0)
                if execution_time == 0:
                    assert call.is_client == True 
                avg_execution_latency += execution_time 
            avg_execution_latency /= len(replica_set)

            # 计算并行调用中的平均延迟
            max_parallel_latency = 0
            for call_group in call.call_groups:
                latency = 0
                for next_call in call_group:
                    # 计算当前 Call 和下一个 Call 之间的平均网络延迟
                    avg_network_latency = self._network_latency_between_replica_set(next_call.data_size, ms_app_id, call.name, next_call.name)
                    # 计算下一跳的延迟，包括网络延迟和执行时间
                    next_call_latency = _calculate_call_latency(next_call)
                    
                    latency += (next_call_latency + avg_network_latency)
                max_parallel_latency = max(latency, max_parallel_latency)
            
            return avg_execution_latency + max_parallel_latency
        # 开始递归计算根 Call 对象的延迟
        total_latency = _calculate_call_latency(endpoint.call_groups)
        return total_latency

    def _network_latency_between_replica_set(self, data_size: int, ms_app_id: str, service_a_name: str, service_b_name: str)-> int:
        """计算两个replica set之间的延迟"""
        replica_set1 = self.apps[ms_app_id].get_replica_set(service_a_name)
        replica_set2 = self.apps[ms_app_id].get_replica_set(service_b_name)
        total_latency = 0
        for instance_id1 in replica_set1:
            for instance_id2 in replica_set2:
                total_latency += self.network_latency_between_instances(ms_app_id, instance_id1, instance_id2, data_size)
        return total_latency / (len(replica_set1) * len(replica_set2))

    def get_app(self, app_name: str) -> Application:
        """获取微服务应用"""
        return self.apps[app_name]
    
    
    def check_node_deployable(self, ms_app_id: str, pod_id: int, node_id: int) -> bool:
        """Check if a node is deployable"""
        ms_app = self.apps[ms_app_id]
        instance = ms_app.get_pod(pod_id)
        node = self.nodes[node_id]
        return node.check_resource(instance.cpu_requests, instance.memory_requests)

    def get_all_pods(self) -> List[int]:
        all_service_instances = []
        for app in self.apps.values():
            service_instances = [instance.id for instance in app.pods.values() if instance.type == "service" or instance.type == "persistent"]
            all_service_instances.extend(service_instances)
        return all_service_instances

    def get_cpu_types(self) -> List[str]:
        return self.cpu_types

    def get_node_layers(self) -> List[str]:
        return self.layers
    def output_simulator_status_to_file(self, filename: str):
        """将当前的模拟器状态输出到文件中"""
        # 首先构造当前node的信息
        node_info = {}
        for node_id, node in self.nodes.items():
            node_info[node_id] = {
                "node_name": node.node_name,
                "node_type": node.node_type,
                "cpu_availability": node.cpu_availability,
                "memory_availability": node.memory_availability,
                "bandwidth": node.bandwidth,
                "bandwidth_usage": node.bandwidth_usage,
                "layer": node.layer
            }
        # 开始构造microservice的信息:
        ms_info = {}
        for ms_name, app in self.apps.items():
            ms_info[ms_name] = {
                "pods": {}
            }
            for pod_id, pod in app.pods.items():
                ms_info[ms_name]["pods"][pod.name] = {
                    "node_id": pod.node_id,
                    "node_name": "no_node" if pod.node_id == -1 else self.nodes[pod.node_id].node_name,
                    "cpu_requests": pod.cpu_requests,
                    "memory_requests": pod.memory_requests,
                    "total_bandwidth": pod.total_bandwidth
                }
        with open(filename, 'w') as file:
            json.dump({
                "nodes": node_info,
                "microservices": ms_info
            }, file)

if __name__ == "__main__":
    # 创建 MicroserviceEnvironment 实例
    env = MicroserviceSimulator()
    logging.basicConfig(level=logging.INFO)
    # 测试 1: 初始化节点
    logger.info("Test 1: 初始化节点")
    logger.info(f"Total nodes initialized: {len(env.nodes)}")
    for node_id, node in env.nodes.items():
        logger.info(f"Node ID: {node_id}, CPU: {node.cpu_availability}, Memory: {node.memory_availability}, Bandwidth Usage: {node.bandwidth_usage}")
    
    # 测试 2: 加载微服务应用
    logger.info("\nTest 2: 加载微服务应用")
    ms_name = "iot-ms-app"
    logger.info(f"Microservice {ms_name} loaded with {len(env.apps[ms_name].get_all_pods())} instances")

    # 测试 3: 部署微服务
    logger.info("\nTest 3: 部署微服务")
    is_deployed = env.deploy_app(ms_name)
    if is_deployed:
        logger.info(f"Microservice {ms_name} successfully deployed")
    else:
        logger.error(f"Microservice {ms_name} deployment failed due to insufficient resources")
    # 测试 4: 开始流量模拟
    logger.info("\nTest 4: 开始流量模拟")
    for node_id, node in env.nodes.items():
        logger.info(f"Node ID: {node_id}, Bandwidth Usage: {node.bandwidth_usage}")

    # 测试 5: 迁移微服务实例
    logger.info("\nTest 5: 迁移微服务实例")
    ms_instance_id = env.apps[ms_name].get_all_pods()[0]  # 选择一个实例
    ms = env.apps[ms_name].get_pod(ms_instance_id)
    cur_node_id = ms.node_id
    target_node_id = -1
    
    # 选择另外一个available的节点
    for node_id, node in env.nodes.items():
        if node_id != cur_node_id and node.check_resource(ms.cpu_requests, ms.memory_requests):
            target_node_id = node_id
            break
    if target_node_id == -1:
        logger.error(f"No available node to migrate instance {ms_instance_id}")
    else:    
        env.migrate_pods(ms_name, ms_instance_id, target_node_id)
        logger.info(f"Instance {ms_instance_id} migrated from {cur_node_id} to node {target_node_id}")

    for node_id, node in env.nodes.items():
        logger.info(f"Node ID: {node_id}, Bandwidth Usage: {node.bandwidth_usage}")

    # 测试 6: 打印 Call Path
    logger.info("\nTest 6: 打印 Call Path")
    env.apps[ms_name].print_trace()

    # 测试 7：测试所有node之间的latency
    logger.info("\nTest 7: 测试所有node之间的latency")
    for node_id1 in env.nodes:
        for node_id2 in env.nodes:
            logger.info(f"Latency between {node_id1} and {node_id2}: {env._latency_between_nodes(node_id1, node_id2)}")

    # 测试 8: 计算端到端延迟
    logger.info("\nTest 8: 计算端到端延迟")
    endpoint_id = "data-persistent"  # 假设这个 endpoint ID 存在于 calls.json 中
    end_to_end_latency = env.end_to_end_latency(ms_name, endpoint_id)
    logger.info(f"End-to-End Latency for {endpoint_id}: {end_to_end_latency} ms")

    logger.info("\nTest 9: 验证get_endpoint api")
    logger.info(env.get_endpoints())

    logger.info("\nTest 10: 验证get_all_instances api")
    logger.info(env.get_all_pods())


    # 测试：check_node_deployable
    logger.info("\nTest 11: check_node_deployable")
    for node_id, node in env.nodes.items():
        logger.info(f"Node ID: {node_id}, CPU: {node.cpu_availability}, Memory: {node.memory_availability}, Bandwidth Usage: {node.bandwidth_usage}")
    instanecs = env.apps[ms_name].get_all_pods()
    ms_id = random.choice(instanecs)  # 选择一个实例
    for target_node_id in env.get_node_ids():
        logger.info(f"Check if instance {ms_id} can be deployed on node {target_node_id}: {env.check_node_deployable(ms_name, ms_id, target_node_id)}")