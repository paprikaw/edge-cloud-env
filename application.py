import json
import random
import parser
from typing import Dict, List
from call import Call
from pod import Pod
from endpoint import Endpoint
import logging
from service import Service
class Application:
    def __init__(self, microservices_config_path: str, calls_config_path: str, app_name: str):
        self.incre_id = 0
        self.pods: Dict[int, Pod] = {}
        self.bandwidth_adj_list = {} # 存储不同instance之间所产生的带宽。
        self.name = app_name
        self.traffic_started = False
        # self.replica_sets = {} # key为ms_name，value为instance_name的列表
        self.endpoints: Endpoint = {} # 存储所有的endpoints
        self.services: Dict[str, Service] = {} # 存储所有的服务
        # ms app有三种状态:
        # Undeployed: 所有ms都没有被部署
        # Ongoing: 部分ms被部署
        # Deployed: 所有ms都被部署
        self.deployedInstanceCnt = 0
        self.deployState = "Undeployed" 

        # 加载配置文件
        self.microservices_config = self._load_json(microservices_config_path)
        self.calls_config = self._load_json(calls_config_path)

        # 初始化microserivces和endpoints
        self._initialize_pods()
        self._init_endpoints()


    def _load_json(self, path):
        """从文件加载 JSON 配置"""
        with open(path, 'r') as json_file:
            return json.load(json_file)
        
    def _initialize_pods(self):
        """初始化所有微服务实例及其依赖关系"""
        for service_name, service_meta_data in self.microservices_config.items():
            max_replicas = service_meta_data["max-replica"]
            num_replicas = random.randint(1, max_replicas)
            new_service = Service(service_name, max_replicas, num_replicas)
            for i in range(num_replicas):
                pod_name = f"{service_name}-{i+1}"
                # 将副本集存储在key为ms_name所对应的列表当中，需要handle没有初始化的情况
                pod = Pod(
                    id=self.incre_id,
                    name=pod_name,
                    service_name=service_name,
                    cpu_requests=parser.parse_cpu_requests(service_meta_data["cpu-requests"]),
                    memory_requests=parser.parse_memory(service_meta_data["memory-requests"]),
                    num_replicas=num_replicas,
                    type=service_meta_data["type"]
                )
                self.pods[self.incre_id]=pod
                self.bandwidth_adj_list[pod.id] = []
                new_service.add_pod(pod.id)
                self.incre_id += 1
            self.services[service_name] = new_service
        self._parse_calls()

    def _parse_calls(self):
        """处理 calls.json 中的调用路径，生成带宽使用情况的邻接列表并构造 Call 结构体"""
        for call_name, call_data in self.calls_config.items():
            rps = call_data["rps"]
            client = call_data["client"]
            self._dfs_parse_calls(call_data["call-path"], rps, client)

    def _dfs_parse_calls(self, seq_calls, rps, pod_id):
        """递归处理调用路径，计算带宽并更新邻接列表"""
        ms_replica = self._get_replica_count(pod_id)
        for call_group in seq_calls:
            for nxt_svc_name, call_data in call_group.items():
                data_size = parser.parse_datasize(call_data.get("data-size", "0M"))
                total_bandwidth = (rps * data_size) / ms_replica
                nxt_svc_replica_cnt = self._get_replica_count(nxt_svc_name)
                bandwidth_per_replica = total_bandwidth / nxt_svc_replica_cnt
                # 更新邻接列表中的带宽信息
                for microservice in self.pods.values():
                    if microservice.is_instance_of(pod_id):
                        for next_microservice in self.pods.values():
                            if next_microservice.is_instance_of(nxt_svc_name):
                                existing_entry = next(
                                    (entry for entry in self.bandwidth_adj_list[microservice.id] if entry[0] == next_microservice.id),
                                    None
                                )
                                if existing_entry:
                                    existing_entry[1] += bandwidth_per_replica
                                else:
                                    self.bandwidth_adj_list[microservice.id].append([next_microservice.id, bandwidth_per_replica])
                # 递归处理下一层的调用路径
                if "call-path" in call_data and call_data["call-path"]:
                    self._dfs_parse_calls(call_data["call-path"], rps, nxt_svc_name)

    def _init_endpoints(self):
        """初始化所有端点"""
        for endpoint_name, config in self.calls_config.items():
            self.endpoints[endpoint_name] = Endpoint(endpoint_name, config)

    def print_trace(self):
        """打印所有端点的调用路径"""
        pass
        # for endpoint in self.endpoints.values():
            
    # def _print_call_trace(self, call: Call, level: int):
    #    """辅助递归函数，用于打印 Call 对象结构"""
    #    indent = "  " * level
    #    logging.info(f"{indent}Call: {call.name}, Data Size: {call.data_size}MB")
    #    # 找到当前的服务所对应的所有pods，并且打印这些pods的信息
    #    for parallel_calls in call.call_groups:
    #        for next_call in parallel_calls:
    #            self._print_call_trace(next_call, level + 1)

    def calculate_microservice_bandwidth(self):
        """计算每个微服务的总带宽使用量，并考虑是否部署在同一节点上"""

        # 微服务带宽的计算是一个idenpotent操作，因此首先将所有微服务的带宽设置为 0
        for ms in self.pods.values():
            ms.total_bandwidth = 0
            if ms.node_id == -1:
                # 抛出异常，因为微服务没有调度到节点
                raise ValueError(f"Microservice {ms.name} has not been scheduled to a node.")

        for ms in self.pods.values():
            total_bandwidth = 0
            # print(f"开始计算 {ms.name} 的带宽")
            for next_service_id, bandwidth in self.bandwidth_adj_list[ms.id]:
                next_microservice = self.pods[next_service_id]
                if ms.node_id != next_microservice.node_id:
                    total_bandwidth += bandwidth
                    # old_bandwidth = next_microservice.total_bandwidth
                    next_microservice.total_bandwidth += bandwidth
                    # print(f"更新 {next_microservice.name} 从 {old_bandwidth} 到 {next_microservice.total_bandwidth}")
            ms.total_bandwidth += total_bandwidth
            logging.debug(f"最终 {ms.name} 的带宽为: {ms.total_bandwidth}")


    def _get_replica_count(self, svc_name):
        """获取微服务的副本数量"""
        return self.services[svc_name].sched_replica_cnt


    def get_next(self, ms_id):
        """返回指定微服务实例的所有依赖"""
        if ms_id not in self.bandwidth_adj_list:
            raise ValueError(f"Microservice {ms_id} not found.")
        return self.bandwidth_adj_list[ms_id]

    def get_replica_set(self, service_name):
        """返回指定微服务实例的副本集"""
        return [pod_id for pod_id in self.services[service_name].get_pods()]

    def get_pods(self)->List[int]:
        """返回所有微服务实例的id"""
        return [ms for ms in self.pods.keys()] 

    def get_pod(self, pod_id:int)->Pod:
        """返回所有微服务实例的id"""
        return self.pods[pod_id]


    def get_bandwidth(self, instance_id: int) -> int:
        if self.deployState != "Deployed":
            raise ValueError("微服务尚未部署")
        return self.get_pod(instance_id).total_bandwidth

    def __repr__(self):
        return f"Microservices({len(self.pods)} instances)"

    def schedule_pod_to_node(self, pod_id: str, node_id: int):
        """将微服务实例调度到指定节点"""
        pod = self.get_pod(pod_id)
        if pod.node_id == -1:
            self.deployedInstanceCnt += 1
        pod.node_id = node_id
        logging.debug(f"调度 {pod.name} 到节点 {node_id}")
        self._handle_deploy_state_change()

    def _unschedule_pod(self, pod_name: str):
        """将微服务实例从节点上撤销"""
        pod = self.get_pod(pod_name)
        assert(pod.node_id != -1)
        pod.node_id = -1
        self.deployedInstanceCnt -= 1
        self._handle_deploy_state_change()

    def _handle_deploy_state_change(self):
        if self.deployedInstanceCnt == len(self.pods):
            self.deployState = "Deployed"
        elif self.deployedInstanceCnt > 0:
            self.deployState = "Ongoing"
        else:
            self.deployState = "Undeployed"

    def get_endpoint(self, endpoint_name: str)->Endpoint:
        return self.endpoints[endpoint_name]
    def get_endpoints(self)->List[Endpoint]:
        return self.endpoints
    def get_endpoint_qos(self, endpoint_name: str):
        return self.endpoints[endpoint_name].qos