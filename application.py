import json
import random
import parser
from typing import Dict, List
from calls import Call
from microservice import Microservice
from endpoint import Endpoint
class Application:
    def __init__(self, microservices_config_path: str, calls_config_path: str, ms_name: str):
        self.ms_id = 1
        self.instances: Dict[str, Microservice] = {}
        self.bandwidth_adj_list = {} # 存储不同instance之间所产生的带宽。
        self.ms_name = ms_name
        self.traffic_started = False
        self.replica_sets = {} # key为ms_name，value为instance_name的列表
        self.endpoints: Endpoint = {} # 存储所有的endpoints

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
        self._initialize_microservices()
        self._init_endpoints()


    def _load_json(self, path):
        """从文件加载 JSON 配置"""
        with open(path, 'r') as json_file:
            return json.load(json_file)

    def _initialize_microservices(self):
        """初始化所有微服务实例及其依赖关系"""
        for ms_name, ms_data in self.microservices_config.items():
            num_replicas = random.randint(1, ms_data["replica"])
            for i in range(num_replicas):
                instance_name = f"{ms_name}-instance-{i+1}"
                # 将副本集存储在key为ms_name所对应的列表当中，需要handle没有初始化的情况
                if ms_name not in self.replica_sets:
                    self.replica_sets[ms_name] = []
                self.replica_sets[ms_name].append(instance_name)
                microservice_instance = Microservice(
                    id=self.ms_id,
                    name=instance_name,
                    original_name=ms_name,
                    cpu_requests=parser.parse_cpu_requests(ms_data["cpu-requests"]),
                    memory_requests=parser.parse_memory(ms_data["memory-requests"]),
                    num_replicas=num_replicas,
                    type=ms_data["type"] 
                )
                self.instances[instance_name]=microservice_instance
                self.bandwidth_adj_list[microservice_instance.name] = []
                self.ms_id += 1
        self._process_calls()

    def _process_calls(self):
        """处理 calls.json 中的调用路径，生成带宽使用情况的邻接列表并构造 Call 结构体"""
        for call_name, call_data in self.calls_config.items():
            rps = call_data["rps"]
            client = call_data["client"]
            self._dfs_process_seq_call(call_data["call-path"], rps, client)

    def _dfs_process_seq_call(self, seq_calls, rps, ms_name):
        """递归处理调用路径，计算带宽并更新邻接列表，并构造 Call 结构体"""
        ms_replica = self._get_replica_count(ms_name)
        for call_group in seq_calls:
            for next_ms_name, call_data in call_group.items():
                data_size = parser.parse_datasize(call_data.get("data-size", "0M"))
                total_bandwidth = (rps * data_size) / ms_replica
                next_ms_replica = self._get_replica_count(next_ms_name)
                bandwidth_per_replica = total_bandwidth / next_ms_replica
                # 更新邻接列表中的带宽信息
                for microservice in self.instances.values():
                    if microservice.is_instance_of(ms_name):
                        for next_microservice in self.instances.values():
                            if next_microservice.is_instance_of(next_ms_name):
                                existing_entry = next(
                                    (entry for entry in self.bandwidth_adj_list[microservice.name] if entry[0] == next_microservice.name),
                                    None
                                )
                                if existing_entry:
                                    existing_entry[1] += bandwidth_per_replica
                                else:
                                    self.bandwidth_adj_list[microservice.name].append([next_microservice.name, bandwidth_per_replica])
                # 递归处理下一层的调用路径
                if "call-path" in call_data and call_data["call-path"]:
                    self._dfs_process_seq_call(call_data["call-path"], rps, next_ms_name)

    def _init_endpoints(self):
        """初始化所有端点"""
        for endpoint_name, config in self.calls_config.items():
            self.endpoints[endpoint_name] = Endpoint(endpoint_name, config)

    def print_trace(self):
        """打印所有端点的调用路径"""
        for endpoint in self.endpoints.values():
            endpoint.print_trace()

    def calculate_microservice_bandwidth(self):
        """计算每个微服务的总带宽使用量，并考虑是否部署在同一节点上"""

        # 微服务带宽的计算是一个idenpotent操作，因此首先将所有微服务的带宽设置为 0
        for ms in self.instances.values():
            ms.total_bandwidth = 0
            if ms.node_id == -1:
                # 抛出异常，因为微服务没有调度到节点
                raise ValueError(f"Microservice {ms.name} has not been scheduled to a node.")

        for ms in self.instances.values():
            total_bandwidth = 0
            # print(f"开始计算 {ms.name} 的带宽")
            for next_service, bandwidth in self.bandwidth_adj_list[ms.name]:
                next_microservice = self.instances[next_service]
                if ms.node_id != next_microservice.node_id:
                    total_bandwidth += bandwidth
                    old_bandwidth = next_microservice.total_bandwidth
                    next_microservice.total_bandwidth += bandwidth
                    # print(f"更新 {next_microservice.name} 从 {old_bandwidth} 到 {next_microservice.total_bandwidth}")
            ms.total_bandwidth += total_bandwidth
            print(f"最终 {ms.name} 的带宽为: {ms.total_bandwidth}")


    def _get_replica_count(self, ms_name):
        """获取微服务的副本数量"""
        for microservice in self.instances.values():
            if microservice.original_name == ms_name:
                return microservice.num_replicas
        return 1


    def get_next(self, ms_id):
        """返回指定微服务实例的所有依赖"""
        if ms_id not in self.bandwidth_adj_list:
            raise ValueError(f"Microservice {ms_id} not found.")
        return self.bandwidth_adj_list[ms_id]

    def get_replica_set(self, ms_id):
        """返回指定微服务实例的副本集"""
        return [ms.name for ms in self.instances.values() if ms.original_name == ms_id]

    def get_instances(self):
        """返回所有微服务实例的id"""
        return [ms for ms in self.instances.keys()] 

    def get_instance(self, instance_id:str):
        """返回所有微服务实例的id"""
        return self.instances[instance_id]

    def get_bandwidth(self, instance_id: int) -> int:
        if self.deployState != "Deployed":
            raise ValueError("微服务尚未部署")
        return self.get_instance(instance_id).total_bandwidth

    def __repr__(self):
        return f"Microservices({len(self.instances)} instances)"

    def schedule_instance_to_node(self, instance_id: str, node_id: str):
        """将微服务实例调度到指定节点"""
        ms = self.get_instance(instance_id)
        if ms.node_id == -1:
            self.deployedInstanceCnt += 1
        ms.node_id = node_id
        print(f"调度 {ms.name} 到节点 {node_id}")
        self._handle_deploy_state_change()

    def unschedule_instance(self, instance_id: str):
        """将微服务实例从节点上撤销"""
        ms = self.get_instance(instance_id)
        assert(ms.node_id != -1)
        ms.node_id = -1
        self.deployedInstanceCnt -= 1
        self._handle_deploy_state_change()

    def _handle_deploy_state_change(self):
        if self.deployedInstanceCnt == len(self.instances):
            self.deployState = "Deployed"
        elif self.deployedInstanceCnt > 0:
            self.deployState = "Ongoing"
        else:
            self.deployState = "Undeployed"
    def get_endpoint(self, endpoint_id: str):
        return self.endpoints[endpoint_id]