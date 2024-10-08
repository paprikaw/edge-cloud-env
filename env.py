import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List
from simulator import MicroserviceSimulator
import random
from gymnasium.wrappers import FlattenObservation
import logging
from node import Node
from pod import Pod

logger = logging.getLogger(__name__)
class MicroserviceEnv(gym.Env):
    """
    Env Version 1:
    RL agent migrates microservices in the application
    RL agent only makes decisions after all microservices are deployed
    """
    def __init__(self, is_testing=False, is_eval=False, dynamic_env=True, num_nodes=0, num_pods=0, step_panelty=2, end_panelty=2):
        super(MicroserviceEnv, self).__init__()
        self.step_panelty = step_panelty
        self.end_panelty = end_panelty
        self.microservices_config_path = './config/services.json'
        self.calls_config_path = './config/call_patterns.json'
        self.node_config_path = './config/nodes.json'
        if not dynamic_env:
            self.node_config_path = './config/nodes-simple.json'

        self.current_ms = None  # 当前待调度的微服务实例
        self.app_name = "iot-ms-app"  # 当前微服务应用的名称
        self.is_testing = is_testing
        self.is_eval = is_eval
        # 定义最低延迟和最大奖励
        self.episode = 0
        self.cluster_reset_interval_by_episode = 1
        self.step_cnt = 0
        self.invalid_training_step = 50000
        self.stopped_action = num_nodes * num_pods
        if is_testing:
            self.max_episode_steps = 15
        else:
            self.max_episode_steps = 100
        self.num_nodes = num_nodes
        self.num_pods = num_pods
        self.nodes: List[Node] = []
        self.pods: List[Pod] = []
        self.action_space = spaces.Discrete(num_nodes * num_pods+1)
        self.observation_space = spaces.Dict({
            # "Node_id": spaces.Box(low=0, high=num_nodes, shape=(num_nodes,), dtype=np.int32),
            "Node_cpu_availability": spaces.Box(low=0, high=16, shape=(num_nodes,), dtype=np.float32),
            "Node_memory_availability": spaces.Box(low=0, high=16, shape=(num_nodes,), dtype=np.float32),
            # "Node_bandwidth_usage": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            # "Node_bandwidth": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            # "Node_layer": spaces.Box(low=0, high=2, shape=(num_nodes,), dtype=np.int32),
            # "Node_cpu_type": spaces.Box(low=0, high=3, shape=(num_nodes,), dtype=np.int32),
            "Pod_node_id": spaces.Box(low=0, high=num_nodes, shape=(num_pods,), dtype=np.int32),  # Current node of each microservice
            # "Pod_layer": spaces.Box(low=0, high=4, shape=(num_pods,), dtype=np.int32),
            # "Pod_type": spaces.Box(low=0, high=2, shape=(num_pods,), dtype=np.int32),
            # "Pod_total_bandwidth": spaces.Box(low=0, high=100, shape=(num_pods,), dtype=np.float32), # How much bandwidth a pod has on a node.
            "Pod_cpu_requests": spaces.Box(low=0, high=4, shape=(num_pods,), dtype=np.float32),
            "Pod_memory_requests": spaces.Box(low=0, high=4, shape=(num_pods,), dtype=np.float32)
        })

        # "Node_cpu_type": spaces.MultiDiscrete([4] * num_nodes),
        # "Node_layer": spaces.MultiDiscrete([3] * num_nodes),
        # "Node_id": spaces.MultiDiscrete([num_nodes] * num_nodes), #nodes的id
        # "Pod_id": spaces.MultiDiscrete([num_ms] * num_ms),  # 对于每个微服务实例，表示其ID
        # 定义动作空间
        # self.action_space = spaces.MultiDiscrete([num_nodes, num_ms])
    def reset(self, seed=None, options=None):
        '''Reset simulator, this happened during the end of the episode'''
        self.episode_steps = 0
        self.cloud_latency = 50
        # if self.episode % self.cluster_reset_interval_by_episode == 0:
        '''重新初始化一个simulator状态'''
        self._init_valid_simulator()
        if self.is_testing and self.episode == 0:
            self.simulator.output_simulator_status_to_file("./logs/test_start.json")
        self.app = self.simulator.get_app(self.app_name)
        self.node_ids = self.simulator.get_node_ids()
        # random.shuffle(self.node_ids)
        self.endpoints = self.app.get_endpoints()
        self.lowest_latency = self.latency_func()
        self.lower_latency_reward = 3
        # else:
        #     self.simulator.reset_ms(self.app_name)
        #     self.simulator.deploy_app(self.app_name)
        logger.info("---------------- Episode Resetted ----------------")
        # for endpoint_id in self.app.get_endpoints():
        #     self.app.get_endpoint(endpoint_id).print_trace()
        self.episode += 1
        # 构建初始状态
        return self._get_state(), {}
 
    def _get_state(self):
        """根据当前环境状态，构建状态空间"""
        # 获取待调度的微服务实例ID
        # 构建CPU类型和node layer的名字到数字的映射
        self.nodes = []
        self.pods = []
        for id in self.node_ids:
            self.nodes.append(self.simulator.nodes[id])
        for service in self.app.services.values():
            for pod_id in service.get_pods():
                pod = self.app.get_pod(pod_id)
                self.pods.append(pod)
            for _ in range(service.sched_replica_cnt, service.max_replica_cnt):
                self.pods.append(Pod("dummy", -1, "dummy", 0, 0, 0, "dummy","dummy", 0, False))

        pod_layer_map = {
            "cloud": 0,
            "edge": 1,
            "client": 2,
            "all": 3,
            "dummy": 4
        }
        layer_map = {
            "cloud": 0,
            "edge": 1,
            "client": 2,
        }

        pod_type_map = {
            "persistent": 0,
            "service": 1,
            "dummy": 2
        }
        # 构建节点的状态
        nodes_state = {
            # "Node_id": np.array([node.node_id for node in self.nodes], dtype=np.int32),
            "Node_cpu_availability": np.array([node.cpu_availability for node in self.nodes], dtype=np.float32),
            "Node_memory_availability": np.array([node.memory_availability for node in self.nodes], dtype=np.float32),
            # "Node_cpu_type": np.array([int(node.cpu_type) for node in self.nodes], dtype=np.int32),
            # "Node_bandwidth": np.array([node.bandwidth for node in self.nodes], dtype=np.float32),
            # "Node_bandwidth_usage": np.array([node.bandwidth_usage for node in self.nodes], dtype=np.float32),
            # "Node_layer": np.array([layer_map[node.layer] for node in self.nodes], dtype=np.int32)
        }

        # 构建微服务实例的状态
        ms_state = {
            # "Pod_id": np.array(self.pod_ids, dtype=np.int32),
            "Pod_node_id": np.array([pod.node_id for pod in self.pods], dtype=np.int32),
            # "Pod_layer": np.array([pod_layer_map[pod.layer] for pod in self.pods], dtype=np.int32),
            # "Pod_type": np.array([pod_type_map[pod.type] for pod in self.pods], dtype=np.int32),
            # "Pod_total_bandwidth": np.array([pod.total_bandwidth for pod in self.pods], dtype=np.float32),
            "Pod_cpu_requests": np.array([pod.cpu_requests for pod in self.pods], dtype=np.float32),
            "Pod_memory_requests": np.array([pod.memory_requests for pod in self.pods], dtype=np.float32),
        }
        # 返回状态字典
        state = {
            **nodes_state,
            **ms_state
        }
        # print(state)
        return state

    def get_action(self, action: int)->tuple[Node, Pod]:
        num_pods = self.num_pods
        node_id = action // num_pods
        pod_id = action % num_pods
        return self.nodes[node_id],self.pods[pod_id]

    def get_action_name(self, action: int)->tuple[str, str]:
        num_pods = self.num_pods
        node_id = action // num_pods
        pod_id = action % num_pods
        return self.nodes[node_id].node_name, self.pods[pod_id].get_name()
    
    def step(self, action):
        self.step_cnt += 1
        if action == self.stopped_action:
            if not self.is_eval and not self.is_testing and self.step_cnt < self.invalid_training_step:
                return self._get_state(), -self.end_panelty, False, False, {}
            logger.info("Stop action selected")
            logger.info(f"Episode steps: {self.episode_steps}, Final latency: {self.latency_func()}")
            if self.is_testing:
                self.simulator.output_simulator_status_to_file("./logs/test_end.json")
            return None, -self.end_panelty, True, False, {}
        node, pod = self.get_action(action)
        node_id = node.get_id()
        pod_id = pod.get_id()
        cur_node_id = pod.get_node_id()
        cur_pod_name = pod.get_name()
        cur_latency = self.latency_func()
        self.episode_steps += 1
        reward = 0
        done = False

        if not self.check_valid_action(pod, node):
            reward = -10
            done = True
            logger.info("Action resulted in a failure: Node not deployable")
        else:
            logger.info(f"Scheduling {cur_pod_name}: \n node {self.simulator.get_node(cur_node_id).node_name} -> node {node.node_name}")
            before_latency = cur_latency
            self.simulator.migrate_pods(self.app_name, pod_id, node_id)
            cur_latency = self.latency_func()
            logger.info(f"Latency {before_latency} -> {cur_latency}")
            reward = before_latency - cur_latency
            if reward < 0:
                logger.info("Action resulted in worse latency")
        if self.episode_steps >= self.max_episode_steps:
            done = True
        if done:
            logger.info(f"Episode steps: {self.episode_steps}, Final latency: {cur_latency}")
            if self.is_testing:
                self.simulator.output_simulator_status_to_file("./logs/test_end.json")
        elif self.step_cnt > self.invalid_training_step:
            reward -= self.step_panelty
        state = self._get_state() if not done else None
        return state, reward, done, False, {}

    def latency_func(self) -> float:
        latency = 0
        for endpoint in self.endpoints:
            latency += self.simulator.end_to_end_latency(self.app_name, endpoint)
        return latency / len(self.endpoints)

    def new_latency_func(self) -> float:
        total_contribution = 0
        total_latency = 0
        for endpoint in self.endpoints:
            latency = self.simulator.end_to_end_latency(self.app_name, endpoint)
            contribution = 1 / (1 + latency)  # 使用倒数函数来实现低延迟贡献高，高延迟贡献低
            total_contribution += contribution
            total_latency += latency

        if total_contribution == 0:
            return float('inf')  # 避免除以零的情况
        
        return total_latency / total_contribution  # 返回加权平均延迟
    
    def qos_func(self)->float:
        qos = 0
        for endpoint in self.endpoints:
            qos += self.app.get_endpoint(endpoint).get_qos()
        return qos / len(self.endpoints)
    def render(self, mode="human"):
        pass

    def check_valid_action(self, pod, node)->bool:
        if not pod.is_scheduled or \
            pod.get_node_id() == node.get_id() or \
            pod.get_type() == "persistent" or \
            pod.layer == "client" or \
            node.layer == "client" or \
            not self.simulator.check_node_deployable(self.app_name, pod.id, node.node_id):
            return False
        return True

    def _init_simulator(self, simulator: MicroserviceSimulator, ms_name: str) -> bool:
        logger.debug("\n部署微服务")
        is_deployed = simulator.deploy_app(ms_name)
        if is_deployed:
            logger.debug(f"Microservice {ms_name} successfully deployed")
        else:
            logger.debug(f"Microservice {ms_name} deployment failed due to insufficient resources")
            return False

        logger.debug("\n开始流量模拟")
        for node_id, node in simulator.nodes.items():
            logger.debug(f"Node ID: {node_id}, Bandwidth Usage: {node.bandwidth_usage}")
        return True

    def _init_valid_simulator(self):
        '''
        初始化模拟器，重复尝试10次
        在一些情况下，可能microservice无法在既有资源下完成部署
        '''
        for i in range(100):
            self.simulator = MicroserviceSimulator(self.microservices_config_path, self.calls_config_path, self.node_config_path, self.cloud_latency)
            if self._init_simulator(self.simulator, self.app_name):
                return
        assert(False)

    def type_discrete(self, type: List[any])-> Dict[any, int]:
        return {t: i for i, t in enumerate(type)}