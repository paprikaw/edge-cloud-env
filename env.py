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
    def __init__(self, is_training=True, num_nodes=4, num_ms=4):
        super(MicroserviceEnv, self).__init__()
        
        self.current_ms = None  # 当前待调度的微服务实例
        self.app_name = "iot-ms-app"  # 当前微服务应用的名称
        self.is_training = is_training

        # 定义最低延迟和最大奖励
        self.episode = 0
        self.cluster_reset_interval_by_episode = 1
        self.episode_steps = 0
        self.max_episode_steps = 10

        self.observation_space = spaces.Dict({
            "Node_cpu_availability": spaces.Box(low=0, high=16, shape=(num_nodes,), dtype=np.float32),
            "Node_memory_availability": spaces.Box(low=0, high=16, shape=(num_nodes,), dtype=np.float32),
            "Node_bandwidth": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            "Node_bandwidth_usage": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            "Pod_node_id": spaces.MultiDiscrete([num_nodes] * num_ms),  # Current node of each microservice
            "Pod_total_bandwidth": spaces.Box(low=0, high=100, shape=(num_ms,), dtype=np.float32), # How much bandwidth a pod has on a node.
            "Pod_cpu_requests": spaces.Box(low=0, high=4, shape=(num_ms,), dtype=np.float32),
            "Pod_memory_requests": spaces.Box(low=0, high=4, shape=(num_ms,), dtype=np.float32)
        })


        # "Node_cpu_type": spaces.MultiDiscrete([4] * num_nodes),
        # "Node_layer": spaces.MultiDiscrete([3] * num_nodes),
        # "Node_id": spaces.MultiDiscrete([num_nodes] * num_nodes), #nodes的id
        # "Pod_id": spaces.MultiDiscrete([num_ms] * num_ms),  # 对于每个微服务实例，表示其ID
        # 定义动作空间
        # self.action_space = spaces.MultiDiscrete([num_nodes, num_ms])
        self.action_space = spaces.Discrete(num_nodes * num_ms)
    def reset(self, seed=None, options=None):
        '''Reset simulator, this happened during the end of the episode'''
        self.episode_steps = 0
        if self.episode % self.cluster_reset_interval_by_episode == 0:
            '''重新初始化一个simulator状态'''
            self._init_valid_simulator()
            if not self.is_training:
                self.simulator.output_simulator_status_to_file("./logs/test_start.json")
            self.app = self.simulator.get_app(self.app_name)
            self.node_ids = self.simulator.get_node_ids()
            self.pod_ids = self.app.get_pods()
            self.endpoints = self.app.get_endpoints()
            self.lowest_latency = self.latency_func()
            self.lower_latency_reward = 3
            self.node_ids = self.simulator.get_node_ids()
        else:
            self.simulator.reset_ms(self.app_name)
            self.simulator.deploy_app(self.app_name)

        self.episode += 1
        # 构建初始状态
        return self._get_state(), {}
 
    def _get_state(self):
        """根据当前环境状态，构建状态空间"""
        # 获取待调度的微服务实例ID
        # 构建CPU类型和node layer的名字到数字的映射
        nodes: List[Node] = []
        pods: List[Pod] = []
        for id in self.node_ids:
            nodes.append(self.simulator.nodes[id])
        for id in self.pod_ids:
            pods.append(self.app.pods[id])
        
        # 构建节点的状态
        nodes_state = {
            # "Node_id": np.array(self.node_ids, dtype=np.int32),
            "Node_cpu_availability": np.array([node.cpu_availability for node in nodes], dtype=np.float32),
            "Node_memory_availability": np.array([node.memory_availability for node in nodes], dtype=np.float32),
            # "Node_cpu_type": np.array([cpu_type_map[node.cpu_type] for node in nodes], dtype=np.int32),
            "Node_bandwidth": np.array([node.bandwidth for node in nodes], dtype=np.float32),
            "Node_bandwidth_usage": np.array([node.bandwidth_usage for node in nodes], dtype=np.float32),
            # "Node_layer": np.array([layer_map[node.layer] for node in nodes], dtype=np.int32)
        }
        # 构建微服务实例的状态
        ms_state = {
            # "Pod_id": np.array(self.pod_ids, dtype=np.int32),
            "Pod_node_id": np.array([pod.node_id for pod in pods], dtype=np.int32),
            "Pod_total_bandwidth": np.array([pod.total_bandwidth for pod in pods], dtype=np.float32),
            "Pod_cpu_requests": np.array([pod.cpu_requests for pod in pods], dtype=np.float32),
            "Pod_memory_requests": np.array([pod.memory_requests for pod in pods], dtype=np.float32)
        }
        # 返回状态字典
        state = {
            **nodes_state,
            **ms_state
        }
        return state

    def get_action(self, action: int)->tuple[int, int]:
        num_ms = len(self.pod_ids)
        num_nodes = len(self.node_ids)
        node_id = action // num_ms
        ms_id = action % num_ms
        return int(node_id), int(ms_id)
    
    def step(self, action):
        node_id, ms_id = self.get_action(action)
        cur_node_id = self.app.get_pod(ms_id).get_node_id()
        cur_ms_name = self.app.get_pod(ms_id).get_name()
        logger.info(f"Scheduling {cur_ms_name} (node {cur_node_id} -> node {node_id})")
        cur_latency = self.latency_func()
        qos_threshold = self.qos_func()
        self.episode_steps += 1
        reward = 0
        done = False

        # No node can be selected, episode ends  
        if node_id == 3:
            done = True
        # QoS threshold reached, episode ends
        elif cur_latency <= qos_threshold:
            logger.debug(f"Qos Threshold Reached: {cur_latency}")
            done = True
        elif cur_node_id == node_id:
            reward = -5
            done = True
            logger.debug("Deploy to the same node, episode ends")
        elif not self.simulator.check_node_deployable(self.app_name, ms_id, node_id):
            reward = -1000
            if not self.is_training:
                done = True
            logger.debug("Action resulted in a failure: Node not deployable")
        else:
            before_latency = cur_latency
            self.simulator.migrate_pods(self.app_name, ms_id, node_id)
            cur_latency = self.latency_func()
            logger.debug(f"Latency {before_latency} -> {cur_latency}")
            # if cur_latency > before_latency:
            reward = before_latency - cur_latency
            if reward < 0:
                logger.debug("Action resulted in worse latency")
        

        if self.episode_steps >= self.max_episode_steps or cur_latency <= qos_threshold:
            done = True
        if done:
            logger.debug(f"Episode steps: {self.episode_steps}, Final latency: {cur_latency}")
            if not self.is_training:
                self.simulator.output_simulator_status_to_file("./logs/test_end.json")
        else:
            reward -= 5
            
        logger.info("lowest_latency: ", self.lowest_latency)
        logger.info("reward: ", reward)
        logger.info(f"cur_latency: {cur_latency}, qos_threshold: {qos_threshold}")
        state = self._get_state() if not done else None
        return state, reward, done, False, {}

    def latency_func(self) -> float:
        latency = 0
        for endpoint in self.endpoints:
            latency += self.simulator.end_to_end_latency(self.app_name, endpoint)
        return latency / len(self.endpoints)
    
    def qos_func(self)->float:
        qos = 0
        for endpoint in self.endpoints:
            qos += self.app.get_endpoint(endpoint).get_qos()
        return qos / len(self.endpoints)

    def render(self, mode="human"):
        pass
    
    def action_masks(self)->List[bool]:
        """
        Returns an action mask to filter out invalid actions.
        The mask is a boolean array where True indicates a valid action.
        """
        mask = []
        for node_id in self.node_ids:
            node = self.simulator.get_node(node_id)
            for pod_id in self.pod_ids:
                pod = self.app.get_pod(pod_id)
                if pod.get_node_id() == node_id:
                    mask.append(True)
                    continue
                if pod.get_type() == "client" or node.layer == "client" or not self.simulator.check_node_deployable(self.app_name, pod_id, node_id):
                    mask.append(False)
                else:
                    mask.append(True)
        return mask

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
            self.simulator = MicroserviceSimulator()
            if self._init_simulator(self.simulator, self.app_name):
                return
        assert(False)

    def type_discrete(self, type: List[any])-> Dict[any, int]:
        return {t: i for i, t in enumerate(type)}