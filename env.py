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
    def __init__(self, is_training=True):
        super(MicroserviceEnv, self).__init__()
        
        self.current_ms = None  # 当前待调度的微服务实例
        self.app_name = "iot-ms-app"  # 当前微服务应用的名称
        self.is_training = is_training

        # 定义最低延迟和最大奖励
        self.lower_latency_reward = 3
        self.final_reward = 64
        self.episode = 1
        self.node_reset_episode_interval = 1
        self.episode_steps = 0
        self.max_episode_steps = 10

        # 定义状态空间和动作空间

        # 将node_ids和pod_ids打乱
        # random.shuffle(self.node_ids)
        # random.shuffle(self.pod_ids)
        num_nodes = 4
        num_ms = 4
        self.observation_space = spaces.Dict({
            "Node_cpu_availability": spaces.Box(low=0, high=16, shape=(num_nodes,), dtype=np.float32),
            "Node_memory_availability": spaces.Box(low=0, high=16, shape=(num_nodes,), dtype=np.float32),
            "Node_cpu_type": spaces.MultiDiscrete([4] * num_nodes),  # 假设有4种CPU类型，长度为num_nodes的数组
            "Node_bandwidth": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            "Node_bandwidth_usage": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            "Node_layer": spaces.MultiDiscrete([3] * num_nodes),  # 假设有3个层级，长度为num_nodes的数组
            # "Node_id": spaces.MultiDiscrete([num_nodes] * num_nodes), #nodes的id
            # "Pod_id": spaces.MultiDiscrete([num_ms] * num_ms),  # 对于每个微服务实例，表示其ID
            "Pod_node_id": spaces.MultiDiscrete([num_nodes] * num_ms),  # 对于每个微服务实例，表示其所在的节点ID
            "Pod_total_bandwidth": spaces.Box(low=0, high=100, shape=(num_ms,), dtype=np.float32),
            # "Pod_cpu_requests": spaces.Box(low=0, high=4, shape=(num_ms,), dtype=np.float32),
            # "Pod_memory_requests": spaces.Box(low=0, high=4, shape=(num_ms,), dtype=np.float32)
        })
        # 定义动作空间
        self.action_space = spaces.MultiDiscrete([num_nodes, num_ms])
    def reset(self, seed=None, options=None):
        '''重置当前的simulator'''
        self.final_reward = 32
        # 将node_ids和pod_ids打乱
        # random.shuffle(self.node_ids)
        # random.shuffle(self.pod_ids)
                # 读取文件路径和相关配置
        if self.episode % self.node_reset_episode_interval == 0:
            '''重新初始化一个simulator状态'''
            self._init_valid_simulator()
            if not self.is_training:
                self.simulator.output_simulator_status_to_file("status.json")
            self.app = self.simulator.get_app(self.app_name)
            self.node_ids = self.simulator.get_schedulable_nodes()
            self.pod_ids = self.app.get_pods()
            self.endpoints = self.app.get_endpoints()
            self.lowest_latency = self.latency_func()
            self.lower_latency_reward = 3
            self.node_ids = self.simulator.get_schedulable_nodes()
        else:
            self.simulator.reset_ms(self.app_name)
            self.simulator.deploy_ms_app(self.app_name)

        self.episode += 1
        # 构建初始状态
        return self._get_state(), {}
 
    def _get_state(self):
        """根据当前环境状态，构建状态空间"""
        # 获取待调度的微服务实例ID
        # 构建CPU类型和node layer的名字到数字的映射
        cpu_type_map = self.type_discrete(self.simulator.get_cpu_types())
        layer_map = self.type_discrete(self.simulator.get_node_layers())
        nodes: List[Node] = []
        pods: List[Pod] = []
        for id in self.node_ids:
            nodes.append(self.simulator.nodes[id])
        for id in self.pod_ids:
            pods.append(self.app.pods[id])
        # 使用 self.nodeid和self.podid的顺序来生成node和pod的list
        
        # 构建节点的状态
        nodes_state = {
            # "Node_id": np.array(self.node_ids, dtype=np.int32),
            "Node_cpu_availability": np.array([node.cpu_availability for node in nodes], dtype=np.float32),
            "Node_memory_availability": np.array([node.memory_availability for node in nodes], dtype=np.float32),
            "Node_cpu_type": np.array([cpu_type_map[node.cpu_type] for node in nodes], dtype=np.int32),
            "Node_bandwidth": np.array([node.bandwidth for node in nodes], dtype=np.float32),
            "Node_bandwidth_usage": np.array([node.bandwidth_usage for node in nodes], dtype=np.float32),
            "Node_layer": np.array([layer_map[node.layer] for node in nodes], dtype=np.int32)
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

    # def step(self, action:int):
    #     target_node_id = self.nodes[action]
    #     cur_node_id = self.simulator.ms_apps[self.ms_app_name].get_instance(self.ms_index).node_id
    #     if not self.simulator.check_node_deployable(self.ms_app_name, self.ms_index, target_node_id): 
    #         reward = -100
    #     elif cur_node_id == target_node_id:
    #         reward = 0
    #     else:
    #         before_latency = self.latency_func()
    #         self.simulator.migrate_microservices(self.ms_app_name, self.ms_index, target_node_id)
    #         after_latency = self.latency_func()
    #         if after_latency > before_latency:
    #             reward = -10
    #         elif after_latency < self.lowest_latency:
    #             self.max_reward += 10
    #             reward = self.max_reward
    #         else:
    #             reward = 10

    #     if len(self.instance_id_list) == 0:
    #         done = True
    #     else:
    #         done = False

    #     # 获取新的状态
    #     state = self._get_state() if not done else None
    #     return state, reward, done, False, {}
    def step(self, action: int):
        node_id, ms_id = action
        cur_node_id = self.app.get_pod(ms_id).get_node_id()
        cur_ms_name = self.app.get_pod(ms_id).get_name()
        logger.info(f"Scheduling {cur_ms_name} (node {cur_node_id} -> node {node_id})")
        cur_latency = self.latency_func()
        qos_threshold = self.qos_func()
        self.episode_steps += 1
        reward = 0
        done = False
        
        if node_id == 3:
            if not self.is_training:
                self.simulator.output_simulator_status_to_file("status.json")
            # print("no node can be selected")
            done = True
            self.episode_steps = 0
            state = self._get_state() if not done else None
            return state, reward, done, False, {}
        
        # node_id = self.node_ids[node_id]
        # ms_id = self.pod_ids[ms_id]
        if cur_latency <= qos_threshold:
            logger.info(f"Qos Threshold Reached: {cur_latency}")
            if not self.is_training:
                self.simulator.output_simulator_status_to_file("status.json")
            done = True
            self.episode_steps = 0
            state = self._get_state() if not done else None
            return state, reward, done, False, {}
        # elif self.episode_steps >= self.max_episode_steps:
        #     reward = -50
        #     logger.info("Episode steps reached maximum")
        # if self.app.get_pod(ms_id).get_type() == "client" or self.simulator.get_node(node_id).layer == "client" or not self.simulator.check_node_deployable(self.app_name, ms_id, node_id) or cur_node_id == node_id:
        if cur_node_id == node_id:
            reward = -10
            if not self.is_training:
                done = True
            logger.info("Action resulted in a failure: Deploy to the same node")
            # if self.episode > 1000:
        #     reward = -5
        #     done = True
        #     logger.info("Action resulted in a failure: Client node selected")
        elif not self.simulator.check_node_deployable(self.app_name, ms_id, node_id):
            reward = -100
            if not self.is_training:
                done = True
            logger.info("Action resulted in a failure: Node not deployable")
        # elif cur_node_id == real_node_id:
        #     reward = -5
        #     done = True
        #     logger.info("Action resulted in no change: Same node selected")
        else:
            before_latency = cur_latency
            self.simulator.migrate_pods(self.app_name, ms_id, node_id)
            cur_latency = self.latency_func()
            logger.info(f"Latency {before_latency} -> {cur_latency}")
            # if cur_latency > before_latency:
            reward = before_latency - cur_latency
            if reward < 0:
                logger.info("Action resulted in worse latency")
            # elif cur_latency < self.lowest_latency:
            #     reward += self.lower_latency_reward
            #     self.lowest_latency = cur_latency
            #     self.lower_latency_reward += 3
            #     logger.info(f"New lowest latency achieved! Reward: {reward}")
                # print(f"New lowest latency achieved! Latency {self.lowest_latency}")
                # print(f"New lowest latency achieved! Reward: {reward}")
            # else:
        
        if not done and cur_latency <= qos_threshold:
            logger.info(f"QoS threshold reached, final reward: {self.final_reward}")
            reward += self.final_reward
            done = True
        else: 
            self.final_reward //= 2
        
        reward -= self.episode_steps * 3
        # print("lowest_latency: ", self.lowest_latency)
        # print("reward: ", reward)
        # print(f"cur_latency: {cur_latency}, qos_threshold: {qos_threshold}")
        if self.episode_steps >= self.max_episode_steps:
        # if cur_latency <= qos_threshold:
            done = True

        if done:
            logger.info(f"Episode steps: {self.episode_steps}, Final latency: {cur_latency}")
            self.episode_steps = 0
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
            if  node.layer == "client":
                mask.append(False)
                continue
            isFound = False
            for pod_id in self.pod_ids:
                if self.simulator.check_node_deployable(self.app_name, pod_id, node_id):
                    isFound = True
                    break
            if isFound:
                mask.append(True)
            else:
                mask.append(False)
        for pod_id in self.pod_ids:
            pod = self.app.get_pod(pod_id)
            if pod.get_type() == "client": 
                mask.append(False)
            else:
                mask.append(True)
        return mask
    
                # if pod.get_node_id() == node_id:
                #     mask[i, j] = False    for node_index, node_id in enumerate(self.node_ids):
    # def action_masks(self) -> np.ndarray:
    #     """
    #     Returns a mask indicating which actions are valid.
    #     The mask is a boolean array with True indicating valid actions.

    #     The mask is a 2D array where the first dimension corresponds to the nodes
    #     and the second dimension corresponds to the microservices (pods).
    #     """
    #     num_nodes = len(self.node_ids)
    #     num_pods = len(self.pod_ids)

    #     # Initialize a mask with all True values
    #     mask = np.ones((num_nodes, num_pods), dtype=bool)

    #     for node_index, node_id in enumerate(self.node_ids):
    #         node = self.simulator.get_node(node_id)
    #         for pod_index, pod_id in enumerate(self.pod_ids):
    #             pod = self.app.get_pod(pod_id)
    #             # Check if the action is invalid for the given pod and node
    #             if pod.get_type() == "client" or node.layer == "client" or not self.simulator.check_node_deployable (self.app_name, pod_id, node_id):
    #                 mask[node_index, pod_index] = False

    #     return mask
    def _init_simulator(self, simulator: MicroserviceSimulator, ms_name: str) -> bool:
        logger.debug("\n部署微服务")
        is_deployed = simulator.deploy_ms_app(ms_name)
        if is_deployed:
            logger.info(f"Microservice {ms_name} successfully deployed")
        else:
            logger.info(f"Microservice {ms_name} deployment failed due to insufficient resources")
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