import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List
from simulator import MicroserviceSimulator
import random
from gymnasium.wrappers import FlattenObservation
import logging

logger = logging.getLogger(__name__)
class MicroserviceEnv(gym.Env):
    """
    Env Version 1:
    RL agent migrates microservices in the application
    RL agent only makes decisions after all microservices are deployed
    """
    def __init__(self):
        super(MicroserviceEnv, self).__init__()
        
        self.current_ms = None  # 当前待调度的微服务实例
        self.app_name = "iot-ms-app"  # 当前微服务应用的名称

        # 读取文件路径和相关配置
        self._init_valid_simulator()
        self.app = self.simulator.get_app(self.app_name)
        self.endpoints = self.app.get_endpoints()
        self.lowest_latency = self.latency_func()

        # 定义最低延迟和最大奖励
        self.lower_latency_reward = 3
        self.final_reward = 100
        self.episode = 1
        self.node_reset_episode_interval = 10
        self.episode_steps = 0
        self.max_episode_steps = 500

        # 定义状态空间和动作空间
        self.node_ids = self.simulator.get_schedulable_nodes()
        self.pod_ids = self.app.get_pods()
        # 将node_ids和pod_ids打乱
        random.shuffle(self.node_ids)
        random.shuffle(self.pod_ids)
        num_nodes = len(self.node_ids)
        num_ms = len(self.app.pods)
        self.observation_space = spaces.Dict({
            "Node_cpu_availability": spaces.Box(low=0, high=32, shape=(num_nodes,), dtype=np.float32),
            "Node_memory_availability": spaces.Box(low=0, high=32, shape=(num_nodes,), dtype=np.float32),
            "Node_cpu_type": spaces.MultiDiscrete([4] * num_nodes),  # 假设有4种CPU类型，长度为num_nodes的数组
            "Node_bandwidth": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            "Node_bandwidth_usage": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            "Node_layer": spaces.MultiDiscrete([3] * num_nodes),  # 假设有3个层级，长度为num_nodes的数组
            # "Node_id": spaces.MultiDiscrete([num_nodes] * num_nodes), #nodes的id
            # "MS_id": spaces.MultiDiscrete([num_ms] * num_ms),  # 对于每个微服务实例，表示其ID
            "MS_node_id": spaces.MultiDiscrete([num_nodes] * num_ms),  # 对于每个微服务实例，表示其所在的节点ID
            "MS_total_bandwidth": spaces.Box(low=0, high=100, shape=(num_ms,), dtype=np.float32),
            "MS_cpu_requests": spaces.Box(low=0, high=8, shape=(num_ms,), dtype=np.float32),
            "MS_memory_requests": spaces.Box(low=0, high=8, shape=(num_ms,), dtype=np.float32)
        })
        self.simulator.output_simulator_status_to_file("status.json")
        # 定义动作空间
        self.action_space = spaces.MultiDiscrete([num_nodes, num_ms])
    def reset(self, seed=None, options=None):
        '''重置当前的simulator'''
        self.final_reward = 100

        if self.episode % self.node_reset_episode_interval == 0:
            '''重新初始化一个simulator状态'''
            self._init_valid_simulator()
            self.app = self.simulator.get_app(self.app_name)
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
        nodes = self.simulator.nodes
        pods = self.app.pods

        # 构建节点的状态
        nodes_state = {
            "Node_cpu_availability": np.array([node.cpu_availability for node in self.simulator.nodes.values()], dtype=np.float32),
            "Node_memory_availability": np.array([node.memory_availability for node in self.simulator.nodes.values()], dtype=np.float32),
            "Node_cpu_type": np.array([cpu_type_map[node.cpu_type] for node in self.simulator.nodes.values()], dtype=np.int32),
            "Node_bandwidth": np.array([node.bandwidth for node in self.simulator.nodes.values()], dtype=np.float32),
            "Node_bandwidth_usage": np.array([node.bandwidth_usage for node in self.simulator.nodes.values()], dtype=np.float32),
            "Node_layer": np.array([layer_map[node.layer] for node in self.simulator.nodes.values()], dtype=np.int32)
        }
        # 构建微服务实例的状态
        ms_state = {
            # "MS_id": np.array(self.pod_ids, dtype=np.int32),
            "MS_node_id": np.array([ms.node_id for ms in self.simulator.apps[self.app_name].pods.values()], dtype=np.int32),
            "MS_total_bandwidth": np.array([ms.total_bandwidth for ms in self.simulator.apps[self.app_name].pods.values()], dtype=np.float32),
            "MS_cpu_requests": np.array([ms.cpu_requests for ms in self.simulator.apps[self.app_name].pods.values()], dtype=np.float32),
            "MS_memory_requests": np.array([ms.memory_requests for ms in self.simulator.apps[self.app_name].pods.values()], dtype=np.float32)
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
        logger.info(f"Scheduling {cur_ms_name} (Target node: {node_id}, Current node: {cur_node_id})")
        cur_latency = self.latency_func()
        qos_threshold = self.qos_func()
        self.episode_steps += 1
        reward = 0
        done = False
        if cur_latency <= qos_threshold:
            done = True
        # elif self.episode_steps >= self.max_episode_steps:
        #     reward = -50
        #     logger.info("Episode steps reached maximum")
        elif self.app.get_pod(ms_id).get_type() == "client":
            reward = -5
            logger.info("Action resulted in a failure: Client node selected")
        elif self.simulator.get_node(node_id).layer == "client":
            reward = -5
            logger.info("Action resulted in a failure: Client node selected")
        elif not self.simulator.check_node_deployable(self.app_name, ms_id, node_id):
            reward = -5
            logger.info("Action resulted in a failure: Node not deployable")
        elif cur_node_id == node_id:
            reward = -5
            done = True
            logger.info("Action resulted in no change: Same node selected")
        else:
            before_latency = cur_latency
            self.simulator.migrate_pods(self.app_name, ms_id, node_id)
            cur_latency = self.latency_func()
            logger.info(f"Latency before action: {before_latency}, Latency after action: {cur_latency}")
            if cur_latency > before_latency:
                reward = -1
                logger.info("Action resulted in worse latency")
            elif cur_latency < self.lowest_latency:
                reward = self.lower_latency_reward
                self.lowest_latency = cur_latency
                self.lower_latency_reward += 3
                logger.info(f"New lowest latency achieved! Reward: {reward}")
                # print(f"New lowest latency achieved! Latency {self.lowest_latency}")
                # print(f"New lowest latency achieved! Reward: {reward}")
            else:
                reward = 3
        
        if not done and cur_latency <= qos_threshold:
            logger.info(f"QoS threshold reached, final reward: {self.final_reward}")
            reward += self.final_reward
        else: 
            self.final_reward //= 2
                
        # 惩罚过多的episode step
        # print("lowest_latency: ", self.lowest_latency)
        # print("reward: ", reward)
        # print(f"cur_latency: {cur_latency}, qos_threshold: {qos_threshold}")
        if self.episode_steps >= self.max_episode_steps or cur_latency <= qos_threshold:
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

    def _init_simulator(self, simulator: MicroserviceSimulator, ms_name: str) -> bool:
        logger.debug("\n部署微服务")
        is_deployed = simulator.deploy_ms_app(ms_name)
        if is_deployed:
            logger.info(f"Microservice {ms_name} successfully deployed")
        else:
            logger.error(f"Microservice {ms_name} deployment failed due to insufficient resources")
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
        for i in range(10):
            self.simulator = MicroserviceSimulator()
            if self._init_simulator(self.simulator, self.app_name):
                return
        assert(False)

    def type_discrete(self, type: List[any])-> Dict[any, int]:
        return {t: i for i, t in enumerate(type)}