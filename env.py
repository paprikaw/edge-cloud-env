import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List
from simulator import MicroserviceSimulator
import random
from gymnasium.wrappers import FlattenObservation
import logging

class MicroserviceEnv(gym.Env):
    """
    Env Version 1:
    RL agent migrates microservices in the application
    RL agent only makes decisions after all microservices are deployed
    """
    def __init__(self):
        logging.basicConfig(level=logging.ERROR)
        super(MicroserviceEnv, self).__init__()
        
        self.current_ms = None  # 当前待调度的微服务实例
        self.ms_app_name = "iot-ms-app"  # 当前微服务应用的名称
        self.instance_id_list = []  # 记录等待调度的ms

        # 读取文件路径和相关配置
        # self.node_config = 'nodes.json'
        # self.microservices_config_path = 'microservices.json'
        # self.calls_config_path = 'call_patterns.json'
        self._init_valid_simulator()
        # 定义最低延迟和最大奖励
        self.max_reward = 0
        self.endpoints = self.simulator.get_endpoints()
        self.lowest_latency = self.latency_func()
        self.epsoide = 1
        self.node_setup_epsoid = 10000

        # 定义状态空间和动作空间
        self.nodes = self.simulator.get_schedulable_nodes()
        num_nodes = len(self.simulator.nodes)
        num_ms = sum(len(app.instances) for app in self.simulator.ms_apps.values())
        self.observation_space = spaces.Dict({
            "Ms_to_be_scheduled": spaces.Discrete(num_ms),
            "Node_cpu_availability": spaces.Box(low=0, high=32, shape=(num_nodes,), dtype=np.float32),
            "Node_memory_availability": spaces.Box(low=0, high=32, shape=(num_nodes,), dtype=np.float32),
            "Node_cpu_type": spaces.MultiDiscrete([4] * num_nodes),  # 假设有4种CPU类型，长度为num_nodes的数组
            "Node_bandwidth": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            "Node_bandwidth_usage": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            "Node_layer": spaces.MultiDiscrete([3] * num_nodes),  # 假设有3个层级，长度为num_nodes的数组
            # "MS_node_id": spaces.MultiDiscrete([num_nodes] * num_ms),  # 对于每个微服务实例，表示其所在的节点ID
            # "MS_total_bandwidth": spaces.Box(low=0, high=100, shape=(num_ms,), dtype=np.float32),
            # "MS_cpu_requests": spaces.Box(low=0, high=8, shape=(num_ms,), dtype=np.float32),
            # "MS_memory_requests": spaces.Box(low=0, high=8, shape=(num_ms,), dtype=np.float32)
        })

        # 定义动作空间
        self.action_space = spaces.Discrete(len(self.nodes))

    def reset(self, seed=None, options=None):
        if self.epsoide % self.node_setup_epsoid == 0:
            '''重新初始化一个simulator'''
            self._init_valid_simulator()
            self.max_reward = 0
            self.lowest_latency = self.latency_func()
        else:
            '''重置当前的simulator'''
            self.simulator.reset_ms(self.ms_app_name)
            self.simulator.deploy_ms_app(self.ms_app_name)

        self.epsoide += 1
        # 记录等待调度的ms
        self.instance_id_list = self.simulator.get_all_instances()
        # 构建初始状态
        return self._get_state(), {}
 
    def _get_state(self):
        """根据当前环境状态，构建状态空间"""
        # 获取待调度的微服务实例ID
        self.ms_index = self.instance_id_list.pop()
        # 构建CPU类型和node layer的名字到数字的映射
        cpu_type_map = self.type_discrete(self.simulator.get_cpu_types())
        layer_map = self.type_discrete(self.simulator.get_node_layers())

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
        # ms_state = {
        #     "MS_node_id": np.array([ms.node_id for ms in self.simulator.ms_apps[self.ms_app_name].instances.values()], dtype=np.int32),
        #     "MS_total_bandwidth": np.array([ms.total_bandwidth for ms in self.simulator.ms_apps[self.ms_app_name].instances.values()], dtype=np.float32),
        #     "MS_cpu_requests": np.array([ms.cpu_requests for ms in self.simulator.ms_apps[self.ms_app_name].instances.values()], dtype=np.float32),
        #     "MS_memory_requests": np.array([ms.memory_requests for ms in self.simulator.ms_apps[self.ms_app_name].instances.values()], dtype=np.float32)
        # }
        
        # 返回状态字典
        state = {
            "Ms_to_be_scheduled": self.ms_index,
            **nodes_state,
            # **ms_state
        }
        return state

    def step(self, action:int):
        target_node_id = self.nodes[action]
        cur_node_id = self.simulator.ms_apps[self.ms_app_name].get_instance(self.ms_index).node_id
        if not self.simulator.check_node_deployable(self.ms_app_name, self.ms_index, target_node_id): 
            reward = -100
        elif cur_node_id == target_node_id:
            reward = 0
        else:
            before_latency = self.latency_func()
            self.simulator.migrate_microservices(self.ms_app_name, self.ms_index, target_node_id)
            after_latency = self.latency_func()
            if after_latency > before_latency:
                reward = -10
            elif after_latency < self.lowest_latency:
                self.max_reward += 10
                reward = self.max_reward
            else:
                reward = self.max_reward

        if len(self.instance_id_list) == 0:
            done = True
        else:
            done = False

        # 获取新的状态
        state = self._get_state() if not done else None
        return state, reward, done, False, {}

    def latency_func(self) -> float:
        latency = 0
        for endpoint in self.endpoints:
            latency += self.simulator.end_to_end_latency(self.ms_app_name, endpoint)
        return latency / len(self.endpoints)

    def render(self, mode="human"):
        pass

    def _init_simulator(self, simulator: MicroserviceSimulator, ms_name: str) -> bool:
        logging.debug("\n部署微服务")
        is_deployed = simulator.deploy_ms_app(ms_name)
        if is_deployed:
            logging.info(f"Microservice {ms_name} successfully deployed")
        else:
            logging.error(f"Microservice {ms_name} deployment failed due to insufficient resources")
            return False

        logging.debug("\n开始流量模拟")
        for node_id, node in simulator.nodes.items():
            logging.debug(f"Node ID: {node_id}, Bandwidth Usage: {node.bandwidth_usage}")
        return True

    def _init_valid_simulator(self):
        '''
        初始化模拟器，重复尝试10次
        在一些情况下，可能microservice无法在既有资源下完成部署
        '''
        for i in range(10):
            self.simulator = MicroserviceSimulator()
            if self._init_simulator(self.simulator, self.ms_app_name):
                return
        assert(False)

    def type_discrete(self, type: List[any])-> Dict[any, int]:
        return {t: i for i, t in enumerate(type)}

env = MicroserviceEnv()
env = FlattenObservation(env)