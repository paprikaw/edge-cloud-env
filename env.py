import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List
from simulator import MicroserviceSimulator
import random


class MicroserviceEnv(gym.Env):
    """
    Env Version 1:
    RL agent migrates microservices in the application
    RL agent only makes decisions after all microservices are deployed
    """
    def __init__(self):
        super(MicroserviceEnv, self).__init__()
        
        self.current_ms = None  # 当前待调度的微服务实例
        self.ms_app_name = "iot-ms-app"  # 当前微服务应用的名称
        self.instance_id_list = []  # 记录等待调度的ms

        # 读取文件路径和相关配置
        self.node_config = 'nodes.json'
        self.microservices_config_path = 'microservices.json'
        self.calls_config_path = 'call_patterns.json'
        
        # 初始化模拟器和环境
        self.simulator = MicroserviceSimulator(self.node_config)
        self._init_simulator(self.simulator, self.ms_app_name)

        # 定义最低延迟和最大奖励
        self.max_reward = 0
        self.endpoints = self.simulator.get_endpoints()
        self.lowest_latency = self.latency_func()
        self.epsoide = 1
        self.node_setup_epsoid = 1000

        # 定义状态空间和动作空间
        num_nodes = len(self.simulator.nodes)
        num_ms = sum(len(app.instances) for app in self.simulator.ms_apps.values())

        # 定义状态空间
        self.observation_space = spaces.Dict({
            "Ms_to_be_scheduled": spaces.Discrete(num_ms),
            "Nodes": spaces.Dict({
                "cpu_availability": spaces.Box(low=0, high=32, shape=(num_nodes,), dtype=np.float32),  # 假设CPU可用量范围为 [0, 32]
                "memory_availability": spaces.Box(low=0, high=32, shape=(num_nodes,), dtype=np.float32),  # 假设内存范围为 [0, 32GB]
                "cpu_type": spaces.Discrete(3),  # 假设有三种CPU类型
                "bandwidth": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),  # 假设带宽范围为 [0, 1000MBps]
                "bandwidth_usage": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),  # 假设带宽使用量范围为 [0, 1000Mbps]
                "layer": spaces.Discrete(2)  # 假设有两个层级（cloud 和 edge）
            }),
            "MSs": spaces.Dict({
                "node_id": spaces.Discrete(num_nodes),
                "total_bandwidth": spaces.Box(low=0, high=100, shape=(num_ms,), dtype=np.float32),  # 假设总带宽范围为 [0, 1000Mbps]
                "cpu_requests": spaces.Box(low=0, high=8, shape=(num_ms,), dtype=np.float32),  # 假设CPU请求范围为 [0, 32]
                "memory_requests": spaces.Box(low=0, high=8, shape=(num_ms,), dtype=np.float32)  # 假设内存请求范围为 [0, 32GB]
            })
        })

        # 定义动作空间
        self.action_space = spaces.Discrete(num_nodes)

    def reset(self):
        if self.epsoide % self.node_setup_epsoid == 0:
            self.simulator = MicroserviceSimulator(self.node_config)
            self._init_simulator(self.simulator, self.ms_app_name)
            self.max_reward = 0
            self.lowest_latency = self.latency_func()
        else:
            self.simulator.reset_ms()
            self.simulator.deploy_ms_app(self.ms_app_name)

        self.epsoide += 1
        # 记录等待调度的ms
        self.instance_id_list = self.simulator.get_all_instances()
        # 构建初始状态
        return self._get_state()
    
    def _get_state(self):
        """根据当前环境状态，构建状态空间"""
        # 获取待调度的微服务实例ID
        self.ms_index = self.instance_id_list[-1]
        
        # 构建节点的状态
        nodes_state = {
            "cpu_availability": np.array([node.cpu_availability for node in self.simulator.nodes.values()], dtype=np.float32),
            "memory_availability": np.array([node.memory_availability for node in self.simulator.nodes.values()], dtype=np.float32),
            "cpu_type": np.array([node.cpu_type for node in self.simulator.nodes.values()], dtype=np.int32),
            "bandwidth": np.array([node.bandwidth for node in self.simulator.nodes.values()], dtype=np.float32),
            "bandwidth_usage": np.array([node.bandwidth_usage for node in self.simulator.nodes.values()], dtype=np.float32),
            "layer": np.array([node.layer for node in self.simulator.nodes.values()], dtype=np.int32)
        }
        
        # 构建微服务实例的状态
        ms_state = {
            "node_id": np.array([ms.node_id for ms in self.simulator.ms_apps[self.ms_app_name].instances.values()], dtype=np.int32),
            "total_bandwidth": np.array([ms.total_bandwidth for ms in self.simulator.ms_apps[self.ms_app_name].instances.values()], dtype=np.float32),
            "cpu_requests": np.array([ms.cpu_requests for ms in self.simulator.ms_apps[self.ms_app_name].instances.values()], dtype=np.float32),
            "memory_requests": np.array([ms.memory_requests for ms in self.simulator.ms_apps[self.ms_app_name].instances.values()], dtype=np.float32)
        }
        
        # 返回状态字典
        return {
            "Ms_to_be_scheduled": self.ms_index,
            "Nodes": nodes_state,
            "MSs": ms_state
        }

    def step(self, action):
        cur_node_id = self.simulator.ms_apps[self.ms_app_name].get_instance(self.ms_index).node_id
        if not self.simulator.check_node_deployable(self.ms_app_name, self.ms_index, action): 
            reward = -100
        elif cur_node_id == action:
            reward = 0
        else:
            before_latency = self.latency_func()
            self.simulator.migrate_microservices(self.ms_app_name, self.ms_index, action)
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
            self.ms_index = self.instance_id_list.pop()

        # 获取新的状态
        state = self._get_state() if not done else None
        return state, reward, done, {}

    def latency_func(self) -> float:
        latency = 0
        for endpoint in self.endpoints:
            latency += self.simulator.end_to_end_latency(self.ms_app_name, endpoint)
        return latency / len(self.endpoints)

    def render(self, mode="human"):
        pass

    def _init_simulator(self, simulator: MicroserviceSimulator, ms_name: str):
        print("\n加载微服务应用")
        simulator.load_ms_app(self.microservices_config_path, self.calls_config_path, ms_name)
        print(f"Microservice {ms_name} loaded with {len(simulator.ms_apps[ms_name].get_instances())} instances")

        print("\n部署微服务")
        is_deployed = simulator.deploy_ms_app(ms_name)
        if is_deployed:
            print(f"Microservice {ms_name} successfully deployed")
        else:
            print(f"Microservice {ms_name} deployment failed due to insufficient resources")

        print("\n开始流量模拟")
        simulator.start_traffic(ms_name)
        for node_id, node in simulator.nodes.items():
            print(f"Node ID: {node_id}, Bandwidth Usage: {node.bandwidth_usage}")
