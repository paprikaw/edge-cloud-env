import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List
from simulator import MicroserviceSimulator
import random


class MicroserviceEnv(gym.Env):
    """
    Env Version 1:
    RL agent migrate microservice for every microservice in the application
    RL agent only make decisions after all microservices are deployed
    """ 
    def __init__(self):
        super(MicroserviceEnv, self).__init__()
        
        self.current_ms = None  # 当前待调度的微服务实例
        self.ms_app_name = None  # 当前微服务应用的名称
        # 记录等待调度的ms
        self.id_list = []

        # 读取文件路径和相关配置
        self.profiling_path = 'default_profile.json'
        self.microservices_config_path = 'microservices.json'
        self.calls_config_path = 'call_patterns.json'
        self.node_config = 'nodes.json'

    
        # 创建 MicroserviceEnvironment 实例并且初始化
        self.ms_app_name = "iot-ms-app"
        self.simulator = MicroserviceSimulator(self.node_config, self.ms_app_name)
        self._init_simulator(self.simulator, self.ms_app_name)

        self.lowest_latency = self.latency_func()
        self.max_reward = 0
        self.endpoints = self.simulator.get_endpoints()

        # 定义状态空间和动作空间
        num_nodes = len(self.simulator.nodes)
        num_ms = sum(len(app.instances) for app in self.simulator.ms_apps.values())

        # 定义状态空间
        self.observation_space = spaces.Dict({
            "NodeID_To_be_scheduled": spaces.Discrete(num_ms),
            "Nodes": spaces.Box(low=0, high=np.inf, shape=(num_nodes, 7), dtype=np.float32),  # 7个特征
            "MSs": spaces.Box(low=0, high=np.inf, shape=(num_ms, 5), dtype=np.float32)  # 5个特征
        })

        # 定义动作空间
        self.action_space = spaces.Discrete(num_nodes)


    def reset(self):
        # 重置模拟器状态
        self.simulator = MicroserviceSimulator(self.node_config)
        self._init_simulator(self.simulator, self.ms_app_name)

        # 记录等待调度的ms
        self.id_list = self.simulator.get_all_instances()
        # 构建初始状态
        return self._get_state()
    
    def _get_state(self):
        """根据当前环境状态，构建状态空间"""
        # 获取待调度的微服务实例ID
        self.ms_index = self.id_list[len(self.id_list) - 1]
        # 构建节点的状态
        nodes_state = []
        for node in self.simulator.nodes.values():
            nodes_state.append([
                node.node_id,
                node.cpu_availability,
                node.memory_availability,
                node.cpu_type,
                node.bandwidth,
                node.bandwidth_usage,
                node.layer
            ])
        nodes_state = np.array(nodes_state, dtype=np.float32)
        
        # 构建微服务实例的状态
        ms_state = []
        for ms in self.simulator.ms_apps[self.ms_app_name].instances.values():
            assert ms.node_id != -1
            ms_state.append([
                ms.id,
                ms.node_id,
                ms.total_bandwidth,
                ms.cpu_requests,
                ms.memory_requests
            ])
        ms_state = np.array(ms_state, dtype=np.float32)
        
        # 返回状态字典
        return {
            "MSID": self.ms_index,
            "Nodes": nodes_state,
            "MSs": ms_state
        }

    def step(self, action):
        cur_node_id = self.simulator.ms_apps[self.ms_app_name].get_instance(self.ms_index).node_id
        if not self.simulator.check_node_deployable(self.ms_app_name, self.ms_index, action): 
            reward = -100
        # 根据动作（选择的NodeID），尝试调度当前的微服务实例
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

        if len(self.id_list) == 0:
            done = True
        else:
            done = False
            self.ms_index = self.id_list.pop()

        # 获取新的状态
        state = self._get_state() if not done else None
        return state, reward, done, {}
    def latency_func(self) -> float:
        latency = 0
        for endpoint in self.endpoints:
            latency += self.simulator.end_to_end_latency(self.ms_app_name, endpoint)
        return latency / len(self.endpoints)

    def render(self, mode="human"):
        # 可以在此添加可视化代码，显示微服务实例和节点的状态
        pass

    def _init_simulator(self, simulator:MicroserviceSimulator, ms_name: str):
        # 测试 2: 加载微服务应用
        print("\n加载微服务应用")
        simulator.load_ms_app(self.microservices_config_path, self.calls_config_path, ms_name)
        print(f"Microservice {ms_name} loaded with {len(simulator.ms_apps[ms_name].get_instances())} instances")

        # 测试 3: 部署微服务
        print("\n部署微服务")
        is_deployed = simulator.deploy_ms_app(ms_name)
        if is_deployed:
            print(f"Microservice {ms_name} successfully deployed")
        else:
            print(f"Microservice {ms_name} deployment failed due to insufficient resources")

        # 测试 4: 开始流量模拟
        print("\n开始流量模拟")
        simulator.start_traffic(ms_name)
        for node_id, node in simulator.nodes.items():
            print(f"Node ID: {node_id}, Bandwidth Usage: {node.bandwidth_usage}")