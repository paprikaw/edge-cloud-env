import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List
from simulator import MicroserviceSimulator
import random

class MicroserviceEnv(gym.Env):
    def __init__(self, simulator: MicroserviceSimulator):
        super(MicroserviceEnv, self).__init__()
        
        self.simulator = simulator
        self.current_ms = None  # 当前待调度的微服务实例
        self.ms_name = None  # 当前微服务应用的名称
        
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
        self.action_space = spaces.Discrete(num_nodes)  # 动作是选择一个节点

    def reset(self):
        # 重置模拟器状态，随机选择一个微服务应用进行调度
        self.ms_name = random.choice(list(self.simulator.ms_apps.keys()))
        ms_app = self.simulator.ms_apps[self.ms_name]
        
        # 随机选择一个未调度的微服务实例
        unscheduled_instances = [ms for ms in ms_app.instances.values() if ms.node_id == -1]
        if len(unscheduled_instances) == 0:
            return None  # 如果所有实例都已调度，返回 None
        
        self.current_ms = random.choice(unscheduled_instances)
        
        # 构建初始状态
        state = self._get_state()
        return state
    
    def _get_state(self):
        """根据当前环境状态，构建状态空间"""
        # 获取待调度的微服务实例ID
        ms_index = list(self.simulator.ms_apps[self.ms_name].instances.keys()).index(self.current_ms.name)
        
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
        for ms in self.simulator.ms_apps[self.ms_name].instances.values():
            ms_state.append([
                ms.name,
                ms.node_id if ms.node_id != -1 else 0,
                ms.total_bandwidth,
                ms.cpu_requests,
                ms.memory_requests
            ])
        ms_state = np.array(ms_state, dtype=np.float32)
        
        # 返回状态字典
        return {
            "NodeID_To_be_scheduled": ms_index,
            "Nodes": nodes_state,
            "MSs": ms_state
        }

    def step(self, action):
        # 根据动作（选择的NodeID），尝试调度当前的微服务实例
        node_id = list(self.simulator.nodes.keys())[action]
        success = self.simulator.deploy_ms(self.ms_name, self.current_ms.name, node_id)
        
        # 计算奖励和更新状态
        done = False
        reward = 0
        
        if success:
            reward = 1  # 成功调度，给予正奖励
        else:
            reward = -1  # 失败调度，给予负奖励
        
        # 判断是否所有实例都已调度完毕
        unscheduled_instances = [ms for ms in self.simulator.ms_apps[self.ms_name].instances.values() if ms.node_id == -1]
        if len(unscheduled_instances) == 0:
            done = True
        
        # 获取新的状态
        state = self._get_state() if not done else None
        
        return state, reward, done, {}

    def render(self, mode="human"):
        # 可以在此添加可视化代码，显示微服务实例和节点的状态
        pass
