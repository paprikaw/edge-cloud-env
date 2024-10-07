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
from sb3_contrib import MaskablePPO
logger = logging.getLogger(__name__)
class MicroserviceMaskEnv(gym.Env):
    """
    Env Version 1:
    RL agent migrates microservices in the application
    RL agent only makes decisions after all microservices are deployed
    """
    def __init__(self, is_training=True, dynamic_env=True, num_nodes=0, num_pods=0, relative_para=None, accumulated_para=None):
        if relative_para is None or accumulated_para is None:
            raise ValueError("relative_para is required")
        self.relative_para = relative_para
        self.accumulated_para = accumulated_para
        super(MicroserviceMaskEnv, self).__init__()
        self.microservices_config_path = './config/services.json'
        self.calls_config_path = './config/call_patterns.json'
        self.node_config_path = './config/nodes.json'
        if not dynamic_env:
            self.node_config_path = './config/nodes-simple.json'
        self.current_ms = None  # 当前待调度的微服务实例
        self.app_name = "iot-ms-app"  # 当前微服务应用的名称
        self.is_training = is_training

        # 定义最低延迟和最大奖励
        self.episode = 0
        self.cluster_reset_interval_by_episode = 1
        if is_training:
            self.max_episode_steps = 1000
        else:
            self.max_episode_steps = 100
        self.num_nodes = num_nodes
        self.num_pods = num_pods
        self.nodes: List[Node] = []
        self.pods: List[Pod] = []
        self.action_space = spaces.Discrete(num_nodes * num_pods+1)
        self.stopped_action = num_nodes * num_pods
        self.observation_space = spaces.Dict({
            # "Node_id": spaces.Box(low=0, high=num_nodes, shape=(num_nodes,), dtype=np.int32),
            "Node_cpu_availability": spaces.Box(low=0, high=16, shape=(num_nodes,), dtype=np.float32),
            "Node_memory_availability": spaces.Box(low=0, high=16, shape=(num_nodes,), dtype=np.float32),
            # "Node_bandwidth_usage": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            # "Node_bandwidth": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            # "Node_layer": spaces.Box(low=0, high=3, shape=(num_nodes,), dtype=np.int32),
            # "Node_cpu_type": spaces.Box(low=0, high=3, shape=(num_nodes,), dtype=np.int32),
            "Pod_node_id": spaces.MultiDiscrete([num_nodes+1] * num_pods),  # Current node of each microservice
            # "Pod_layer": spaces.Box(low=0, high=4, shape=(num_pods,), dtype=np.int32),
            # "Pod_type": spaces.Box(low=0, high=2, shape=(num_pods,), dtype=np.int32),
            # "Pod_total_bandwidth": spaces.Box(low=0, high=100, shape=(num_pods,), dtype=np.float32), # How much bandwidth a pod has on a node.
            # "Pod_cpu_requests": spaces.Box(low=0, high=4, shape=(num_pods,), dtype=np.float32),
            # "Pod_memory_requests": spaces.Box(low=0, high=4, shape=(num_pods,), dtype=np.float32),
            # "client_latency": spaces.Box(low=0, high=500, shape=(3,), dtype=np.float32),
            # "edge_latency": spaces.Box(low=0, high=500, shape=(3,), dtype=np.float32),
            # "cloud_latency": spaces.Box(low=0, high=500, shape=(3,), dtype=np.float32),
            "Latency": spaces.Box(low=0, high=300, shape=(1,), dtype=np.float32),
            # "Cur_latency": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
            # "Latency": spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32),
            "time_step": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)
        })

        # "Node_cpu_type": spaces.MultiDiscrete([4] * num_nodes),
        # "Node_layer": spaces.MultiDiscrete([3] * num_nodes),
        # "Node_id": spaces.MultiDiscrete([num_nodes] * num_nodes), #nodes的id
        # "Pod_id": spaces.MultiDiscrete([num_ms] * num_ms),  # 对于每个微服务实例，表示其ID
        # 定义动作空间
        # self.action_space = spaces.MultiDiscrete([num_nodes, num_ms])
    def reset(self, seed=None, options=None):
        '''Reset simulator, this happened during the end of the episode'''
        self.isDone = False
        self.episode_steps = 0
        self.cloud_latency = random.uniform(50, 200)
        # if self.episode % self.cluster_reset_interval_by_episode == 0:
        '''重新初始化一个simulator状态'''
        self._init_valid_simulator()
        if not self.is_training and self.episode == 0:
            self.simulator.output_simulator_status_to_file("./logs/test_start.json")
        self.app = self.simulator.get_app(self.app_name)
        self.node_ids = self.simulator.get_node_ids()
        self.endpoints = self.app.get_endpoints()
        logger.warning("---------------- Episode Resetted ----------------")
        logger.warning(f"cloud latency: {self.simulator.get_latency_between_layers('client', 'cloud')}")
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
        layer_map = {
            "cloud": 0,
            "edge": 1,
            "client": 2,
        }

        # 构建节点的状态
        nodes_state = {
            # "Node_id": np.array([node.node_id for node in self.nodes], dtype=np.int32),
            "Node_cpu_availability": np.array([node.cpu_availability for node in self.nodes], dtype=np.float32),
            "Node_memory_availability": np.array([node.memory_availability for node in self.nodes], dtype=np.float32),
            # "Latency": np.array([self.simulator.get_latency_between_layers("client", "cloud")], dtype=np.float32),
            "Latency": np.array([self.latency_func()], dtype=np.float32),
            # "Cur_latency": np.array([self.latency_func()], dtype=np.float32),
            # "Node_cpu_type": np.array([int(node.cpu_type) for node in self.nodes], dtype=np.int32),
            # "Node_bandwidth": np.array([node.bandwidth for node in self.nodes], dtype=np.float32),
            # "Node_bandwidth_usage": np.array([node.bandwidth_usage for node in self.nodes], dtype=np.float32),
            # "Node_layer": np.array([layer_map[node.layer] for node in self.nodes], dtype=np.int32),
            # "client_latency": np.array([self.simulator.get_latency_between_layers("client", "client"),
            #                             self.simulator.get_latency_between_layers("client", "edge"),
            #                             self.simulator.get_latency_between_layers("client", "cloud")], dtype=np.float32),
            # "edge_latency": np.array([ self.simulator.get_latency_between_layers("edge", "edge"),
            #                           self.simulator.get_latency_between_layers("edge", "client"),
            #                           self.simulator.get_latency_between_layers("edge", "cloud")], dtype=np.float32),
            # "cloud_latency": np.array([self.simulator.get_latency_between_layers("cloud", "cloud"),
            #                           self.simulator.get_latency_between_layers("cloud", "client"),
            #                           self.simulator.get_latency_between_layers("cloud", "edge")], dtype=np.float32),
        }

        # 构建微服务实例的状态
        ms_state = {
            # "Pod_id": np.array(self.pod_ids, dtype=np.int32),
            "Pod_node_id": np.array([pod.node_id for pod in self.pods], dtype=np.int32),
            # "Pod_layer": np.array([pod.layer for pod in self.pods], dtype=np.int32),
            # "Pod_total_bandwidth": np.array([pod.total_bandwidth for pod in self.pods], dtype=np.float32),
            # "Pod_cpu_requests": np.array([pod.cpu_requests for pod in self.pods], dtype=np.float32),
            # "Pod_memory_requests": np.array([pod.memory_requests for pod in self.pods], dtype=np.float32),
        }
        # 返回状态字典
        state = {
            **nodes_state,
            **ms_state,
            "time_step": np.array([self.episode_steps], dtype=np.int32)
        }
        logger.info(f"state: {state}")
        # print(state)
        return state

    def set_cloud_latency(self, latency):
        self.cloud_latency = latency
        self.simulator.set_cloud_latency(latency)
    def set_cur_timestep(self, timestep):
        self.episode_steps = timestep

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
        if action == self.stopped_action:
            logger.warning("Stop action selected")
            logger.warning(f"Episode steps: {self.episode_steps}, Final latency: {self.latency_func()}")
            if not self.is_training:
                self.simulator.output_simulator_status_to_file("./logs/test_end.json")
            self.isDone = True
            return None, 0, True, False, {}
        node, pod = self.get_action(action)
        node_id = node.get_id()
        pod_id = pod.get_id()
        cur_node_id = pod.get_node_id()
        cur_pod_name = pod.get_name()
        logger.warning(f"Scheduling {cur_pod_name}: \n node {self.simulator.get_node(cur_node_id).node_name} -> node {node.node_name}")
        cur_latency = self.latency_func()
        qos_threshold = self.qos_func()
        self.episode_steps += 1
        reward = 0
        done = False

        if cur_latency <= qos_threshold:
            assert(False) 
            logger.info(f"Qos Threshold Reached Before Scheduling: {cur_latency}")
            done = True
        elif not self.check_valid_action(pod, node):
            assert(False)
        else:
            before_latency = cur_latency
            self.simulator.migrate_pods(self.app_name, pod_id, node_id)
            cur_latency = self.latency_func()
            logger.warning(f"Latency {before_latency} -> {cur_latency}")
            # if cur_latency > before_latency:
            #     reward = 2*(before_latency - cur_latency)
            # else:
            reward = before_latency - cur_latency
            # if cur_latency <= qos_threshold:
            #     logger.info(f"Qos Threshold Reached!")
            #     reward += 10
            if reward < 0:
                logger.warning("Action resulted in worse latency")

        if self.episode_steps >= self.max_episode_steps or (not self.is_training and cur_latency <= qos_threshold):
            done = True

        if done:
            logger.info(f"Episode steps: {self.episode_steps}, Final latency: {cur_latency}")
            if cur_latency <= qos_threshold:
                reward += 100
            if not self.is_training:
                self.simulator.output_simulator_status_to_file("./logs/test_end.json")
        else:
            reward -= (cur_latency / self.relative_para) + self.accumulated_para * self.episode_steps
            # reward -= 2 + self.accumulated_para * self.episode_steps
        # logger.info(f"reward: {reward}")
        # logger.info(f"cur_latency: {cur_latency}, qos_threshold: {qos_threshold}")
        state = self._get_state() if not done else None
        self.isDone = done
        return state, reward, done, False, {"terminal_observation": state}
    def is_done(self)->bool:
        return self.isDone
    # def latency_func(self) -> float:
    #     total_contribution = 0
    #     total_latency = 0
    #     for endpoint in self.endpoints:
    #         latency = self.simulator.end_to_end_latency(self.app_name, endpoint)
    #         contribution = 1 / (1 + latency)  # 使用倒数函数来实现低延迟贡献高，高延迟贡献低
    #         total_contribution += contribution
    #         total_latency += latency

    #     if total_contribution == 0:
    #         return float('inf')  # 避免除以零的情况
        
    #     return total_latency / total_contribution  # 返回加权平均延迟
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

    def end_reward(self, para: float)->float:
        return 

    def check_valid_action(self, pod, node)->bool:
        if not pod.is_scheduled or \
            pod.get_node_id() == node.get_id() or \
            pod.get_type() == "persistent" or \
            pod.layer == "client" or \
            node.layer == "client" or \
            not self.simulator.check_node_deployable(self.app_name, pod.id, node.node_id):
            return False
        return True

    def action_masks(self)->List[bool]:
        """
        Returns an action mask to filter out invalid actions.
        The mask is a boolean array where True indicates a valid action.
        """
        mask = []
        # false_mask_cnt = 0
        for node in self.nodes:
            # node = self.simulator.get_node(node_id)
            for pod in self.pods:
                # if pod.get_node_id() == node.get_id():
                #     mask.append(True)
                #     false_mask_cnt += 1
                if self.check_valid_action(pod, node):
                    mask.append(True)
                else:
                    mask.append(False)
                logger.info(f"Node {node.node_name} Pod {pod.get_name()} Mask: {mask[-1]}\n")
        mask.append(True) # Stop action is always valid

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
            self.simulator = MicroserviceSimulator(self.microservices_config_path, self.calls_config_path, self.node_config_path, self.cloud_latency)
            if self._init_simulator(self.simulator, self.app_name):
                return
        assert(False)

    def type_discrete(self, type: List[any])-> Dict[any, int]:
        return {t: i for i, t in enumerate(type)}