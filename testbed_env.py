import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List
from simulator import MicroserviceSimulator
import random
import json
from gymnasium.wrappers import FlattenObservation
import logging
from node import Node
from pod import Pod

logger = logging.getLogger(__name__)
class TestBedEnv(gym.Env):
    """
    Env Version 1:
    RL agent migrates microservices in the application
    RL agent only makes decisions after all microservices are deployed
    """
    def __init__(self, num_nodes, num_pods):
        super(TestBedEnv, self).__init__()
        self.ClusterState = {} 
        self.PodDeployable = {} 
        self.node_layer_map = {
             "tb-cloud-vm1": "cloud",
             "tb-edge-vm1": "edge",
             "tb-edge-vm2": "edge",
             "tb-edge-vm3": "edge",
             "tb-client-vm1": "client"
        }
        self.layer_discrete_map = {
            "cloud": 0,
            "edge": 1,
            "client": 2
        }
        self.node_cpu_type_map = {
             "tb-cloud-vm1": 0,
             "tb-edge-vm1": 2,
             "tb-edge-vm2": 2,
             "tb-edge-vm3": 1,
             "tb-client-vm1": 1
        }
        self.node_bandiwidth_map = {
             "tb-cloud-vm1": 62.5,
             "tb-edge-vm1": 25,
             "tb-edge-vm2": 25,
             "tb-edge-vm3": 25,
             "tb-client-vm1": 25
        }
        self.observation_space = spaces.Dict({
            "Node_id": spaces.Box(low=0, high=num_nodes, shape=(num_nodes,), dtype=np.int32),
            "Node_cpu_availability": spaces.Box(low=0, high=16, shape=(num_nodes,), dtype=np.float32),
            "Node_memory_availability": spaces.Box(low=0, high=16, shape=(num_nodes,), dtype=np.float32),
            "Node_bandwidth_usage": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            "Node_bandwidth": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            "Node_layer": spaces.Box(low=0, high=3, shape=(num_nodes,), dtype=np.int32),
            "Node_cpu_type": spaces.Box(low=0, high=3, shape=(num_nodes,), dtype=np.int32),
            "Pod_node_id": spaces.MultiDiscrete([num_nodes] * num_pods),  # Current node of each microservice
            "Pod_total_bandwidth": spaces.Box(low=0, high=100, shape=(num_pods,), dtype=np.float32), # How much bandwidth a pod has on a node.
            "Pod_cpu_requests": spaces.Box(low=0, high=4, shape=(num_pods,), dtype=np.float32),
            "Pod_memory_requests": spaces.Box(low=0, high=4, shape=(num_pods,), dtype=np.float32)
        })

        with open('node_name_order.json', 'r') as f:
            self.node_name_order = json.load(f)
        with open('service_order.json', 'r') as f:
            self.service_name_order = json.load(f)
        self.num_nodes = num_nodes
        self.num_pods = num_pods
        self.nodes: List[Node] = []
        self.pods: List[Pod] = []

        self.action_space = spaces.Discrete(num_nodes * num_pods+1)
        self.stopped_action = num_nodes * num_pods
        self.all_masked = False
    def reset(self, seed=None, options=None, ClusterState=None, PodDeployable=None):
        self.ClusterState = ClusterState
        self.PodDeployable = PodDeployable
        return self._get_state(), {}
 
    def _get_state(self):
        # 根据node_name_order的顺序来踢去node信息
        node_names = []
        node_cpu_availability = []
        node_memory_availability = []
        node_bandwidth_usage = []
        node_layer = []
        node_cpu_type = []
        node_bandwidth = []
        for node_name in self.node_name_order:
            node = self.ClusterState["nodes"][node_name]
            node_names.append(node_name)
            node_cpu_availability.append(node["cpu_availability"] / 1000.0)
            node_memory_availability.append(node["memory_availability"] / (1024 ** 3))
            node_bandwidth_usage.append(node["bandwidth_usage"] / 1000000.0) # Convert to MB
            node_bandwidth.append(self.node_bandiwidth_map[node_name])
            node_layer.append(self.layer_discrete_map[self.node_layer_map[node_name]])
            node_cpu_type.append(self.node_cpu_type_map[node_name])

        # 提取 Pod 信息
        # pod_node_ids = [self._get_node_id(pod["node_name"], node_names) for pod in self.ClusterState["pods"]]
        # pod_cpu_requests = [0.0] * self.num_pods  # 假设初始CPU请求为0，可以根据实际情况修改
        # pod_memory_requests = [0.0] * self.num_pods  # 假设初始内存请求为0，可以根据实际情况修改

        # 构建 observation space
        observation = {
            "Node_id": np.arange(self.num_nodes),
            "Node_cpu_availability": np.array(node_cpu_availability, dtype=np.float32),
            "Node_memory_availability": np.array(node_memory_availability, dtype=np.float32),
            "Node_bandwidth_usage": np.array(node_bandwidth_usage, dtype=np.float32),
            "Node_bandwidth": np.array(node_bandwidth, dtype=np.float32),
            "Node_layer": np.array(node_layer, dtype=np.int32),
            "Node_cpu_type": np.array(node_cpu_type, dtype=np.int32),
            # "Pod_node_id": np.array(pod_node_ids, dtype=np.int32),
            # "Pod_cpu_requests": np.array(pod_cpu_requests, dtype=np.float32),
            # "Pod_memory_requests": np.array(pod_memory_requests, dtype=np.float32)
        }

        # 返回初始 observation
        return observation 

    def get_action(self, action: int)->tuple[Node, Pod]:
        num_pods = self.num_pods
        node_id = action // num_pods
        pod_id = action % num_pods
        return self.nodes[node_id],self.pods[pod_id] 
    
    def step(self, action):
        if action == self.stopped_action:
            logger.info("Stop action selected")
            logger.info(f"Episode steps: {self.episode_steps}, Final latency: {self.latency_func()}")
            if not self.is_training:
                self.simulator.output_simulator_status_to_file("./logs/test_end.json")
            return None, -5, True, False, {}

        node, pod = self.get_action(action)
        node_id = node.get_id()
        pod_id = pod.get_id()
        cur_node_id = pod.get_node_id()
        cur_pod_name = pod.get_name()
        logger.info(f"Scheduling {cur_pod_name} (node {cur_node_id} -> node {node_id})")
        cur_latency = self.latency_func()
        qos_threshold = self.qos_func()
        self.episode_steps += 1
        reward = 0
        done = False

        # No node can be selected 
        if self.all_masked:
            assert(False)
            done = True
        # QoS threshold reached, episode ends
        elif cur_latency <= qos_threshold:
            logger.info(f"Qos Threshold Reached Before Scheduling: {cur_latency}")
            done = True
        elif cur_node_id == node_id:
            assert(False)
            reward = -5
            done = True
        elif not self.simulator.check_node_deployable(self.app_name, pod_id, node_id):
            assert(False)
            reward = -1000
            if not self.is_training:
                done = True
            logger.info("Action resulted in a failure: Node not deployable")
        else:
            before_latency = cur_latency
            self.simulator.migrate_pods(self.app_name, pod_id, node_id)
            cur_latency = self.latency_func()
            logger.info(f"Latency {before_latency} -> {cur_latency}")
            # if cur_latency > before_latency:
            reward = before_latency - cur_latency
            # if cur_latency <= qos_threshold:
            #     logger.info(f"Qos Threshold Reached!")
            #     reward += 10
            if reward < 0:
                logger.info("Action resulted in worse latency")

        if self.episode_steps >= self.max_episode_steps or cur_latency <= qos_threshold:
            done = True

        if done:
            logger.info(f"Episode steps: {self.episode_steps}, Final latency: {cur_latency}")
            if not self.is_training:
                self.simulator.output_simulator_status_to_file("./logs/test_end.json")
        else:
            reward -= 3
        # logger.info(f"reward: {reward}")
        # logger.info(f"cur_latency: {cur_latency}, qos_threshold: {qos_threshold}")
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
        # false_mask_cnt = 0
        for node in self.nodes:
            # node = self.simulator.get_node(node_id)
            for pod in self.pods:
                # if pod.get_node_id() == node.get_id():
                #     mask.append(True)
                #     false_mask_cnt += 1
                if not pod.is_scheduled or \
                    pod.get_node_id() == node.get_id() or \
                    pod.get_type() == "client" or \
                    node.layer == "client" or \
                    not self.simulator.check_node_deployable(self.app_name, pod.id, node.node_id):

                    mask.append(False)
                    # false_mask_cnt += 1
                else:
                    mask.append(True)
                logger.debug(f"Node {node.get_id()} Pod {pod.get_name()} Mask: {mask[-1]}\n")
        # if false_mask_cnt == len(mask):
        #     self.all_masked = True
        # else:
        #     self.all_masked = False
        
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
            self.simulator = MicroserviceSimulator()
            if self._init_simulator(self.simulator, self.app_name):
                return
        assert(False)

    def type_discrete(self, type: List[any])-> Dict[any, int]:
        return {t: i for i, t in enumerate(type)}