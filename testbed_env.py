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
    def __init__(self, num_nodes, num_pods, layer_latency):
        super(TestBedEnv, self).__init__()
        self.ClusterState = {} 
        self.PodDeployable = {} 
        self.node_layer_map = {
             "tb-cloud-vm-8-1": "cloud",
             "tb-cloud-vm-8-2": "cloud",
             "tb-edge-vm-2-1": "edge",
             "tb-edge-vm-2-2": "edge",
             "tb-edge-vm-4-1": "edge",
             "tb-edge-vm-4-2": "edge",
             "tb-client-vm-2-1": "client"
        }
        self.layer_discrete_map = {
            "cloud": 0,
            "edge": 1,
            "client": 2
        }
        self.node_cpu_type_map = {
             "tb-cloud-vm-8-1": 8,
             "tb-cloud-vm-8-2": 8,
             "tb-edge-vm-2-1": 2,
             "tb-edge-vm-2-2": 2,
             "tb-edge-vm-4-1": 4,
             "tb-edge-vm-4-2": 4,
             "tb-client-vm-2-1": 2
        }
        self.node_bandiwidth_map = {
             "tb-cloud-vm-8-1": 62.5,
             "tb-cloud-vm-8-2": 62.5,
             "tb-edge-vm-2-1": 25,
             "tb-edge-vm-2-2": 25,
             "tb-edge-vm-4-1": 25,
             "tb-edge-vm-4-2": 25,
             "tb-client-vm-2-1": 25
        }
        self.pod_memory_map = {
            "client": 700,
            "aggregator": 700,
            "detection": 1000,
            "machine-learning": 2000,
            "db": 700,
        }

        self.pod_cpu_map = {
            "client": 0.7,
            "aggregator": 1,
            "detection": 0.7, 
            "machine-learning": 2,
            "db": 0.7,
        }
        self.service_replica_cnt = {
            "client": 1,
            "aggregator": 3,
            "detection": 3, 
            "machine-learning": 3,
            "db": 3,
        }
        self.observation_space = spaces.Dict({
            # "Node_id": spaces.Box(low=0, high=num_nodes, shape=(num_nodes,), dtype=np.int32),
            "Node_cpu_availability": spaces.Box(low=0, high=16, shape=(num_nodes,), dtype=np.float32),
            "Node_memory_availability": spaces.Box(low=0, high=16, shape=(num_nodes,), dtype=np.float32),
            # "Node_bandwidth_usage": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            # "Node_bandwidth": spaces.Box(low=0, high=1000, shape=(num_nodes,), dtype=np.float32),
            # "Node_layer": spaces.Box(low=0, high=3, shape=(num_nodes,), dtype=np.int32),
            # "Node_cpu_type": spaces.Box(low=0, high=3, shape=(num_nodes,), dtype=np.int32),
            "Pod_node_id": spaces.MultiDiscrete([num_nodes+1] * num_pods),  # Current node of each microservice
            # "Layer_latency": spaces.Box(low=0, high=300, shape=(1,), dtype=np.float32),
            # "Pod_total_bandwidth": spaces.Box(low=0, high=100, shape=(num_pods,), dtype=np.float32), # How much bandwidth a pod has on a node.
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
        self.layer_latency = layer_latency
        self.action_space = spaces.Discrete(num_nodes * num_pods+1)
        self.stopped_action = num_nodes * num_pods
        self.all_masked = False

    def reset(self, seed=None, options=None, ClusterState=None, PodDeployable=None):
        self.ClusterState = ClusterState
        self.PodDeployable = PodDeployable
        return self._get_state(), {}
 
    def _get_state(self):
        # 根据node_name_order的顺序来踢去node信息
        self.node_cpu_availability = []
        self.node_memory_availability = []
        self.node_bandwidth_usage = []
        self.node_layer = []
        self.node_cpu_type = []
        self.node_bandwidth = []
        self.node_name_id_map = {}
        self.node_is_client = []
        for i, node_name in enumerate(self.node_name_order):
            self.node_is_client.append(self.node_layer_map[node_name] == "client")
            self.node_name_id_map[node_name] = i
            node = self.ClusterState["nodes"][node_name]
            self.node_cpu_availability.append(node["cpu_availability"] / 1000.0)
            self.node_memory_availability.append(node["memory_availability"])
            self.node_bandwidth_usage.append(node["bandwidth_usage"] / 1000000.0) # Convert to MB
            self.node_bandwidth.append(self.node_bandiwidth_map[node_name])
            self.node_layer.append(self.layer_discrete_map[self.node_layer_map[node_name]])
            self.node_cpu_type.append(self.node_cpu_type_map[node_name])

        self.pod_node_ids = []
        self.pod_cpu_requests = []
        self.pod_memory_requests = []
        self.pod_is_scheduled = []
        self.pod_name = []
        self.pod_is_client = []
        for service_name in self.service_name_order:
            for pod in self.ClusterState["services"][service_name]["pods"]:
                self.pod_node_ids.append(self.node_name_id_map[pod["node_name"]])
                self.pod_cpu_requests.append(self.pod_cpu_map[service_name])
                self.pod_memory_requests.append(self.pod_memory_map[service_name] / 1000.0)
                self.pod_is_scheduled.append(True)
                self.pod_name.append(pod["pod_name"])
                if service_name == "client":
                    self.pod_is_client.append(True)
                else:
                    self.pod_is_client.append(False)
            for _ in range(len(self.ClusterState["services"][service_name]["pods"]), self.service_replica_cnt[service_name]):
                self.pod_node_ids.append(0)
                self.pod_cpu_requests.append(0)
                self.pod_memory_requests.append(0)
                self.pod_is_scheduled.append(False)      
                self.pod_name.append(service_name)
                self.pod_is_client.append(False)

        # 构建 observation space
        observation = {
            # "Node_id": np.arange(self.num_nodes, dtype=np.int32),
            "Node_cpu_availability": np.array(self.node_cpu_availability, dtype=np.float32),
            "Node_memory_availability": np.array(self.node_memory_availability, dtype=np.float32),
            # "Node_bandwidth_usage": np.array(self.node_bandwidth_usage, dtype=np.float32),
            # "Node_bandwidth": np.array(self.node_bandwidth, dtype=np.float32),
            # "Node_layer": np.array(self.node_layer, dtype=np.int32),
            # "Node_cpu_type": np.array(self.node_cpu_type, dtype=np.int32),
            "Pod_node_id": np.array(self.pod_node_ids, dtype=np.int32),
            # "Layer_latency": np.array([self.layer_latency], dtype=np.float32),
            "Pod_cpu_requests": np.array(self.pod_cpu_requests, dtype=np.float32),
            "Pod_memory_requests": np.array(self.pod_memory_requests, dtype=np.float32)
        }

        # 返回初始 observation
        return observation 
    def get_action(self, action: int)->tuple[int, int]:
        num_pods = self.num_pods
        node_id = action // num_pods
        pod_id = action % num_pods
        return self.node_name_order[node_id],self.pod_name[pod_id] 

    def action_masks(self)->List[bool]:
        """
        Returns an action mask to filter out invalid actions.
        The mask is a boolean array where True indicates a valid action.
        """
        mask = []
        # false_mask_cnt = 0
        for i, node_name in enumerate(self.node_name_order):
            # node = self.simulator.get_node(node_id)
            for j, pod_name in enumerate(self.pod_name):
                # if pod.get_node_id() == node.get_id():
                #     mask.append(True)
                #     false_mask_cnt += 1

                mask_flag = False
                if not self.pod_is_scheduled[j] or \
                    self.pod_node_ids[j] == i or \
                    self.node_is_client[i] or \
                    self.pod_is_client[j] or \
                    node_name not in self.PodDeployable[pod_name]:
                    mask_flag = False
                else:
                    mask_flag = True
                mask.append(mask_flag)
                logger.debug(f"node_name: {node_name}, pod_name: {pod_name}, maskFlag:{mask_flag}")
        mask.append(True) # Stop action is always valid

        return mask