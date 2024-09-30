
import json
import random
from typing import Dict, List
from call import Call

class Pod:
    def __init__(self, name: str, id: int, service_name: str, cpu_requests: float, memory_requests: float, num_replicas: int, type: str, layer: str="all", node_id: int=-1, is_scheduled: bool=True):
        self.name = name
        self.id = id
        self.service_name = service_name
        self.cpu_requests = cpu_requests
        self.memory_requests = memory_requests
        self.num_replicas = num_replicas
        self.node_id = node_id
        self.total_bandwidth = 0
        self.type = type # service / persistent
        self.layer = layer # edge / cloud / client / all(without client layer)
        self.is_scheduled = is_scheduled

    def is_instance_of(self, ms_name):
        return ms_name == self.service_name
    def get_node_id(self):
        return self.node_id
    def get_name(self):
        return self.name
    def get_type(self):
        return self.type
    def get_id(self):
        return self.id
    def __repr__(self):
        return f"<Microservice {self.name}: {self.cpu_requests} CPU, {self.memory_requests} Memory, {self.num_replicas} replicas>"