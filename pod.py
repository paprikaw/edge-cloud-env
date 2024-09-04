
import json
import random
from typing import Dict, List
from call import Call


class Pod:
    def __init__(self, name: str, id: int, original_name: str, cpu_requests: float, memory_requests: float, num_replicas: int, type: str):
        self.name = name
        self.id = id
        self.original_name = original_name
        self.cpu_requests = cpu_requests
        self.memory_requests = memory_requests
        self.num_replicas = num_replicas
        self.node_id = -1
        self.total_bandwidth = 0
        self.type = type # service / client

    def is_instance_of(self, ms_name):
        return ms_name == self.original_name
    def get_node_id(self):
        return self.node_id
    def get_name(self):
        return self.name
    def get_type(self):
        return self.type
    def __repr__(self):
        return f"<Microservice {self.name}: {self.cpu_requests} CPU, {self.memory_requests} Memory, {self.num_replicas} replicas>"