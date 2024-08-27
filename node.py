class Node:
    def __init__(self, node_name, node_id, node_type, cpu_type, cpu_availability, memory_availability, bandwidth_usage, bandwidth, layer):
        self.node_id = node_id
        self.node_name = node_name
        self.node_type = node_type
        self.cpu_type = cpu_type
        self.cpu_availability = cpu_availability
        self.memory_availability = memory_availability
        self.bandwidth_usage = bandwidth_usage 
        self.bandwidth = bandwidth
        self.layer  = layer

    def claim_resource(self, cpu_requests: int, memory_requests: int):
        """在节点上申请资源"""
        if not self.check_resource(cpu_requests, memory_requests):
            raise Exception(f"Insufficient resources on node {self.node_id}")
        self.cpu_availability -= cpu_requests
        self.memory_availability -= memory_requests

    def release_resource(self, cpu_requests: int, memory_requests: int):
        """释放节点上的资源"""
        self.cpu_availability += cpu_requests
        self.memory_availability += memory_requests

    def check_resource(self, cpu_requests: int, memory_requests: int)->bool:
        """检查节点上的资源是否足够"""
        return self.cpu_availability >= cpu_requests and self.memory_availability >= memory_requests