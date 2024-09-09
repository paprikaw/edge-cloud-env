class Service:
    def __init__(self, name, max_replica_cnt, sched_replica_cnt):
        self.name = name
        self.pod_ids = []
        self.sched_replica_cnt = sched_replica_cnt
        self.max_replica_cnt = max_replica_cnt

    def get_name(self):
        return self.name
    
    def add_pod(self, pod_id):
        self.pod_ids.append(pod_id)
    
    def get_pods(self):
        return self.pod_ids