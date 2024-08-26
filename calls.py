from typing import List, Dict

class Call:
    def __init__(self, name, data_size=0):
        self.instance_id = name
        self.data_size = data_size
        self.call_groups: List[List[Call]] = []  # 第一层是并行调用，第二层是顺序调用
        self.execution_time: Dict[str, float] = {}
    def __repr__(self):
        return f"<Call {self.instance_id}: {self.data_size} Data Size, {len(self.call_groups)} Next Calls>"