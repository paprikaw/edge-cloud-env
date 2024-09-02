from call import Call
import json
from typing import Dict, List
import parser
import logging

class Endpoint:
    def __init__(self, name, config: Dict):
        self.name = name
        self.config = config
        self.call_groups: Call = None
        self.rps = config.get("rps", 0)
        self.qos = parser.parse_time(config.get("qos", 0))
        self.process_call()

    def process_call(self):
        """处理调用路径，构建 Call 对象结构"""
        client = self.config.get("client", "")

        # 初始化根 Call 对象
        root_call = Call(client, is_client=True)
        self.call_groups = root_call

        # 递归处理调用路径
        self._dfs_process_call_path(self.config.get("call-path", []), root_call)

    def _dfs_process_call_path(self, call_groups: List[Dict], parent_call: Call):
        """递归处理调用路径，构建 Call 对象"""
        for call_group in call_groups:
            seq_calls = []
            for next_ms_name, call_data in call_group.items():
                data_size = parser.parse_datasize(call_data.get("data-size", "0M"))
                execution_time = {
                    cpu_type: parser.parse_time(time)
                    for cpu_type, time in call_data.get("execution-time", {}).items()
                }

                # 构建 Call 对象
                new_call = Call(next_ms_name, data_size)
                new_call.execution_time = execution_time
                seq_calls.append(new_call)

                # 递归处理下一层的调用路径
                if "call-path" in call_data and call_data["call-path"]:
                    self._dfs_process_call_path(call_data["call-path"], new_call)
            parent_call.call_groups.append(seq_calls)
    def get_qos(self)->float:
        return self.qos
    
    def print_trace(self):
        """递归打印所有的 call_path"""
        logging.info(f"Endpoint: {self.name}, RPS: {self.rps}")
        self._print_call_trace(self.call_groups, level=0)

    def _print_call_trace(self, call: Call, level: int):
        """辅助递归函数，用于打印 Call 对象结构"""
        indent = "  " * level
        logging.info(f"{indent}Call: {call.name}, Data Size: {call.data_size}MB, Execution Time: {call.execution_time}")

        for parallel_calls in call.call_groups:
            for next_call in parallel_calls:
                self._print_call_trace(next_call, level + 1)