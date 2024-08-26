def parse_memory(data_size: str)->int:
    """将数据大小转换为以 MB 为单位"""
    if data_size.endswith("M"):
        return float(data_size[:-1])
    if data_size.endswith("G"):
        return float(data_size[:-1]) * 1024
    return 0  # 默认情况为 0M

def parse_datasize(data_size: str)->int:
    return parse_memory(data_size)

def parse_cpu_requests(cpu_requests: str)->int:
    """解析 CPU 请求字符串"""
    if cpu_requests.endswith("m"):
        return int(cpu_requests[:-1]) / 1000
    return int(cpu_requests)

def parse_bandwidth(bandwidth: str)->int:
    """解析带宽字符串"""
    if bandwidth.endswith("m"):
        return int(bandwidth[:-1]) / 8
    raise ValueError("Invalid bandwidth string")

def parse_percentage(number: str)->float:
    """解析百分比字符串"""
    return float(number)

def parse_time(time: str)->float:
    """解析时间字符串，需要能够解析ms, s"""
    if time.endswith("ms"):
        return float(time[:-2])
    if time.endswith("s"):
        return float(time[:-1]) * 1000
    raise ValueError("Invalid time string")