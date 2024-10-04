import numpy as np
import matplotlib.pyplot as plt

# 定义log-normal分布的参数
mu = np.log(50)  # 设置均值为log(50)，这是log-normal的参数
sigma = 0.5  # 控制分布的形状，值越小，越集中在50附近

# 定义随机数生成函数
def generate_latency_instance(min_val=50, max_val=500):
    while True:
        # 生成log-normal分布的数值
        latency = np.random.lognormal(mu, sigma)
        # 限制在50到500之间
        if min_val <= latency <= max_val:
            return latency

# 生成1000个样本来观察分布
latencies = [generate_latency_instance() for _ in range(10000)]

# 可视化分布
plt.hist(latencies, bins=50, edgecolor='black')
plt.title('Latency Distribution (50-500, concentrated near 50)')
plt.xlabel('Latency')
plt.ylabel('Frequency')
plt.show()