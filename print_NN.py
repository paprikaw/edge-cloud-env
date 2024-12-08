import gymnasium as gym
from stable_baselines3 import PPO, DQN

# 创建一个示例环境，这里使用CartPole-v1作为测试
env = gym.make("CartPole-v1")

# 初始化PPO模型，使用默认的网络结构
ppo_agent = PPO("MlpPolicy", env, verbose=1)

# 或者初始化DQN模型
dqn_agent = DQN("MlpPolicy", env, verbose=1)

# 现在可以查看默认的网络结构
print(ppo_agent.policy)
print(dqn_agent.policy)
