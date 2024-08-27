from stable_baselines3 import DQN
from env import MicroserviceEnv
from gymnasium.wrappers import FlattenObservation
import logging
logging.basicConfig(level=logging.ERROR)
# 初始化环境
env = MicroserviceEnv()

# 创建 DQN 模型
model = DQN("MultiInputPolicy", env, verbose=1)

# 训练代理
model.learn(total_timesteps=100000)

# 保存模型
model.save("dqn_microservice")

# # 加载模型并进行测试
# model = DQN.load("dqn_microservice")
# state = env.reset()
# done = False

# while not done:
#     action, _states = model.predict(state, deterministic=True)
#     state, reward, done, info = env.step(action)
#     env.render()
