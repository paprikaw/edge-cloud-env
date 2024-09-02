from stable_baselines3 import PPO
from env import MicroserviceEnv
from gymnasium.wrappers import FlattenObservation
import logging

logging.basicConfig(level=logging.INFO)
# 加载模型并进行测试
env = MicroserviceEnv()
model = PPO.load("dqn_microservice")
obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    env.render()