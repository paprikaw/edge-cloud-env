from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from env import MicroserviceEnv
from gymnasium.wrappers import FlattenObservation
import logging

logging.basicConfig(level=logging.INFO)
# 加载模型并进行测试
env = MicroserviceEnv(is_training=False)
model = MaskablePPO.load("maskppo", env=env)
obs, info = env.reset()
done = False

while not done:
    action_masks = env.action_masks()
    action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
    obs, reward, done, _, info = env.step(action)
    env.render()