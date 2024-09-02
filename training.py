from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from env import MicroserviceEnv
import logging
import numpy as np

logging.basicConfig(level=logging.ERROR)
# 初始化环境
env = MicroserviceEnv()

# 创建 DQN 模型
# model = DQN("MultiInputPolicy", env, verbose=1)
model = PPO("MultiInputPolicy", env, verbose=1)

class CustomStopTrainingCallback(BaseCallback):
    def __init__(self, reward_threshold, loss_threshold, len_threshold, verbose=0):
        super(CustomStopTrainingCallback, self).__init__(verbose)
        self.reward_threshold = reward_threshold
        self.loss_threshold = loss_threshold
        self.len_threshold = len_threshold
    def _on_step(self) -> bool:
        # 获取最近一批的回报 (reward) 和损失 (loss)
        mean_reward = self.model.logger.name_to_value.get("rollout/ep_rew_mean", -np.inf)
        mean_loss = self.model.logger.name_to_value.get("train/loss", np.inf)
        len_mean = self.model.logger.name_to_value.get("rollout/ep_len_mean", np.inf)
        # 如果损失小于阈值且回报大于阈值，则停止训练
        if mean_loss < self.loss_threshold and mean_reward > self.reward_threshold and len_mean < self.len_threshold:
            print(f"Stopping training as reward {mean_reward} > {self.reward_threshold} "
                  f"and loss {mean_loss} < {self.loss_threshold}")
            return False  # 返回 False 表示停止训练

        return True  # 继续训练
        
# 训练代理
callback = CustomStopTrainingCallback(reward_threshold=40, loss_threshold=250, len_threshold=3)
model.learn(total_timesteps=100000, callback=callback)

# 保存模型
model.save("dqn_microservice")