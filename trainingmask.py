from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3 import DQN
from stable_baselines3 import DDPG
from sb3_contrib import MaskablePPO
from agents.a2c import A2C
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from maskenv import MicroserviceMaskEnv 
import logging
logging.basicConfig(level=logging.ERROR)
# version = "v12-mask-ppo-latency/diffstepdiff-staticenv-200"
version = "v14/mask-ppo/dynamicenv-200"
if __name__ == "__main__":
    env = MicroserviceMaskEnv(num_nodes=7, num_pods=13, dynamic_env=True)
    env = Monitor(env)
    eval_callback = MaskableEvalCallback(
        env,                       
        best_model_save_path='./models/' + version,
        log_path='./logs/results/',       
        eval_freq=10000,                  
        deterministic=True,
        render=False,
        n_eval_episodes=50
    )

    # # 自定义回调函数来记录训练信息
    # class CustomCallback(BaseCallback):
    #     def __init__(self, verbose=0):
    #         super(CustomCallback, self).__init__(verbose)
    #         self.csv_file = open('./logs/training_info.csv', 'w')
    #         self.csv_file.write("Step,Time Elapsed,Total Timesteps,Episode Length Mean,Episode Reward Mean,FPS,Iterations,Approx KL,Clip Fraction,Clip Range,Entropy Loss,Explained Variance,Learning Rate,Loss,Policy Gradient Loss,Value Loss\n")
    #     def _on_step(self) -> bool:
    #         if self.n_calls % 1000 == 0:
    #             self.csv_file.write(f"{self.n_calls},{self.locals['time_elapsed']},{self.num_timesteps},{self.locals['rollout']['ep_len_mean']},{self.locals['rollout']['ep_rew_mean']},{self.locals['time']['fps']},{self.locals['time']['iterations']},{self.locals['train']['approx_kl']},{self.locals['train']['clip_fraction']},{self.locals['train']['clip_range']},{self.locals['train']['entropy_loss']},{self.locals['train']['explained_variance']},{self.locals['train']['learning_rate']},{self.locals['train']['loss']},{self.locals['train']['policy_gradient_loss']},{self.locals['train']['value_loss']}\n")
    #             self.csv_file.flush()
    #         return True

    #     def _on_training_end(self) -> None:
    #         self.csv_file.close()
    #         return True

    # custom_callback = CustomCallback()
    # model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./logs/ppo-mask-tensorboard/{version}")
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./logs/ppo-mask-tensorboard/{version}")
    # 训练代理
    try:
        model.learn(total_timesteps=10000000,callback=eval_callback)
        # 保存模型
        model.save(f"./models/{version}/model")
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model.")
        model.save(f"./models/{version}/model")