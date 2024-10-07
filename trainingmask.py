from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import DDPG
from sb3_contrib import MaskablePPO
from agents.a2c import A2C
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from maskenv import MicroserviceMaskEnv 
from custom_callbacks import LatencyCallback
from dotenv import load_dotenv
import os
import logging
load_dotenv(override=False)
logging.basicConfig(level=logging.ERROR)
version = os.getenv("VERSION")
dynamic_latency = int(os.getenv("DYNAMIC_LATENCY"))
relative_para = int(os.getenv("RELATIVE_PARA"))
accumulated_para = float(os.getenv("ACCUMULATED_PARA"))
final_reward = float(os.getenv("FINAL_REWARD"))
cpu_num = int(os.getenv("NUM_CPU"))
# v14name = f"v{version}/mask-ppo/dynamicenv-{dynamic_latency}-relative-{relative_para}-acc-{accumulated_para}-final-{final_reward}"
# 15name = f"v{version}/mask-ppo/dynamicenv-relative-{relative_para}-layer-{50}"
name = f"v{version}/mask-ppo/dynamicenv-relative-{relative_para}-layer-{50}"
print(f"parameters: version: {version}, relative_para: {relative_para}, accumulated_para: {accumulated_para}, cpu: {cpu_num}, final_reward: {final_reward}")

def make_env():
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = MicroserviceMaskEnv(num_nodes=7, num_pods=13, dynamic_env=True, relative_para=relative_para, accumulated_para=accumulated_para, final_reward=final_reward)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    if cpu_num == 0:
        env = MicroserviceMaskEnv(num_nodes=7, num_pods=13, dynamic_env=True, relative_para=relative_para, accumulated_para=accumulated_para, final_reward=final_reward)
        env = Monitor(env)
    else:
        env = SubprocVecEnv([make_env() for i in range(cpu_num)])
    # env = MicroserviceMaskEnv(num_nodes=7, num_pods=13, dynamic_env=True, relative_para=relative_para, accumulated_para=accumulated_para)

    eval_callback = MaskableEvalCallback(
        env,
        best_model_save_path='./models/' + name,
        log_path='./logs/results/',       
        eval_freq=10000,                  
        deterministic=True,
        render=False,
        n_eval_episodes=50
    )
    latency_callback = LatencyCallback(repeat_target=10, num_nodes=7, num_pods=13, relative_para=relative_para, accumulated_para=accumulated_para, final_reward=final_reward)

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
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./logs/ppo-mask-tensorboard/{name}")
    # 训练代理
    try:
        model.learn(total_timesteps=8000000,callback=[eval_callback, latency_callback])
        # 保存模型
        model.save(f"./models/{name}/model")
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model.")
        model.save(f"./models/{name}/model")
    print(f"parameters: relative_para: {env.relative_para}, accumulated_para: {accumulated_para}")