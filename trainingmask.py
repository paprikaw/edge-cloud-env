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
import signal
load_dotenv(override=True)
logging.basicConfig(level=logging.ERROR)
step_panelty = 1.25
cpu_num = 8
total_timesteps = 5000000
name = f"ppo-leaststate-morepods-chain"
num_pods = 21
num_nodes = 7
pattern = "chain"
def handle_terminate_signal(signum, frame):
    print("Terminate signal received. Saving the model.")
    model.save(f"./models/{name}/model")
    exit(0)

def createEnv():
    env = MicroserviceMaskEnv(num_nodes=num_nodes, num_pods=num_pods, dynamic_env=True, is_testing=False, step_panelty=step_panelty, pattern=pattern)
    env = Monitor(env)
    return env

def make_env():
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = createEnv()
        return env
    return _init

if __name__ == "__main__":
    print(f"step_panelty: {step_panelty}")
    print(f"name: {name}")
    print(f"total_timesteps: {total_timesteps}")
    signal.signal(signal.SIGTERM, handle_terminate_signal)
    if cpu_num == 0:
        env = createEnv()
    else:
        env = SubprocVecEnv([make_env() for i in range(cpu_num)])

    eval_callback = MaskableEvalCallback(
        env,
        best_model_save_path='./models/' + name,
        log_path='./logs/results/',       
        eval_freq=10000,                  
        deterministic=True,
        render=False,
        n_eval_episodes=50
    )
    latency_callback = LatencyCallback(repeat_target=10, num_nodes=num_nodes, num_pods=num_pods, pattern=pattern)
    global model
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./logs/ppo-mask-tensorboard/{name}")
    # model = MaskablePPO("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./logs/ppo-mask-tensorboard/{name}")
    # 训练代理
    model.learn(total_timesteps=total_timesteps,callback=[eval_callback, latency_callback])
    # 保存模型
    model.save(f"./models/{name}/model")