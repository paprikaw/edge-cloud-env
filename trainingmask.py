from tkinter import N
from stable_baselines3.common.callbacks import EvalCallback
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import DDPG
from sb3_contrib import MaskablePPO
from agents.a2c import A2C
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
from maskenv import MicroserviceMaskEnv 
from custom_callbacks import LatencyCallback
from dotenv import load_dotenv
import torch
import os
import logging
import signal
import time
load_dotenv(override=True)
logging.basicConfig(level=logging.ERROR)


parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--total_timesteps', type=int, default=5000000, help='Total timesteps for training')
parser.add_argument('--tag', type=str, default='complete-training', help='Tag for the training session')
parser.add_argument('--pattern', type=str, default='aggregator_sequential', help='Pattern to use')
parser.add_argument('--nodes', type=str, default='22', help='Number of nodes')
parser.add_argument('--pods', type=str, default='41', help='Number of pods')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda'], help='Device to use for training')
parser.add_argument('--cpu_num', type=int, default=31, help='Number of parallel environments (SubprocVecEnv)')
parser.add_argument('--n_steps', type=int, default=2048, help='Rollout steps per environment (PPO)')
parser.add_argument('--batch_size', type=int, default=64, help='Minibatch size for PPO updates')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs per PPO update')

args = parser.parse_args()
total_timesteps = args.total_timesteps
tag = args.tag
pattern = args.pattern
num_nodes = int(args.nodes) 
num_pods = int(args.pods)
device = args.device
step_panelty = 0.2
cpu_num = int(args.cpu_num)
n_steps = int(args.n_steps)
batch_size = int(args.batch_size)
n_epochs = int(args.n_epochs)

name = f"ppo-{num_pods}pods-{num_nodes}nodes-{pattern}-{tag}"
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
    print(f"device: {device} (cuda_available={torch.cuda.is_available()})")
    # Speed-focused defaults for modern NVIDIA GPUs
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    signal.signal(signal.SIGTERM, handle_terminate_signal)
    if cpu_num == 0:
        env = createEnv()
    else:
        env = SubprocVecEnv([make_env() for i in range(cpu_num)])

    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5, verbose=1)
    eval_callback = MaskableEvalCallback(
        env,
        best_model_save_path='./models/' + name,
        log_path='./logs/results/',       
        eval_freq=100000,                  
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        # callback_after_eval=stop_train_callback
    )
    latency_callback = LatencyCallback(repeat_target=10, num_nodes=num_nodes, num_pods=num_pods, pattern=pattern)
    global model
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./logs/ppo-mask-tensorboard/{name}",
        device=device,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=0.0001
    )
    # model = MaskablePPO("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./logs/ppo-mask-tensorboard/{name}")
    # 训练代理
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps,callback=[eval_callback, latency_callback])
    # 保存模型
    model.save(f"./models/{name}/model")

    # 在在训练的最后，输出训练的步数，时间
    print(f"{num_nodes} {num_pods} {pattern} ppo {model.num_timesteps} {(time.time() - start_time)/60}")