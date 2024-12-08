from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from env import MicroserviceEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from custom_callbacks import NoMaskLatencyCallback
import argparse
import logging
logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--total_timesteps', type=int, default=5000000, help='Total timesteps for training')
parser.add_argument('--tag', type=str, default='complete-training', help='Tag for the training session')
parser.add_argument('--pattern', type=str, default='aggregator_sequential', help='Pattern to use')

args = parser.parse_args()
total_timesteps = args.total_timesteps
tag = args.tag
pattern = args.pattern
name = f"dqn-21pods-{pattern}-{tag}"

num_nodes = 7
num_pods = 21
num_cpu = 8
def createEnv():
    env = MicroserviceEnv(num_nodes=num_nodes, num_pods=num_pods, dynamic_env=True, step_panelty=2, end_panelty=2, pattern=pattern)
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
    # env = SubprocVecEnv([make_env() for i in range(8)])
    print(f"name: {name}")
    print(f"total_timesteps: {total_timesteps}")
    env = createEnv()
    latency_callback = NoMaskLatencyCallback(repeat_target=20, num_nodes=num_nodes, num_pods=num_pods, pattern=pattern)
    eval_callback = EvalCallback(
        env,                       
        best_model_save_path='./models/' + name,
        log_path='./logs/results/',       
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=20,
    )
    model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./logs/ppo-mask-tensorboard/{name}")
    # model = A2C("MultiInputPolicy", env, verbose=1)
    # 训练代理
    try:
        model.learn(total_timesteps=total_timesteps,callback=[eval_callback, latency_callback])
        # 保存模型
        model.save(f"./models/{name}/model")
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model.")
        model.save(f"./models/{name}/model")