from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from env import MicroserviceEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from custom_callbacks import NoMaskLatencyCallback
import logging
logging.basicConfig(level=logging.ERROR)
version = "dqn-less-state"
# version = "v11-no-mask-dynamic-ppo"
# version = "v11-no-mask-dynamic-a2c"

def createEnv():
    env = MicroserviceEnv(num_nodes=7, num_pods=13, dynamic_env=True, step_panelty=2, end_panelty=2)
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
    env = createEnv()
    latency_callback = NoMaskLatencyCallback(repeat_target=20, num_nodes=7, num_pods=13)
    eval_callback = EvalCallback(
        env,                       
        best_model_save_path='./models/' + version,
        log_path='./logs/results/',       
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=20,
    )
    model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./logs/ppo-mask-tensorboard/{version}")
    # model = A2C("MultiInputPolicy", env, verbose=1)
    # 训练代理
    try:
        model.learn(total_timesteps=10000000,callback=[eval_callback, latency_callback])
        # 保存模型
        model.save(f"./models/{version}/model")
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model.")
        model.save(f"./models/{version}/model")