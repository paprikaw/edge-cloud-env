from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from env import MicroserviceEnv
import logging
logging.basicConfig(level=logging.ERROR)
version = "v12-no-mask-dynamic-latency-dqn"
# version = "v11-no-mask-dynamic-ppo"
# version = "v11-no-mask-dynamic-a2c"
env = MicroserviceEnv(num_nodes=7, num_pods=13, dynamic_env=True, is_training=True)
env = Monitor(env)
eval_callback = EvalCallback(
    env,                       
    best_model_save_path='./models/' + version,
    log_path='./logs/results/',       
    eval_freq=10000,                  
    deterministic=False,
    render=False,
    n_eval_episodes=20,
)

# model = PPO("MultiInputPolicy", env, verbose=1)
model = DQN("MultiInputPolicy", env, verbose=1)
# model = A2C("MultiInputPolicy", env, verbose=1)
# 训练代理
try:
    model.learn(total_timesteps=1000000,callback=eval_callback)
    # 保存模型
    model.save(f"./models/{version}/model")
except KeyboardInterrupt:
    print("Training interrupted. Saving the model.")
    model.save(f"./models/{version}/model")