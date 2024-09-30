from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from maskenv import MicroserviceMaskEnv
import logging
logging.basicConfig(level=logging.ERROR)
version = "v10-best-model"

env = MicroserviceMaskEnv(num_nodes=7, num_pods=13, isMask=False)
env = Monitor(env)
eval_callback = MaskableEvalCallback(
    env,                       
    best_model_save_path='./models/' + version,
    log_path='./logs/results/',       
    eval_freq=10000,                  
    deterministic=True,
    render=False                      
)

model = MaskablePPO("MultiInputPolicy", env, verbose=1)
# 训练代理
try:
    model.learn(total_timesteps=500000,callback=eval_callback)
    # 保存模型
    model.save(f"./models/{version}/maskppo")
except KeyboardInterrupt:
    print("Training interrupted. Saving the model.")
    model.save(f"./models/{version}/maskppo")