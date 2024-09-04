from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib import MaskablePPO
from env import MicroserviceEnv
import logging
logging.basicConfig(level=logging.ERROR)

env = MicroserviceEnv()
eval_callback = MaskableEvalCallback(
    env,                       
    best_model_save_path='./models/',
    log_path='./logs/results/',       
    eval_freq=10000,                  
    deterministic=True,               
    render=False                      
)

model = MaskablePPO("MultiInputPolicy", env, verbose=1)
# 训练代理
try:
    model.learn(total_timesteps=200000,callback=eval_callback)
    # 保存模型
    model.save("maskppo")
except KeyboardInterrupt:
    print("Training interrupted. Saving the model.")
    model.save("maskppo")