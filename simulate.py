from sb3_contrib import MaskablePPO
from maskenv import MicroserviceMaskEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from gymnasium.wrappers import FlattenObservation
import logging
from dotenv import load_dotenv
import os

version = f"v14/mask-ppo/dynamicenv-200-relative-400-acc-0.05"
logging.basicConfig(level=logging.WARNING)
env = MicroserviceMaskEnv(is_training=False, num_nodes=7, num_pods=13, dynamic_env=False, relative_para=20, accumulated_para=0.05)
model = MaskablePPO.load(f"./models/{version}/best_model.zip", env=env)
obs, info = env.reset()
done = False
logger = logging.getLogger(__name__)

for latency in [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]:
    print(f"latency: {latency}")
    env.set_cloud_latency(latency)
    env.set_cur_timestep(0)
    obs = env._get_state()
    done = False
    while not done:
        action_masks = env.action_masks()
        action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, done, _, info = env.step(action)
        env.render()