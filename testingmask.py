from sb3_contrib import MaskablePPO
from maskenv import MicroserviceMaskEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from gymnasium.wrappers import FlattenObservation
import logging
from dotenv import load_dotenv
import os

# version = f"v15/mask-ppo/dynamicenv-200-relative-400-acc-0.5"
version =f"old_mimic-partial-obs-step-1.25-state-less-final"
logging.basicConfig(level=logging.INFO)
env = MicroserviceMaskEnv(is_testing=True, num_nodes=7, num_pods=13, dynamic_env=False, step_panelty=1.25)
model = MaskablePPO.load(f"./models/{version}/best_model", env=env)
obs, info = env.reset()
done = False
logger = logging.getLogger(__name__)

while not done:
    action_masks = env.action_masks()
    action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
    obs, reward, done, _, info = env.step(action)
    logger.info(f"obs: {obs}")
    env.render()