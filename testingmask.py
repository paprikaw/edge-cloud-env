from sb3_contrib import MaskablePPO
from maskenv import MicroserviceMaskEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from gymnasium.wrappers import FlattenObservation
import logging
# from trainingmask import version
# version = "v12-mask-a2c-latency/diffstepdiff-staticenv-200"
# version = "v12/mask-ppo/diffstepdiff-dynamicenv-200"
version = "v13/mask-ppo/dynamicenv-200"
logging.basicConfig(level=logging.WARNING)
env = MicroserviceMaskEnv(is_training=False, num_nodes=7, num_pods=13, dynamic_env=True)
model = MaskablePPO.load(f"./models/{version}/best_model.zip", env=env)
# model = MaskablePPO.load(f"./models/{version}/model.zip", env=env)
# model = MaskablePPO.load("./models/v11-mask-a2c/best_model.zip", env=env)
# evaluate_policy(model, env, n_eval_episodes=1, reward_threshold=-100, warn=True)
obs, info = env.reset()
done = False
logger = logging.getLogger(__name__)

while not done:
    action_masks = env.action_masks()
    action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
    obs, reward, done, _, info = env.step(action)
    env.render()
