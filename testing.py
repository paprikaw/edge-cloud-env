from sb3_contrib import MaskablePPO
from env import MicroserviceEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from gymnasium.wrappers import FlattenObservation
import logging

logging.basicConfig(level=logging.INFO)
env = MicroserviceEnv(is_training=False, num_nodes=5, num_pods=8)
model = MaskablePPO.load("./tmp_models/best_model.zip", env=env)
# evaluate_policy(model, env, n_eval_episodes=1, reward_threshold=-100, warn=True)
obs, info = env.reset()
done = False

while not done:
    action_masks = env.action_masks()
    action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
    obs, reward, done, _, info = env.step(action)
    env.render()