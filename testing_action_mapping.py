from sb3_contrib import MaskablePPO
from env import MicroserviceEnv
from testbed_env import TestBedEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from gymnasium.wrappers import FlattenObservation
import logging

logging.basicConfig(level=logging.INFO)
env = MicroserviceEnv(is_training=False, num_nodes=5, num_pods=11)
testbed_env = TestBedEnv(num_nodes=5, num_pods=11)
model = MaskablePPO.load("./models/v5/best_model.zip", env=env)
# evaluate_policy(model, env, n_eval_episodes=1, reward_threshold=-100, warn=True)
obs, info = env.reset()
done = False

while not done:
    action_masks = env.action_masks()
    action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
    if action != env.stopped_action:
        node_name, pod_name = testbed_env.get_action(action)
        print(f"test bed action: {node_name} {pod_name}\n")
    obs, reward, done, _, info = env.step(action)
    env.render()