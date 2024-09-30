from env import MicroserviceEnv
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
import logging

logging.basicConfig(level=logging.INFO)
env = MicroserviceEnv(is_training=False, num_nodes=7, num_pods=13)
model = PPO.load("./models/v9-no-mask/best_model.zip", env=env)
# evaluate_policy(model, env, n_eval_episodes=1, reward_threshold=-100, warn=True)
obs, info = env.reset()
done = False
logger = logging.getLogger(__name__)

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    env.render()