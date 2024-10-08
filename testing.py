from env import MicroserviceEnv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from gymnasium.wrappers import FlattenObservation
import logging
from stable_baselines3.common.monitor import Monitor


logging.basicConfig(level=logging.INFO)
env = MicroserviceEnv(is_testing=True, num_nodes=7, num_pods=13, dynamic_env=False)
env = Monitor(env)
model = DQN.load("./models/dqn-less-state/best_model.zip", env=env)
# model = PPO.load("./models/v11-no-mask-dynamic-ppo/model.zip", env=env)
# model = A2C.load("./models/v11-no-mask-dynamic-a2c/model.zip", env=env)
# evaluate_policy(model, env, n_eval_episodes=1, reward_threshold=-100, warn=True)
obs, info = env.reset()
done = False
logger = logging.getLogger(__name__)

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)