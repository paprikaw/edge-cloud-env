from sb3_contrib import MaskablePPO
from maskenv import MicroserviceMaskEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from gymnasium.wrappers import FlattenObservation
import logging
from dotenv import load_dotenv
import os
import time
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--pods", type=int, default=21, help="pod数量")
# parser.add_argument("--nodes", type=int, default=7, help="节点数量")
# args = parser.parse_args()

# pod_num = args.pods
# node_num = args.nodes
# pod_nums = [21, 25, 29, 33, 37]
# node_nums = [7, 10, 13, 15, 19]

total_length = 1000
pod_nums = [21, 25, 29, 33, 37]
node_nums = [7, 10, 13, 15, 19]
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

result = []
for i in range(5):
    pod_num = pod_nums[i]
    node_num = node_nums[i]
    version =f"ppo-{pod_num}pods-{node_num}nodes-aggregator_sequential-scalability"

    env = MicroserviceMaskEnv(is_testing=True, num_nodes=node_num, num_pods=pod_num, dynamic_env=False, step_panelty=1.25, replica_cnt=1, pattern="aggregator_sequential")
    model = MaskablePPO.load(f"./models/{version}/best_model", env=env)
    obs, info = env.reset()
    done = False

    avg_inference_time = 0
    for i in range(total_length):
        action_masks = env.action_masks()
        start_time = time.time()
        action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
        end_time = time.time()
        inference_time = end_time - start_time
        avg_inference_time += inference_time
        obs, reward, done, _, info = env.step(action)
        logger.info(f"obs: {obs}")
        env.render()
        if done:
            obs, info = env.reset()
    avg_inference_time /= total_length
    result.append(avg_inference_time)
for inference_time, i in zip(result, range(5)):
    print(f"pod_num: {pod_nums[i]}, node_num: {node_nums[i]}, avg_inference_time: {inference_time}")