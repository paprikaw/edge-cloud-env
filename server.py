from flask import Flask, request, jsonify
from testbed_env import TestBedEnv
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN
from maskenv import MicroserviceMaskEnv
import json
import os
from datetime import datetime
import logging
app = Flask(__name__)
env = TestBedEnv(7, 13, 50)
# simuEnv = MicroserviceMaskEnv(is_training=False, num_nodes=7, num_pods=13)
# simuEnv.reset()
# model = DQN.load("./models/final/ppo.zip", env=env)
model = MaskablePPO.load("./models/final/ppo.zip", env=env)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 确保logger级别设置为DEBUG
# 定义一个 POST 端点，用于接收客户端发送的 JSON 数据
@app.route('/get_action', methods=['POST'])
def get_action():
    try:
        # 获取请求中的 JSON 数据
        data = request.get_json()
        # print(data)
        # 解析 cluster_state 和 pod_deployable
        cluster_state = data.get("cluster_state", {})
        pod_deployable = data.get("pod_deployable", [])

        # 创建logs目录（如果不存在）
        if not os.path.exists('logs'):
            os.makedirs('logs')
        logger.info(f"pod_deployable: {pod_deployable}")
        obs, info = env.reset(ClusterState=cluster_state, PodDeployable=pod_deployable)
        logger.debug(f"obs: {obs}")
        mask = env.action_masks()
        action, _states = model.predict(obs, deterministic=True, action_masks=mask)
        node, pod = "", ""
        if int(action) != env.stopped_action:
            node, pod = env.get_action(action)
            logger.info(f"action: target node: {node}, target pod: {pod}")
            # logger.info(f"simu action: {simuEnv.get_action_name(action)}")
        else:
            logger.info(f"action: stop")

        response_data = {
            "pod_name": pod,
            "target_node": node,
            "is_stop": int(action) == env.stopped_action
        }

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(e)
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    # 启动 Flask 服务器，监听 5000 端口
    logger.info("server start")
    app.run(host='0.0.0.0', port=5000)
