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
model = MaskablePPO.load("./models/old_mimic-partial-obs-step-1.25-state-least-final/best_model.zip", env=env)
# model = DQN.load("./models/dqn-least-state/best_model", env=env)
logger = app.logger
# 创建日志处理器
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
# 设置日志格式，包括文件名和行号
formatter = logging.Formatter(
    '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# 将格式应用到处理器上
handler.setFormatter(formatter)
# 将处理器添加到 Flask 的 logger 中
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # 设置日志级别
isDQN = False
# 定义一个 POST 端点，用于接收客户端发送的 JSON 数据
@app.route('/get_action', methods=['POST'])
def get_action():
    # try:
    # 获取请求中的 JSON 数据
    data = request.get_json()
    # print(data)
    # 解析 cluster_state 和 pod_deployable
    cluster_state = data.get("cluster_state", {})
    pod_deployable = data.get("pod_deployable", [])
    # 创建logs目录（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger.debug(f"pod_deployable: {pod_deployable}")
    obs, info = env.reset(ClusterState=cluster_state, PodDeployable=pod_deployable)
    logger.debug(f"obs: {obs}")
    if isDQN:
        action, _states = model.predict(obs, deterministic=True)
    else:
        mask = env.action_masks()
        action, _states = model.predict(obs, deterministic=True, action_masks=mask)
    node, pod = "", ""
    if int(action) == env.stopped_action:
        logger.info(f"action: stop action")
    elif not env.check_valid_action(action):
        logger.info(f"action: invalid action")
    if not int(action) == env.stopped_action:
        logger.info("pod names" + str(env.pod_name))
        node, pod = env.get_action(action)
        logger.info(f"action: target node: {node}, target pod: {pod}")

    response_data = {
        "pod_name": pod,
        "target_node": node,
        "is_stop": int(action) == env.stopped_action or not env.check_valid_action(action)
    }

    return jsonify(response_data), 200

    # except Exception as e:
    #     logger.error(e)
    return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    # 启动 Flask 服务器，监听 5000 端口
    logger.info("server start")
    app.run(host='0.0.0.0', port=5000)