from flask import Flask, request, jsonify
from testbed_env import TestBedEnv
from sb3_contrib import MaskablePPO
app = Flask(__name__)
env = TestBedEnv(5, 8)
model = MaskablePPO.load("./tmp_models2/best_model.zip", env=env)
# 定义一个 POST 端点，用于接收客户端发送的 JSON 数据
@app.route('/get_action', methods=['POST'])
def get_action():
    try:
        # 获取请求中的 JSON 数据
        data = request.get_json()

        # 解析 cluster_state 和 pod_deployable
        cluster_state = data.get("cluster_state", {})
        pod_deployable = data.get("pod_deployable", [])
        obs, info = env.reset(ClusterState=cluster_state, PodDeployable=pod_deployable)
        print(obs)
        # 构建伪数据响应 (PodName 和 TargetNode)
        response_data = []
        for pod in pod_deployable:
            response_data.append({
                "pod_name": pod["pod_name"],
                "target_node": "tb-edge-vm3"  # 伪数据: 假设所有 pod 都调度到 tb-edge-vm3
            })

        # 返回伪数据响应
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    # 启动 Flask 服务器，监听 5000 端口
    app.run(host='0.0.0.0', port=5000)
