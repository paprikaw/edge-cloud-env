from sb3_contrib import MaskablePPO
from maskenv import MicroserviceMaskEnv
import argparse
import json
import logging
import os
import re
import myparser


def resolve_model_path(models_dir: str) -> str:
    """根据提供的 models 子目录，解析 best_model 路径（兼容 .zip/无后缀）。"""
    base = os.path.join("./models", models_dir, "best_model")
    if os.path.exists(base + ".zip"):
        return base + ".zip"
    return base


def list_config_dirs(root: str = "./config") -> list:
    """枚举形如 <nodes>-<pods> 的配置目录，返回 [(nodes, pods, path), ...]。"""
    result = []
    if not os.path.isdir(root):
        return result
    for name in os.listdir(root):
        if re.fullmatch(r"\d+-\d+", name):
            nodes_str, pods_str = name.split("-")
            try:
                nodes = int(nodes_str)
                pods = int(pods_str)
                result.append((nodes, pods, os.path.join(root, name)))
            except ValueError:
                continue
    # 按节点/Pod 升序
    result.sort(key=lambda x: (x[0], x[1]))
    return result


def extract_cloud_latency_samples(node_cfg: dict) -> list:
    """
    从 nodes.json 的 latency 段落中，抽取与 cloud 相关的范围，
    为每个范围生成 [low, mid, high] 的样本集合，并去重排序。
    """
    lat_section = node_cfg.get("latency", {})
    samples = []
    for layer, targets in lat_section.items():
        if not isinstance(targets, dict):
            continue
        for target, range_val in targets.items():
            if "cloud" not in (layer, target):
                continue
            if isinstance(range_val, list) and len(range_val) == 2:
                try:
                    low = float(myparser.parse_time(range_val[0]))
                    high = float(myparser.parse_time(range_val[1]))
                except Exception:
                    continue
                if high < low:
                    low, high = high, low
                mid = (low + high) / 2.0
                samples.extend([low, mid, high])
    # 去重并排序
    uniq = sorted({round(x, 6) for x in samples})
    # 若没有解析到，给出一个回退样本
    return uniq if uniq else [50.0]


def main():
    parser = argparse.ArgumentParser(description="Auto-load multiple configs under ./config and simulate with their latency settings")
    parser.add_argument("--configs", type=str, nargs="*", default=[], help="仅测试指定目录名（如 7-21 22-41），不填则自动遍历全部")
    parser.add_argument("--dynamic_env", action="store_true", help="使用 nodes.json（动态环境）；不加则使用 nodes-simple.json")
    parser.add_argument("--pattern", type=str, default="aggregator_sequential", help="调用模式（用于选择调用图配置）")
    parser.add_argument("--replica_cnt", type=int, default=-1, help="每服务副本数上限（传 -1 使用配置默认）")
    parser.add_argument("--repeat", type=int, default=10, help="每个延迟样本重复实验次数")
    parser.add_argument(
        "--models_tpl",
        type=str,
        default="ppo-{pods}pods-{nodes}nodes-aggregator_sequential-gpu",
        help="模型目录模板，支持 {nodes} 与 {pods} 占位符",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

    # 组装待测试的配置目录
    all_cfgs = list_config_dirs("./config")
    if args.configs:
        allow = set(args.configs)
        all_cfgs = [item for item in all_cfgs if os.path.basename(item[2]) in allow]
    if not all_cfgs:
        print("No config directories found under ./config matching <nodes>-<pods>.")
        return

    for nodes, pods, cfg_dir in all_cfgs:
        node_file = os.path.join(cfg_dir, "nodes.json" if args.dynamic_env else "nodes-simple.json")
        svc_file = os.path.join(cfg_dir, "services.json")
        if not os.path.exists(node_file) or not os.path.exists(svc_file):
            print(f"Skip {os.path.basename(cfg_dir)} (missing nodes/services json)")
            continue

        # 提取该配置下的 cloud 相关延迟样本
        try:
            with open(node_file, "r") as f:
                node_cfg = json.load(f)
        except Exception as e:
            print(f"Skip {os.path.basename(cfg_dir)} (invalid nodes json): {e}")
            continue

        latencies = extract_cloud_latency_samples(node_cfg)
        print(f"\n===== Config {os.path.basename(cfg_dir)} | nodes={nodes}, pods={pods} | samples={latencies} =====")

        # 为该配置初始化环境与模型
        env = MicroserviceMaskEnv(
            is_testing=True,
            num_nodes=nodes,
            num_pods=pods,
            dynamic_env=args.dynamic_env,
            step_panelty=1,
            pattern=args.pattern,
            replica_cnt=args.replica_cnt,
        )

        models_dir = args.models_tpl.format(nodes=nodes, pods=pods)
        model_path = resolve_model_path(models_dir)
        if not os.path.exists(model_path):
            print(f"Model not found for {models_dir}, expected {model_path}. Skipping.")
            continue
        model = MaskablePPO.load(model_path, env=env)

        for latency in latencies:
            print(f"\n>>> [{os.path.basename(cfg_dir)}] Set cloud latency: {latency}")
            for rep in range(1, args.repeat + 1):
                print(f"--- Repeat {rep}/{args.repeat} ---")
                obs, info = env.reset()
                env.set_cloud_latency(latency)
                env.set_cur_timestep(0)
                obs = env._get_state()
                done = False
                while not done:
                    action_masks = env.action_masks()
                    action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
                    before = env.latency_func()
                    obs, reward, done, _, info = env.step(action)
                    after = env.latency_func() if not done else before
                    print(f"Latency change: {before} -> {after}")
                    env.render()


if __name__ == "__main__":
    main()