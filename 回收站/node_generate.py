import json
import random

class NodeSetup:
    def __init__(self, config_path='config.json'):
        self.config = self._load_config(config_path)
        self.node_types = self.config['nodeTypes']
        self.node_templates = self.config['nodes']
        self.cluster_setup = self.config['cluster_setup']
        self.cluster = self._initialize_cluster()

    def _load_config(self, path):
        with open(path, 'r') as json_file:
            return json.load(json_file)

    def _initialize_cluster(self):
        cluster = []
        self._add_nodes(cluster, 'cloud')
        self._add_nodes(cluster, 'edge')
        return cluster

    def _add_nodes(self, cluster, node_type):
        setup = self.cluster_setup[node_type]
        node_count = random.randint(*setup['node_count_range'])
        for i in range(node_count):
            node_template_name = random.choice(setup['node'])
            node_template = self.node_templates[node_template_name]
            node_id = f"{node_type}Node{i+1}"
            node_type_config = self.node_types[node_template['nodeType']]
            node = {
                "id": node_id,
                "nodeType": node_template['nodeType'],
                "isCloud": node_template['isCloud'],
                "cpu_availability": self._get_value(node_template, setup, 'cpu_availability'),
                "memory_availability": self._get_value(node_template, setup, 'memory_availability'),
                "bandwidth": node_template.get('bandwidth', random.choice(node_type_config['bandwidth'])),
                "bandwidth_utilization": f"{random.randint(1, 100)}%"
            }
            cluster.append(node)

    def _get_value(self, node_template, setup, key):
        if key in node_template:
            return random.choice(node_template[key])
        else:
            return random.choice(setup[key])

    def get_cluster(self):
        return self.cluster

# 运行示例
config_path = 'nodes1.json'  # 配置文件路径
node_setup = NodeSetup(config_path)
cluster = node_setup.get_cluster()

# 打印生成的节点集群
print(json.dumps(cluster, indent=2))
