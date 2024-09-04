import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

# 修改的 GCNEncoder 类，用于处理边特征
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 修改的 DGI 类，用于处理边特征
class DGI(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(DGI, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels)
        self.summary = torch.nn.Linear(hidden_channels, hidden_channels)
        self.act = torch.nn.PReLU()

    def forward(self, x, edge_index, batch):
        h = self.encoder(x, edge_index)
        g = global_mean_pool(h, batch)  # Global pooling
        summary = self.summary(g)
        return summary, h  # 返回全局图嵌入和所有节点嵌入

    def get_node_graph_embedding(self, node_index, x, edge_index, batch):
        summary, h = self.forward(x, edge_index, batch)
        # 选择特定节点的嵌入
        node_embedding = h[node_index]
        # 将 summary 从 2D (1, hidden_channels) 转为 1D (hidden_channels)
        summary = summary.squeeze(0)
        # 将节点嵌入和全局图嵌入拼接
        node_graph_embedding = torch.cat([node_embedding, summary], dim=0)
        return node_graph_embedding

# 用法示例

# 定义图的数据
# 节点特征矩阵 x
x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
# 边索引矩阵 edge_index
edge_index = torch.tensor([[0, 1, 2],
                           [1, 2, 0]], dtype=torch.long)
# 边特征矩阵 edge_attr，每条边有 2 个特征
edge_attr = torch.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]], dtype=torch.float)

# 在此示例中，我们只有一个图，因此批次信息是一个全零的张量
batch = torch.tensor([0, 0, 0], dtype=torch.long)

# 创建 DGI 模型实例
in_channels = x.size(1)           # 输入特征的维度
hidden_channels = 5             # 隐藏层的维度
model = DGI(in_channels, hidden_channels)

# 前向传播，得到图的嵌入表示
summary, h = model(x, edge_index, batch)

# 打印图的嵌入表示
print(f"Graph embedding: {summary}")

# 获取特定节点相对于图的嵌入表示
node_index = 0  # 选择节点 0
node_graph_embedding = model.get_node_graph_embedding(node_index, x, edge_index, batch)

# 打印特定节点相对于图的嵌入表示
print(f"Node {node_index} relative to graph embedding: {node_graph_embedding}")
