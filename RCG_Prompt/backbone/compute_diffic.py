import torch


class MLP_Gate(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        MLP-Gate: 用于学习门控权重，控制特征流动
        :param input_dim: 输入特征维度 F
        :param hidden_dim: MLP 隐藏层维度 H
        """
        super(MLP_Gate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # F -> H
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # H -> 1
            nn.Sigmoid()  # 输出范围 [0,1]
        )

    def forward(self, x):
        """
        :param x: 输入特征 (B, M, F)
        :return: 经过门控的特征 (B, M, F)
        """
        B, M, F = x.shape
        g = self.mlp(x)  # 计算门控权重 (B, M, 1)
        x_out = g * x  # 逐元素乘法 (广播机制)
        return x_out, g.squeeze(-1)  # 返回新的特征 和 门控权重
def compute_feature_variance(features):
    """
    计算输入特征的方差，用于衡量任务难度
    :param features: Tensor, 形状为 (batch_size, feature_dim)
    :return: Tensor, 任务难度分数
    """
    # 计算每个特征维度的方差
    feature_variance = torch.var(features, dim=0, unbiased=True)
    # 取平均值作为整体方差度量
    difficulty_score = torch.mean(feature_variance)
    return difficulty_score


import torch
import torch.nn.functional as F

def compute_attention_entropy(attention_matrix):
    """
    计算注意力分布的熵，用于衡量任务复杂度
    :param attention_matrix: Tensor, 形状为 (batch_size, num_heads, seq_len, seq_len)
    :return: Tensor, 任务难度分数
    """
    # 计算每个 batch、每个头的熵
    entropy = -torch.sum(attention_matrix * torch.log(attention_matrix + 1e-8), dim=-1)
    # 取平均值作为全局熵
    difficulty_score = torch.mean(entropy)
    return difficulty_score


def compute_task_difficulty(features, attention_matrix, alpha=0.5):
    """
    结合特征方差和注意力熵计算任务难度
    :param features: Tensor, (batch_size, feature_dim)
    :param attention_matrix: Tensor, (batch_size, num_heads, seq_len, seq_len)
    :param alpha: float, 权重参数，控制两种度量的影响
    :return: Tensor, 任务难度得分
    """
    variance_score = compute_feature_variance(features)
    entropy_score = compute_attention_entropy(attention_matrix)

    # 线性加权求和
    difficulty_score = alpha * variance_score + (1 - alpha) * entropy_score
    return difficulty_score


def compute_gate_weights(features, attention_matrix, mlp_gate, beta=1.0):
    """
    计算门控权重 g
    :param features: Tensor, (batch_size, feature_dim)
    :param attention_matrix: Tensor, (batch_size, num_heads, seq_len, seq_len)
    :param mlp_gate: MLP 模型，用于生成门控权重
    :param beta: float, 调节任务难度对门控的影响
    :return: Tensor, 门控权重 (batch_size, ku)
    """
    # 计算任务难度
    difficulty_score = compute_task_difficulty(features, attention_matrix)

    # 计算动态阈值 tau
    tau = torch.sigmoid(beta * difficulty_score)

    # 计算门控权重
    g = torch.sigmoid(mlp_gate(features))

    # 进行二值化处理
    g_binary = (g >= tau).float()

    return g_binary