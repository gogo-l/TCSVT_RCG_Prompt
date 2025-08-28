import torch


class MLP_Gate(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        MLP-Gate: Used to learn gating weights to control feature flow
        :param input_dim: input feature dimension F
        :param hidden_dim: hidden layer dimension H of the MLP
        """
        super(MLP_Gate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # F -> H
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # H -> 1
            nn.Sigmoid()  # output range [0,1]
        )

    def forward(self, x):
        """
        :param x: input features (B, M, F)
        :return: gated features (B, M, F)
        """
        B, M, F = x.shape
        g = self.mlp(x)  # compute gating weights (B, M, 1)
        x_out = g * x  # element-wise multiplication (broadcast mechanism)
        return x_out, g.squeeze(-1)  # return new features and gating weights


def compute_feature_variance(features):
    """
    Compute the variance of input features to measure task difficulty
    :param features: Tensor, shape (batch_size, feature_dim)
    :return: Tensor, task difficulty score
    """
    # compute variance for each feature dimension
    feature_variance = torch.var(features, dim=0, unbiased=True)
    # take mean as an overall variance metric
    difficulty_score = torch.mean(feature_variance)
    return difficulty_score


import torch
import torch.nn.functional as F

def compute_attention_entropy(attention_matrix):
    """
    Compute the entropy of the attention distribution to measure task complexity
    :param attention_matrix: Tensor, shape (batch_size, num_heads, seq_len, seq_len)
    :return: Tensor, task difficulty score
    """
    # compute entropy for each batch and each head
    entropy = -torch.sum(attention_matrix * torch.log(attention_matrix + 1e-8), dim=-1)
    # take mean as global entropy
    difficulty_score = torch.mean(entropy)
    return difficulty_score


def compute_task_difficulty(features, attention_matrix, alpha=0.5):
    """
    Combine feature variance and attention entropy to compute task difficulty
    :param features: Tensor, (batch_size, feature_dim)
    :param attention_matrix: Tensor, (batch_size, num_heads, seq_len, seq_len)
    :param alpha: float, weight parameter controlling the influence of two metrics
    :return: Tensor, task difficulty score
    """
    variance_score = compute_feature_variance(features)
    entropy_score = compute_attention_entropy(attention_matrix)

    # linear weighted sum
    difficulty_score = alpha * variance_score + (1 - alpha) * entropy_score
    return difficulty_score


def compute_gate_weights(features, attention_matrix, mlp_gate, beta=1.0):
    """
    Compute gating weights g
    :param features: Tensor, (batch_size, feature_dim)
    :param attention_matrix: Tensor, (batch_size, num_heads, seq_len, seq_len)
    :param mlp_gate: MLP model for generating gating weights
    :param beta: float, scaling factor controlling the influence of task difficulty on gating
    :return: Tensor, gating weights (batch_size, ku)
    """
    # compute task difficulty
    difficulty_score = compute_task_difficulty(features, attention_matrix)

    # compute dynamic threshold tau
    tau = torch.sigmoid(beta * difficulty_score)

    # compute gating weights
    g = torch.sigmoid(mlp_gate(features))

    # binarize
    g_binary = (g >= tau).float()

    return g_binary
