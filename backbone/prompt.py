import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class DynamicPromptGate(nn.Module):
    def __init__(self, input_dim, max_prompt_length):
        """
        动态Prompt门控网络
        :param input_dim: 输入特征维度 (F)
        :param max_prompt_length: 最大Prompt长度 (ku)
        """
        super(DynamicPromptGate, self).__init__()
        self.ku = max_prompt_length
        # 生成ku个门控权重
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.ku),  # F -> ku
            nn.Sigmoid()  # 输出范围 [0,1]
        )

    def forward(self, features):
        """
        :param features: 输入特征 (B, F)
        :return: 门控权重 (B, ku)
        """
        g = self.mlp(features)  # (B, ku)
        return g


def compute_gate_weights(features, attention_matrix, gate_network, beta=1.0):
    """
    计算动态门控权重
    :param features: 输入特征 (B, F)
    :param attention_matrix: 注意力矩阵 (B, H, S, S)
    :param gate_network: DynamicPromptGate实例
    :param beta: 阈值缩放因子
    :return: 二值化门控权重 (B, ku)
    """
    # 1. 计算任务难度得分 (B,)
    difficulty_score = compute_task_difficulty(features, attention_matrix)  # (B,)

    # 2. 生成基础门控权重 (B, ku)
    g = gate_network(features)  # (B, ku)

    # 3. 动态阈值计算
    tau = torch.sigmoid(beta * difficulty_score)  # (B,)

    # 4. 二值化门控权重
    g_binary = (g >= tau.unsqueeze(-1)).float()  # (B, ku)

    return g_binary


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
        #B, M, F = x.shape
        g = self.mlp(x)  # 计算门控权重 (B, M, 1)
        x_out = g * x  # 逐元素乘法 (广播机制)
        return g.squeeze(-1)  # 返回新的特征 和 门控权重


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


# Instance whitening
class InstanceWhitening(nn.Module):

    def __init__(self, dim):
        super(InstanceWhitening, self).__init__()
        self.instance_standardization = nn.InstanceNorm2d(dim, affine=False)

    def forward(self, x):
        x = self.instance_standardization(x)
        w = x
        return x, w


# Semantic concept modeling for VIT
class SCMModule(nn.Module):

    def __init__(self,
                 outs: dict,
                 fine_num_classes: int,
                 coarse_num_classes: int,
                 mid_feature_num: int):

        super(SCMModule, self).__init__()

        if len(outs['layer4'].shape) == 4:
            input_feature_num = outs['layer4'].shape[-1] * outs['layer4'].shape[-2]
        elif len(outs['layer4'].shape) == 3:
            input_feature_num = outs['layer4'].shape[1]
        self.branch1_linear1 = nn.Sequential(nn.Linear(input_feature_num, mid_feature_num), nn.ReLU())  # swin-144
        self.branch1_linear2 = nn.Linear(mid_feature_num, coarse_num_classes)
        self.branch1_iw = InstanceWhitening(coarse_num_classes)

        self.branch2_linear1 = nn.Sequential(nn.Linear(input_feature_num, mid_feature_num), nn.ReLU())  # swin-144
        self.branch2_linear2 = nn.Linear(mid_feature_num, fine_num_classes)
        self.branch21_linear = nn.Linear(fine_num_classes, coarse_num_classes)
        self.branch21_iw = InstanceWhitening(coarse_num_classes)
        self.branch22_linear = nn.Linear(fine_num_classes, fine_num_classes)
        self.branch22_iw = InstanceWhitening(fine_num_classes)
        self.constraint = nn.MSELoss()

    def forward(self, x):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.view((B, C, H * W))
        else:
            x = x.transpose(1, 2).contiguous()
        branch1 = self.branch1_linear1(x)
        branch1 = self.branch1_linear2(branch1)
        branch1 = branch1.transpose(1, 2).contiguous()
        branch1 = branch1.unsqueeze(3)
        branch1, _ = self.branch1_iw(branch1)
        branch1 = branch1.squeeze(3)
        branch2 = self.branch2_linear1(x)
        branch2 = self.branch2_linear2(branch2)
        branch21 = self.branch21_linear(branch2)
        branch21 = branch21.transpose(1, 2).contiguous()
        branch21 = branch21.unsqueeze(3)
        branch21, _ = self.branch21_iw(branch21)
        branch21 = branch21.squeeze(3)  #共享特征
        branch22 = self.branch22_linear(branch2)
        branch22 = branch22.transpose(1, 2).contiguous()
        branch22 = branch22.unsqueeze(3)
        output_e, _ = self.branch22_iw(branch22)
        output_e = output_e.squeeze(3)
        constraint = self.constraint(branch1, branch21)
        branch21_expanded = branch21.unsqueeze(2)  # 形状 [16, n, 1, 768]
        output_expanded = output_e.unsqueeze(1)  # 形状 [16, 1, k, 768]
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(branch21_expanded, output_expanded, dim=-1)  # 形状 [16, n, k]
        # 计算损失（例如最小化相似度）
        min_cosine_loss = cosine_sim.mean()  # 对所有样本和维度取平均
        return branch21, output_e, 0.5 * constraint + 0.5 * min_cosine_loss


# Semantic concept embedding
#coda_prompt
class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        #self.gate = MLP_Gate(768, 16)
        self.promptgate = DynamicPromptGate(768, self.e_p_length)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full parameters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = self.tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = self.tensor_prompt(self.e_pool_size, self.key_d)
            a = self.tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0, 1, 2, 3, 4]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]

    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self, f'e_k_{e}')
            A = getattr(self, f'e_a_{e}')
            P = getattr(self, f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def forward(self, x_querry, l, x_block, train=False):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self, f'e_k_{l}')
            A = getattr(self, f'e_a_{l}')
            p = getattr(self, f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)

            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            #处理共享特征，选取最大响应值
            q_s = torch.norm(self.output_share, p=2, dim=-1)
            max_index_q_s = torch.argmax(q_s, dim=-1)
            q_s = self.output_share[torch.arange(B), max_index_q_s]
            x_querry_s = x_querry + q_s  # 添加共享特征
            a_querry = torch.einsum('bd,kd->bkd', x_querry_s, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_s = torch.einsum('bk,kld->bld', aq_k, p)

            # with attention and cosine sim

            #处理独立特征，选取最大响应值
            q_e = torch.norm(self.output_esp, p=2, dim=-1)
            max_index_q_s = torch.argmax(q_e, dim=-1)
            q_e = self.output_esp[torch.arange(B), max_index_q_s]
            x_querry_e = x_querry + q_e  # 添加共享特征
            a_querry = torch.einsum('bd,kd->bkd', x_querry_e, A)
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            P_e = torch.einsum('bk,kld->bld', aq_k, p)
            # select prompts
            i = int(self.e_p_length / 2)

            Ek = P_s
            Ev = P_e

            g_binary = compute_gate_weights(x_querry, self.attenmatrix, self.promptgate)
            # if torch.any(g_binary>0):
            #     print("大于0000")
            Ev = Ev * g_binary.unsqueeze(-1)

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = self.ortho_penalty(K) * self.ortho_mu
                loss += self.ortho_penalty(A) * self.ortho_mu
                loss += self.ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

    def ortho_penalty(self, t):
        return ((t @ t.T - torch.eye(t.shape[0])) ** 2).mean()

    def tensor_prompt(self, a, b, c=None, ortho=False):
        if c is None:
            p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
        else:
            p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
        if ortho:
            nn.init.orthogonal_(p)
        else:
            nn.init.uniform_(p)
        return p

    def concate(self, feature_dict, inc_class, task, attenmatrix):
        n=0
        if task == 1:
            n = 1
            # 只有当 task 发生变化时，才执行这两行代码
            self.scm_module = SCMModule(feature_dict, inc_class, 1, 64).to(device='cuda')
            #self.sce_module = SCEModule().to(device='cuda')
        self.attenmatrix = attenmatrix
        self.output_share, self.output_esp, self.constraint1 = self.scm_module(feature_dict['layer4'])
        # print(self.constraint1)
        # print("cons loss")
        #output2 = self.sce_module(feature_dict, self.output_esp)
        return self.output_esp, n, self.constraint1


# dual prompt
class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False, ):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value

        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(
                            prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
            else:
                prompt_pool_shape = (self.num_layers, self.pool_size, self.length, embed_dim)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_key_norm = self.l2_normalize(self.prompt_key, dim=-1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            similarity = similarity.t()  # B, pool_size

            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1)  # B, top_k
            out['similarity'] = similarity

            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id,
                                           torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()),
                                                      device=prompt_id.device)])
                    id_counts = torch.cat(
                        [id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k
                major_prompt_id = prompt_id[major_idx]  # top_k
                # expand to batch
                idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous()  # B, top_k

            if prompt_mask is not None:
                idx = prompt_mask  # B, top_k

            out['prompt_idx'] = idx
            if self.use_prefix_tune_for_e_prompt:
                batched_prompt_raw = self.prompt[:, :, idx]  # num_layers, B, top_k, length, C
                num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                )
            else:
                batched_prompt_raw = self.prompt[:, idx]
                num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, top_k * length, embed_dim
                )

            batched_key_norm = prompt_key_norm[idx]  # B, top_k, C

            out['selected_key'] = batched_key_norm
            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            # Put pull_constraint loss calculation inside
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(
                            torch.randn(prompt_pool_shape))  # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)

        out['batched_prompt'] = batched_prompt

        return out


# l2p prompt
class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform', ):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size

            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1)  # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],),
                                                                     torch.min(idx.flatten()),
                                                                     device=prompt_id.device)])
                        id_counts = torch.cat(
                            [id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k
                    major_prompt_id = prompt_id[major_idx]  # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1)  # B, top_k
            else:
                idx = prompt_mask  # B, top_k

            batched_prompt_raw = self.prompt[idx]  # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)  # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx]  # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out
