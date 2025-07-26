import math
import torch
import torch.nn as nn
from torch.nn import init
import time
import torch.nn.functional as F
from models.expert.layers.Embedding import *
# from models.expert.models.Mamba import MambaLayer
from models.expert.models.Mamba import MambaLayer


class Transformer_Layer(nn.Module):
    def __init__(self, device, d_model, d_ff, num_nodes, patch_nums, patch_size, dynamic, factorized, layer_number,
                 batch_norm, args=None):
        super(Transformer_Layer, self).__init__()
        self.device = device  # 设备信息（CPU或GPU）
        self.d_model = d_model  # 模型的嵌入维度
        self.num_nodes = num_nodes  # 节点数（用于处理多变量序列）
        self.dynamic = dynamic  # 是否使用动态机制
        self.patch_nums = patch_nums  # patch 数量（时间序列分块数）
        self.patch_size = patch_size  # patch 大小（每个时间块包含的时间步数）
        self.layer_number = layer_number  # Transformer 层数
        self.batch_norm = batch_norm  # 是否使用批量归一化
        self.args = args
        ## 局部注意力（Intra-Patch Attention）的嵌入初始化
        self.intra_embeddings = nn.Parameter(torch.rand(self.patch_nums, 1, 1, self.num_nodes, 16),
                                             requires_grad=True)
        # 每个 patch 初始化随机嵌入
        self.embeddings_generator = nn.ModuleList([nn.Sequential(*[
            nn.Linear(16, self.d_model)]) for _ in range(self.patch_nums)])
        # 嵌入生成器，作用是将嵌入映射到 d_model 维度
        self.intra_d_model = self.d_model  # 局部注意力的维度与模型的嵌入维度相同
        self.intra_patch_attention = Intra_Patch_Attention(self.intra_d_model, factorized=factorized)
        # 初始化局部注意力机制
        self.weights_generator_distinct = WeightGenerator(self.intra_d_model, self.intra_d_model, mem_dim=16,
                                                          num_nodes=num_nodes,
                                                          factorized=factorized, number_of_weights=2)
        # 为 distinct 权重生成器创建一个生成器
        self.weights_generator_shared = WeightGenerator(self.intra_d_model, self.intra_d_model, mem_dim=None,
                                                        num_nodes=num_nodes,
                                                        factorized=False, number_of_weights=2)
        # 为 shared 权重生成器创建一个生成器
        self.intra_Linear = nn.Linear(self.patch_nums, self.patch_nums * self.patch_size)
        # 用于调整局部注意力输出的线性层

        ## 全局注意力（Inter-Patch Attention）
        self.stride = patch_size  # patch 大小决定了滑动窗口的步长
        self.inter_d_model = self.d_model * self.patch_size  # 全局注意力的维度是局部注意力维度与 patch 大小的乘积
        self.mambaLayer = MambaLayer(dim=self.inter_d_model)

        ## 全局注意力的嵌入初始化
        self.emb_linear = nn.Linear(self.inter_d_model, self.inter_d_model)
        # 线性层，用于处理全局嵌入
        self.W_pos = positional_encoding(pe='zeros', learn_pe=True, q_len=self.patch_nums, d_model=self.inter_d_model)
        # 初始化位置编码，q_len 是 patch 数量，d_model 是嵌入维度

        n_heads = self.d_model  # 定义多头注意力的头数
        d_k = self.inter_d_model // n_heads  # 每个头的 key 的维度
        d_v = self.inter_d_model // n_heads  # 每个头的 value 的维度
        self.inter_patch_attention = Inter_Patch_Attention(self.inter_d_model, self.inter_d_model, n_heads, d_k, d_v,
                                                           attn_dropout=0,
                                                           proj_dropout=0.1, res_attention=False)
        # 初始化全局注意力模块，使用多头注意力

        ## 归一化层
        self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(self.d_model), Transpose(1, 2))
        # 用于局部注意力的归一化
        self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(self.d_model), Transpose(1, 2))
        # 用于全连接层的归一化

        ## 前馈网络（FFN）
        self.d_ff = d_ff  # FFN 中间层的维度
        self.dropout = nn.Dropout(0.1)  # Dropout 防止过拟合
        self.ff = nn.Sequential(nn.Linear(self.d_model, self.d_ff, bias=True),
                                nn.GELU(),
                                nn.Dropout(0.2),
                                nn.Linear(self.d_ff, self.d_model, bias=True))
        # 前馈网络，使用 GELU 激活函数

    def forward(self, x):
        # x(bs,seq_len,n_vars,d_model)
        new_x = x  # 保留输入 x
        batch_size = x.size(0)  # 获取批量大小
        intra_out_concat = None  # 用于存储局部注意力的输出
        weights_shared, biases_shared = self.weights_generator_shared()  # 获取共享权重和偏置
        weights_distinct, biases_distinct = self.weights_generator_distinct()  # 获取独立权重和偏置

        #### 局部注意力 #####
        for i in range(self.patch_nums):
            # t(bs,patch_size,n_vars,d_model)
            t = x[:, i * self.patch_size:(i + 1) * self.patch_size, :, :]  # 选择每个 patch 的输入
            # self.intra_embeddings[i](1,1,n_vars,d_model)
            # intra_emb(bs,1,n_vars,d_model),初始化一个参数矩阵来汇聚patch内计算注意力的结果
            intra_emb = self.embeddings_generator[i](self.intra_embeddings[i]).expand(batch_size, -1, -1, -1)
            # 使用嵌入生成器扩展嵌入到合适的维度
            # t(bs,patch_size+1,n_vars,d_model)
            t = torch.cat([intra_emb, t], dim=1)  # 将嵌入与输入拼接在一起

            out, attention = self.intra_patch_attention(intra_emb, t, t, weights_distinct, biases_distinct,
                                                        weights_shared, biases_shared)
            # 通过局部注意力层计算输出和注意力权重

            if intra_out_concat is None:
                intra_out_concat = out  # 如果是第一个 patch，初始化输出
            else:
                intra_out_concat = torch.cat([intra_out_concat, out], dim=1)  # 否则拼接输出
        # intra_out_concat:(bs,patch_num,n_vars,d_model)
        intra_out_concat = intra_out_concat.permute(0, 3, 2, 1)  # 调整维度顺序
        intra_out_concat = self.intra_Linear(intra_out_concat)  # 通过线性层
        # intra_out_concat:(bs,seq_len,n_vars,d_model)
        intra_out_concat = intra_out_concat.permute(0, 3, 2, 1)  # 调整回原始维度
        # intra_out_concat:(bs,seq_len,n_vars,d_model)
        #### 全局注意力 ######
        # x:(bs,patch_num,n_vars,d_model,patch_size)
        x = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        x = x.permute(0, 2, 1, 3, 4)  # 调整维度顺序
        # x(bs,n_vars,patch_num,d_model,patch_size)
        b, nvar, patch_num, dim, patch_len = x.shape  # 获取调整后的维度
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3] * x.shape[-1]))
        # 调整形状以适应全局注意力的输入要求
        # x(bs*n_vars,patch_num,d_model*patch_size)
        x = self.emb_linear(x)  # 通过线性层
        x = self.dropout(x + self.W_pos)  # 加上位置编码并进行 dropout

        inter_out = self.mambaLayer(x)
        attention = None
        inter_out = torch.reshape(inter_out, (b, nvar, inter_out.shape[-2], inter_out.shape[-1]))  # 调整维度
        inter_out = torch.reshape(inter_out, (b, nvar, inter_out.shape[-2], self.patch_size, self.d_model))
        inter_out = torch.reshape(inter_out, (b, self.patch_size * self.patch_nums, nvar, self.d_model))
        # inter_out:(bs,seq_len,n_vars,d_model)
        # 继续调整全局注意力输出的形状

        out = new_x + intra_out_concat + inter_out  # 将原始输入、局部和全局注意力输出相加
        if not self.args.use_inter:
            out = new_x + intra_out_concat
        if not self.args.use_intra:
            out = new_x + inter_out

        if self.batch_norm:
            out = self.norm_attn(out.reshape(b * nvar, self.patch_size * self.patch_nums, self.d_model))
        # 如果使用批量归一化，则应用归一化

        ## 前馈网络
        out = self.dropout(out)  # Dropout 层
        out = self.ff(out) + out  # 前馈网络并加上残差连接
        if self.batch_norm:
            out = self.norm_ffn(out).reshape(b, self.patch_size * self.patch_nums, nvar, self.d_model)
        # 如果使用批量归一化，继续应用归一化

        return out, attention  # 返回输出和注意力权重


class CustomLinear(nn.Module):
    def __init__(self, factorized):
        super(CustomLinear, self).__init__()
        self.factorized = factorized  # 是否使用因式分解

    def forward(self, input, weights, biases):
        if self.factorized:
            return torch.matmul(input.unsqueeze(3), weights).squeeze(3) + biases
        # 使用因式分解的线性变换
        else:
            return torch.matmul(input, weights) + biases
        # 常规的线性变换


class Intra_Patch_Attention(nn.Module):
    def __init__(self, d_model, factorized):
        super(Intra_Patch_Attention, self).__init__()
        self.head = 2  # 注意力头数

        if d_model % self.head != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')
        # 检查 d_model 是否能被 head 整除

        self.head_size = int(d_model // self.head)  # 每个头的维度
        self.custom_linear = CustomLinear(factorized)  # 自定义线性变换

    def forward(self, query, key, value, weights_distinct, biases_distinct, weights_shared, biases_shared):
        batch_size = query.shape[0]  # 获取批量大小

        key = self.custom_linear(key, weights_distinct[0], biases_distinct[0])  # 线性变换 key
        value = self.custom_linear(value, weights_distinct[1], biases_distinct[1])  # 线性变换 value
        # 将 query的d_model按头数拆分,然后按第0维拼接，query:(d_model*bs,n_vars,1,1)
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)
        # key,value:(d_model*bs,patch_size+1,n_vars,1)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)  # 拆分并拼接 key
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)  # 拆分并拼接 value

        query = query.permute((0, 2, 1, 3))  # 调整维度
        key = key.permute((0, 2, 3, 1))  # 调整维度
        value = value.permute((0, 2, 1, 3))  # 调整维度
        # query:(d_model*bs,n_vars,1,1)，
        # key:(d_model*bs,n_vars,1,patch_size+1)
        # value:(d_model*bs,n_vars,patch_size+1,1)
        attention = torch.matmul(query, key)  # 计算注意力得分
        attention /= (self.head_size ** 0.5)  # 缩放注意力得分
        attention = torch.softmax(attention, dim=-1)  # 归一化注意力得分

        x = torch.matmul(attention, value)  # 根据注意力得分计算输出
        # query:(d_model*bs,n_vars,1,1)
        x = x.permute((0, 2, 1, 3))  # 调整输出维度
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)  # 拼接输出
        # x:(bs,1,n_vars,d_model)
        if x.shape[0] == 0:
            x = x.repeat(1, 1, 1, int(weights_shared[0].shape[-1] / x.shape[-1]))
        # 防止输出为空，进行重复

        x = self.custom_linear(x, weights_shared[0], biases_shared[0])  # 线性变换
        x = torch.relu(x)  # ReLU 激活
        x = self.custom_linear(x, weights_shared[1], biases_shared[1])  # 线性变换
        # x:(bs,1,n_vars,d_model)
        return x, attention  # 返回输出和注意力权重


class Inter_Patch_Attention(nn.Module):
    def __init__(self, d_model, out_dim, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0.,
                 proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k  # 每个头的 key 维度
        d_v = d_model // n_heads if d_v is None else d_v  # 每个头的 value 维度

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v  # 初始化参数

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)  # 线性变换 Q
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)  # 线性变换 K
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)  # 线性变换 V

        # 多头注意力
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                  res_attention=self.res_attention, lsa=lsa)

        # 输出层
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, out_dim), nn.Dropout(proj_dropout))

    def forward(self, Q, K=None, V=None, prev=None, key_padding_mask=None, attn_mask=None):
        bs = Q.size(0)  # 获取批量大小
        if K is None: K = Q  # 如果 K 为空，设置为 Q
        if V is None: V = Q  # 如果 V 为空，设置为 Q

        q_s = self.W_Q(Q).view(bs, Q.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        # 将 Q 线性变换并分成多头
        k_s = self.W_K(K).view(bs, K.shape[1], self.n_heads, self.d_k).permute(0, 2, 3, 1)
        # 将 K 线性变换并调整维度
        v_s = self.W_V(V).view(bs, V.shape[1], self.n_heads, self.d_v).transpose(1, 2)
        # 将 V 线性变换并分成多头

        # 通过缩放点积注意力机制
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        output = output.transpose(1, 2).contiguous().view(bs, Q.shape[1], self.n_heads * self.d_v)
        # 调整输出的维度
        output = self.to_out(output)  # 通过线性层输出
        return output, attn_weights  # 返回输出和注意力权重


class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self attention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)  # Dropout 防止过拟合
        self.res_attention = res_attention  # 是否使用残差注意力
        head_dim = d_model // n_heads  # 每个头的维度
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        # 缩放系数
        self.lsa = lsa

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        # 执行前向传播
        attn_scores = torch.matmul(q, k) * self.scale  # 计算点积并进行缩放
        if prev is not None: attn_scores = attn_scores + prev  # 加上前一层的残差注意力

        # 添加可选的注意力掩码
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # 添加键填充掩码
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # 归一化注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # 使用 Softmax 归一化
        attn_weights = self.attn_dropout(attn_weights)  # Dropout

        # 计算最终输出
        output = torch.matmul(attn_weights, v)  # 根据权重计算 V 的加权和
        return output, attn_weights  # 返回输出和注意力权重


class WeightGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, mem_dim, num_nodes, factorized, number_of_weights=4):
        super(WeightGenerator, self).__init__()
        self.number_of_weights = number_of_weights  # 权重数量
        self.mem_dim = mem_dim  # 内存维度
        self.num_nodes = num_nodes  # 节点数量
        self.factorized = factorized  # 是否使用因式分解
        self.out_dim = out_dim  # 输出维度
        if self.factorized:
            self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cpu')
            # 初始化因式分解的内存参数
            self.generator = nn.Sequential(*[
                nn.Linear(mem_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 100)
            ])
            # 因式分解生成器
            self.mem_dim = 10
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, self.mem_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.Q = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim ** 2, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
        # 如果使用因式分解，初始化 P、Q 和 B 矩阵
        else:
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True) for _ in range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(1, out_dim), requires_grad=True) for _ in range(number_of_weights)])
        # 否则初始化普通权重和偏置
        self.reset_parameters()

    def reset_parameters(self):
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        for weight_list in list_params:
            for weight in weight_list:
                init.kaiming_uniform_(weight, a=math.sqrt(5))
        # 对权重进行初始化

        if not self.factorized:
            for i in range(self.number_of_weights):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.B[i], -bound, bound)
        # 初始化偏置项

    def forward(self):
        if self.factorized:
            memory = self.generator(self.memory.unsqueeze(1))
            bias = [torch.matmul(memory, self.B[i]).squeeze(1) for i in range(self.number_of_weights)]
            memory = memory.view(self.num_nodes, self.mem_dim, self.mem_dim)
            weights = [torch.matmul(torch.matmul(self.P[i], memory), self.Q[i]) for i in range(self.number_of_weights)]
            return weights, bias
        else:
            return self.P, self.B
        # 根据是否使用因式分解生成权重和偏置


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)
