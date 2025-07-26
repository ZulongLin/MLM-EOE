import torch
import torch.nn as nn
from models.expert.layers.Expert_layer import expert_layer
from models.expert.layers.RevIN import RevIN



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.layer_nums = configs.layer_nums
        self.num_nodes = configs.num_nodes
        self.pre_len = configs.pred_len
        self.seq_len = configs.true_len
        self.k = configs.k
        self.num_experts_list = configs.num_experts_list
        self.patch_size_list = configs.patch_size_list
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = 1
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.expert_layer_lists = nn.ModuleList()
        self.device = torch.device('cuda:{}'.format(configs.devices[0]))
        self.batch_norm = configs.batch_norm

        for num in range(self.layer_nums):
            self.expert_layer_lists.append(
                expert_layer(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list, noisy_gating=True,
                    distribute_by_score=configs.distribute_by_score,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1,
                    residual_connection=self.residual_connection, batch_norm=self.batch_norm, args=configs))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forward(self, x):
        # x(bs,seq_len,n_vars)
        original_x = x
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        original_out = self.start_fc(x.unsqueeze(-1))

        batch_size = x.shape[0]
        # out(bs,seq_len,n_vars,d_model)
        for i, layer in enumerate(self.expert_layer_lists):
            if i == 0:
                out = layer(x=original_out, original_x=original_x, layer_idx=i)
            else:
                out += layer(x=original_out, original_x=original_x, layer_idx=i)

        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        return out.mean(-1)
