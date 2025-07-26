import torch
import torch.nn as nn
from models.expert.layers.Layer import Transformer_Layer
from models.expert.utils.Other import FourierLayer, series_decomp_multi, MLP
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

class expert_layer(nn.Module):
    def __init__(self, input_size, output_size, num_experts, device, num_nodes=1, d_model=32, d_ff=64, dynamic=False,
                 patch_size=[8, 6, 4, 2], noisy_gating=True, k=4, layer_number=1, residual_connection=1,
                 batch_norm=False, distribute_by_score=True, args=None):
        super(expert_layer, self).__init__()

        self.num_experts = num_experts
        self.output_size = output_size
        self.device = device
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.distribute_by_score = distribute_by_score
        self.args = args

        self.start_linear = nn.Linear(in_features=1, out_features=d_model)

        self.seasonality_model = FourierLayer(pred_len=0, k=k)
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])

        self.experts = nn.ModuleList(
            [Transformer_Layer(device=device, d_model=d_model, d_ff=d_ff, dynamic=dynamic, num_nodes=num_nodes,
                               patch_nums=int(input_size / patch), patch_size=patch, factorized=True, args=args,
                               layer_number=layer_number, batch_norm=batch_norm) for patch in patch_size]
        )

        self.softmax = nn.Softmax(1)
        self.end_MLP = MLP(input_size=input_size, output_size=output_size)

        if self.args and self.args.train_data:
            self.plot_base_dir = f"/usr/local/lzlconda/file/ssh_4090_2/Depression_Analysis_Tool/t-SNE/{self.args.train_data[0]}"
        else:
            self.plot_base_dir = "/usr/local/lzlconda/file/ssh_4090_2/Depression_Analysis_Tool_/t-SNE/default"
            print("Warning: args or args.train_data is not properly set. Using default directory for t-SNE plots.")

    def seasonality_and_trend_decompose(self, x):
        x = x[:, :, :, 0]
        _, trend = self.trend_model(x)
        seasonality, _ = self.seasonality_model(x)
        return x + seasonality + trend

    def split_by_score(self, x, scores):
        batch_size, seq_len, _ = x.shape

        if self.distribute_by_score:
            score_bins = torch.linspace(scores.min(), scores.max(), steps=self.num_experts + 1, device=self.device)
            expert_assignments = torch.bucketize(scores, score_bins, right=True) - 1
            expert_assignments = torch.clamp(expert_assignments, 0, self.num_experts - 1)
            x_no_score = x[:, :, :-1]

            expert_inputs = torch.zeros(self.num_experts, batch_size, seq_len, x_no_score.size(2), device=self.device)
            expert_masks = torch.zeros(self.num_experts, batch_size, seq_len, device=self.device)

            for i in range(self.num_experts):
                mask = expert_assignments == i
                expert_inputs[i] = x_no_score * mask.unsqueeze(-1)
                expert_masks[i] = mask.float()
        else:
            expert_inputs = torch.stack([x[:, :, :-1] for _ in range(self.num_experts)])
            expert_masks = torch.ones(self.num_experts, batch_size, seq_len, device=self.device)
            expert_assignments = torch.zeros(batch_size, seq_len, device=self.device, dtype=torch.long)

        return expert_inputs, expert_masks, expert_assignments

    def combine_results(self, expert_outputs, expert_masks, expert_assignments, batch_size, seq_len):
        expert_outputs = torch.stack(expert_outputs)
        one_hot_assignments = torch.nn.functional.one_hot(expert_assignments, self.num_experts).permute(0, 2, 1)
        one_hot_assignments = one_hot_assignments.unsqueeze(-1).unsqueeze(-1)

        expert_outputs = expert_outputs.permute(1, 0, 2, 3, 4)
        selected_outputs = (one_hot_assignments * expert_outputs).sum(dim=1)

        return selected_outputs

    def forward(self, x, original_x, layer_idx=1, sample_index=None, epoch=None, is_training=True):
        batch_size, seq_len = x.size(0), x.size(1)

        new_x = self.seasonality_and_trend_decompose(x)
        scores = original_x[:, :, -1]

        expert_inputs, expert_masks, expert_assignments = self.split_by_score(new_x, scores)
        expert_outputs = []

        for i in range(self.num_experts):
            if expert_masks[i].sum() > 0:
                output, _ = self.experts[i](self.start_linear(expert_inputs[i].unsqueeze(-1)))
                expert_outputs.append(output)
            else:
                expert_outputs.append(
                    torch.zeros(batch_size, seq_len, self.num_nodes, self.d_model, device=self.device))

        output = self.combine_results(expert_outputs, expert_masks, expert_assignments, batch_size, seq_len)
        return output

    def visualize_tsne(self, expert_features, layer_idx, epoch):
        if not hasattr(self, '_sample_counter'):
            self._sample_counter = 0
            self._total_samples = 15
            self._pbar = tqdm(total=self._total_samples, desc="t-SNE Visualization", unit="sample")
        else:
            self._sample_counter += 1

        if not expert_features or any(x.nelement() == 0 for x in expert_features):
            print("Empty expert features. Skipping t-SNE visualization.")
            if hasattr(self, '_pbar'):
                self._pbar.update(1)
            return

        sample_index = self._sample_counter
        sample_dir = os.path.join(self.plot_base_dir, f"sample_{sample_index}")
        os.makedirs(sample_dir, exist_ok=True)

        try:
            expert_means = torch.stack([feat.mean(dim=(0, 1)).view(-1).detach() for feat in expert_features])
            expert_means = expert_means.cpu().numpy()
            print(f"After processing, expert_means shape: {expert_means.shape}")
        except RuntimeError as e:
            print(f"Feature processing error: {e}. Skipping t-SNE.")
            if hasattr(self, '_pbar'):
                self._pbar.update(1)
            return

        if expert_means.ndim != 2 or expert_means.shape[0] != self.num_experts:
            print("Incorrect number of expert features. Skipping visualization.")
            if hasattr(self, '_pbar'):
                self._pbar.update(1)
            return

        if expert_means.shape[1] > 1000:
            expert_means = expert_means[:, ::3]

        n_components_pca = min(self.num_experts, expert_means.shape[1], 50)
        pca = PCA(n_components=n_components_pca)
        expert_means_pca = pca.fit_transform(expert_means)

        tsne = TSNE(n_components=3, random_state=42, perplexity=min(5, self.num_experts - 1), n_iter=250)

        try:
            tsne_results = tsne.fit_transform(expert_means_pca)
        except ValueError as e:
            print(f"t-SNE error: {e}. Skipping visualization.")
            if hasattr(self, '_pbar'):
                self._pbar.update(1)
            return
        except Exception as e:
            print(f"Unexpected t-SNE error: {e}. Skipping.")
            if hasattr(self, '_pbar'):
                self._pbar.update(1)
            return

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        pastel_colors = [
            '#ADD8E6',
            '#F08080',
            '#90EE90',
            '#D3D3D3',
            '#E6E6FA',
            '#FFFACD',
            '#F5F5DC',
            '#FAEBD7',
            '#00FFFF',
            '#7FFFD4',
        ]

        for i in range(self.num_experts):
            ax.scatter(tsne_results[i, 0], tsne_results[i, 1], tsne_results[i, 2],
                       color=pastel_colors[i % len(pastel_colors)],
                       label=f'Expert_{i + 1}',
                       marker='o', s=50)

            ax.text(tsne_results[i, 0], tsne_results[i, 1], tsne_results[i, 2],
                    f'Expert_{i + 1}', fontsize=12)

        ax.set_title(f'TSNE 3D of Layer {layer_idx}, Sample {sample_index}')
        ax.legend()
        plt.tight_layout()

        filename = f"tsne_visualization_layer{layer_idx}_epoch{epoch}.png"
        filepath = os.path.join(sample_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.draw()
        plt.pause(0.1)
        plt.close(fig)

        if os.path.exists(filepath):
            print(f"t-SNE visualization saved: {filepath}")
        else:
            print(f"Warning: t-SNE visualization not saved correctly: {filepath}")

        if hasattr(self, '_pbar'):
            self._pbar.update(1)
            if self._sample_counter == self._total_samples:
                self._pbar.close()