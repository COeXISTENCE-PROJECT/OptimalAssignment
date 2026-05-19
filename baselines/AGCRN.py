import torch.nn as nn
import torch
import torch.functional as F

class AVWGCN(nn.Module):
    """
    Adaptive Vertex-wise Graph Convolution, AGCRN-style.

    x:              (B, N, C_in)
    node_embeddings:(N, D)
    return:         (B, N, C_out)
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        cheb_k: int,
        embed_dim: int,
    ):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.empty(embed_dim, cheb_k, dim_in, dim_out)
        )
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, dim_out))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.xavier_uniform_(self.bias_pool)

    def forward(
        self,
        x: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = node_embeddings.size(0)
        device = x.device
        dtype = x.dtype

        node_embeddings = node_embeddings.to(device=device, dtype=dtype)

        # adaptive adjacency: (N, N)
        supports = F.softmax(
            F.relu(torch.mm(node_embeddings, node_embeddings.t())),
            dim=1,
        )

        support_set = [
            torch.eye(num_nodes, device=device, dtype=dtype),
            supports,
        ]

        # Chebyshev basis
        for _ in range(2, self.cheb_k):
            support_set.append(
                2 * torch.matmul(supports, support_set[-1]) - support_set[-2]
            )

        supports = torch.stack(support_set, dim=0)  # (K, N, N)

        # graph signal: (B, K, N, C_in) -> (B, N, K, C_in)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)

        # node-specific weights
        weights = torch.einsum(
            "nd,dkio->nkio",
            node_embeddings,
            self.weights_pool.to(device=device, dtype=dtype),
        )  # (N, K, C_in, C_out)

        bias = torch.matmul(
            node_embeddings,
            self.bias_pool.to(device=device, dtype=dtype),
        )  # (N, C_out)

        out = torch.einsum("bnki,nkio->bno", x_g, weights) + bias
        return out


class AGCRNCell(nn.Module):
    """
    GRU cell with adaptive graph convolution.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        cheb_k: int,
        embed_dim: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gate = AVWGCN(
            dim_in=input_dim + hidden_dim,
            dim_out=2 * hidden_dim,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
        )

        self.update = AVWGCN(
            dim_in=input_dim + hidden_dim,
            dim_out=hidden_dim,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, N, input_dim)
        h: (B, N, hidden_dim)
        """
        x_h = torch.cat([x, h], dim=-1)

        z_r = torch.sigmoid(self.gate(x_h, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)

        candidate_input = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.update(candidate_input, node_embeddings))

        h_new = z * h + (1.0 - z) * h_tilde
        return h_new
