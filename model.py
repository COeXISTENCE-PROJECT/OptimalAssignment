import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math





class nconv(nn.Module):
    """
    Simple graph convolution layer.
    Multiplies node features by the adjacency matrix.
    """

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        """
        x: Input features (batch_size, num_channels, num_nodes, time_steps)
        A: Adjacency matrix (num_nodes, num_nodes)

        The einsum operation 'ncvl,vw->ncwl' effectively multiplies each
        node's features by the edge weights of its neighbors.
        """
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    """
    Linear layer implemented as a 1x1 convolution.
    """

    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class fuse(nn.Module):
    """
    function that combines representation of Q from wavenet with Assignments representation
    """
    def __init__(self, dim_Q, dim_A, output_dim, method = 'concatenate'):
        super(fuse, self).__init__()
        self.method = method

        if self.method == 'concatenate':
            self.mlp = nn.Sequential(nn.Linear(dim_Q + dim_A, output_dim), nn.ReLU())

        elif self.method == 'Hadamard':
            self.proj_Q = nn.Linear(dim_Q, output_dim)  #this is probably the same
            self.proj_A = nn.Linear(dim_A, output_dim)  #this can be different in theory

        elif self.method == 'Attention':
            self.K = nn.Linear(dim_A, output_dim) #keys
            self.Q = nn.Linear(dim_Q, output_dim) #queries

            self.attention = nn.MultiheadAttention(embed_dim = output_dim, num_heads = 2, batch_first = True)




    def forward(self, Q, A):

        if self.method == 'concatenate':
            fused = torch.cat([Q,A], dim=-1)
            output = self.mlp(fused)

        elif self.method == 'Hadamard':
            q = self.proj_Q(Q)
            a = self.proj_A(A)
            output = q * a

        elif self.method == 'Attention':
            queries = self.Q(Q)
            keys = self.K(A)
            values = keys

            output, _ = self.attention(query = queries, key = keys, value = values)



        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for self attention using trigonometric functions.
    Supports both even and odd d_model.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        # dla nieparzystego d_model liczba kanałów cos jest o 1 mniejsza
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, d_model)

        Returns:
            Tensor of shape (B, T, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class PathEncoder(nn.Module):
    """
    Encodes a single agent path represented as an NxN binary matrix.
    The same encoder is shared across all agents and all time steps.
    """

    def __init__(self, n_nodes, path_embedding_dim, hidden_size = None, dropout = 0.0):
        super().__init__()

        input_size = n_nodes * n_nodes

        if hidden_size is not None:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, path_embedding_dim),
                nn.ReLU(),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, path_embedding_dim),
                nn.ReLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (num_agents, N*N)

        Returns:
            Tensor of shape (num_agents, path_embedding_dim)
        """
        return self.encoder(x)


class AssignmentEncoder(nn.Module):
    """
    Encodes a full assignment sequence A_seq of shape (B, T, N, N, M)
    into step embeddings of shape (B, T, embedding_size).

    Supported methods:
        - "sum": simple masked sum over agent embeddings
        - "attention_pool": single-vector attention pooling over agents
        - "k_latent": K learned latent queries attending to the set of agents
    """

    supported_methods = {"sum", "attention_pool", "k_latent"}

    def __init__(self,
        n_nodes: int,
        embedding_size: int,
        method: str = "sum",
        path_embedding_dim: int = 64,
        agent_hidden_size: int | None = None,
        dropout: float = 0.0,
        num_latents: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()

        if method not in self.supported_methods:
            raise ValueError(
                f"Unsupported method='{method}'. "
                f"Choose one of {sorted(self.supported_methods)}."
            )

        if method == "k_latent" and path_embedding_dim % num_heads != 0:
            raise ValueError(
                "For method='k_latent', path_embedding_dim must be divisible by num_heads."
            )

        self.n_nodes = n_nodes
        self.embedding_size = embedding_size
        self.method = method
        self.path_embedding_dim = path_embedding_dim
        self.num_latents = num_latents

        # Shared encoder for one agent path (NxN -> D_agent)
        self.agent_encoder = PathEncoder(
            n_nodes = n_nodes,
            path_embedding_dim=path_embedding_dim,
            hidden_size=agent_hidden_size,
            dropout=dropout,
        )

        # Optional attention pooling to one vector
        if self.method == "attention_pool":
            self.attn_score = nn.Linear(path_embedding_dim, 1)

        # K-latent set encoder:
        # K learnable queries attend to the set of agent embeddings
        if self.method == "k_latent":
            self.latent_queries = nn.Parameter(
                torch.randn(num_latents, path_embedding_dim)
            )
            self.latent_attention = nn.MultiheadAttention(
                embed_dim=path_embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        raw_output_size = self._get_raw_output_size()

        # Final projection to the step embedding size expected by GRU
        if raw_output_size == embedding_size:
            self.output_projection = nn.Identity()
        else:
            self.output_projection = nn.Sequential(
                nn.Linear(raw_output_size, embedding_size),
                nn.ReLU(),
            )

    def _get_raw_output_size(self) -> int:
        if self.method == "sum":
            base_size = self.path_embedding_dim
        elif self.method == "attention_pool":
            base_size = self.path_embedding_dim
        elif self.method == "k_latent":
            base_size = self.num_latents * self.path_embedding_dim
        else:
            raise RuntimeError("Unknown method encountered internally.")

        return base_size

    def _prepare_agents(self, A_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Converts (B, T, N, N, M) into:
            agent_tokens: (B*T, M, D_agent)
            agent_mask:   (B*T, M) bool, True for active agents
        """
        if A_seq.ndim != 5:
            raise ValueError(
                f"A_seq must have shape (B, T, N, N, M), got shape={tuple(A_seq.shape)}"
            )

        B, T, N1, N2, M = A_seq.shape

        if N1 != self.n_nodes or N2 != self.n_nodes:
            raise ValueError(
                f"Expected spatial shape ({self.n_nodes}, {self.n_nodes}), got ({N1}, {N2})."
            )

        # Move agent dimension before spatial dimensions:
        # (B, T, N, N, M) -> (B, T, M, N, N)
        A_seq = A_seq.permute(0, 1, 4, 2, 3).contiguous()

        # Detect active agents:
        # active if its NxN matrix is not entirely zero
        # shape: (B, T, M)
        agent_mask = (A_seq.abs().sum(dim=(-1, -2)) > 0)

        # Flatten each agent path matrix:
        # (B, T, M, N, N) -> (B*T, M, N*N)
        A_flat = A_seq.reshape(B * T, M, self.n_nodes * self.n_nodes).float()

        # Encode each agent independently:
        # (B*T, M, N*N) -> (B*T, M, D_agent)
        agent_tokens = self.agent_encoder(A_flat)

        # Flatten mask along (B, T)
        agent_mask = agent_mask.reshape(B * T, M)

        return agent_tokens, agent_mask, B, T

    def _sum_pool(self, agent_tokens: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        """
        Masked sum over agent dimension.
        """
        masked_tokens = agent_tokens * agent_mask.unsqueeze(-1).float()
        pooled = masked_tokens.sum(dim=1)  # (B*T, D_agent)
        return pooled

    def _attention_pool(self, agent_tokens: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        """
        Learned attention pooling from a set of agent embeddings to a single vector.
        """
        # (B*T, M)
        scores = self.attn_score(agent_tokens).squeeze(-1)

        # Mask inactive agents before softmax
        scores = scores.masked_fill(~agent_mask, -1e9)

        # Standard masked softmax with extra protection for all-empty rows
        weights = torch.softmax(scores, dim=-1)
        weights = weights * agent_mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1.0)

        # Weighted sum: (B*T, 1, M) x (B*T, M, D) -> (B*T, D)
        pooled = torch.bmm(weights.unsqueeze(1), agent_tokens).squeeze(1)
        return pooled

    def _k_latent_pool(
        self, agent_tokens: torch.Tensor, agent_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        K learned latent queries attend to the set of agent embeddings.

        Output:
            (B*T, K * D_agent)
        """
        BT, M, D = agent_tokens.shape

        # Expand K latent queries for each sample in the batch
        queries = self.latent_queries.unsqueeze(0).expand(BT, -1, -1)  # (BT, K, D)

        # MultiheadAttention uses key_padding_mask=True for positions to ignore
        key_padding_mask = ~agent_mask  # (BT, M)

        # If a row has no active agents, attention would become numerically unstable.
        # We fix it by unmasking one dummy zero token.
        empty_rows = ~agent_mask.any(dim=1)
        if empty_rows.any():
            agent_tokens = agent_tokens.clone()
            key_padding_mask = key_padding_mask.clone()

            agent_tokens[empty_rows, 0, :] = 0.0
            key_padding_mask[empty_rows, 0] = False

        # Cross-attention from latent queries to agent tokens
        slots, _ = self.latent_attention(
            query=queries,
            key=agent_tokens,
            value=agent_tokens,
            key_padding_mask=key_padding_mask,
        )  # (BT, K, D)

        # Flatten K latent slots into one step vector
        pooled = slots.reshape(BT, self.num_latents * D)
        return pooled

    def forward(self, A_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_seq: Tensor of shape (B, T, N, N, M)

        Returns:
            step_embeddings: Tensor of shape (B, T, embedding_size)
        """
        agent_tokens, agent_mask, B, T = self._prepare_agents(A_seq)

        if self.method == "sum":
            raw_step_repr = self._sum_pool(agent_tokens, agent_mask)
        elif self.method == "attention_pool":
            raw_step_repr = self._attention_pool(agent_tokens, agent_mask)
        elif self.method == "k_latent":
            raw_step_repr = self._k_latent_pool(agent_tokens, agent_mask)
        else:
            raise RuntimeError("Unknown method encountered internally.")

        # Project to the final embedding size for the temporal model
        step_embeddings = self.output_projection(raw_step_repr)  # (B*T, embedding_size)

        # Restore time dimension
        step_embeddings = step_embeddings.reshape(B, T, self.embedding_size)
        return step_embeddings



class GRU_Representation(nn.Module):
    """
    Assignment sequence encoder:
        (B, T, N, N, M) -> AssignmentEncoder -> GRU -> (B, T, hidden_size)

    Notes:
        - AssignmentEncoder is permutation-invariant with respect to agent order in M
          for method="sum", "attention_pool", and "k_latent".
    """

    def __init__(
        self,
        n_nodes: int,
        embedding_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        assignment_method: str = "sum",
        path_embedding_dim: int = 64,
        agent_hidden_size: int | None = None,
        num_latents: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()

        self.assignment_encoder = AssignmentEncoder(
            n_nodes=n_nodes,
            embedding_size=embedding_size,
            method=assignment_method,
            path_embedding_dim=path_embedding_dim,
            agent_hidden_size=agent_hidden_size,
            dropout=dropout,
            num_latents=num_latents,
            num_heads=num_heads,
        )

        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        A_seq: torch.Tensor,
        lengths: torch.Tensor | None = None,
        return_hidden: bool = False,
        return_step_embeddings: bool = False,
    ):
        """
        Args:
            A_seq:
                Tensor of shape (B, T, N, N, M)
            return_hidden:
                If True, also return the final GRU hidden state.
            return_step_embeddings:
                If True, also return step embeddings produced by AssignmentEncoder.

        Returns:
            output:
                Tensor of shape (B, T, hidden_size)

            Optionally also:
                hidden:
                    Tensor of shape (num_layers, B, hidden_size)
                step_embeddings:
                    Tensor of shape (B, T, embedding_size)
        """
        # Step-wise assignment embeddings
        step_embeddings = self.assignment_encoder(A_seq)  # (B, T, embedding_size)

        #GRU with batch_first expect square tensor (with all sequences the same length for some reason)

        if lengths is not None:
            lengths_cpu = lengths.detach().cpu()

            packed = pack_padded_sequence(
                step_embeddings,
                lengths=lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )

            packed_output, hidden = self.gru(packed)

            output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=A_seq.size(1),
            )
        else:
            output, hidden = self.gru(step_embeddings)

        if return_hidden and return_step_embeddings:
            return output, hidden, step_embeddings
        if return_hidden:
            return output, hidden
        if return_step_embeddings:
            return output, step_embeddings

        return output


class LSTM_Representation(nn.Module):
    """
    Assignment sequence encoder:
        (B, T, N, N, M) -> AssignmentEncoder -> LSTM -> (B, T, hidden_size)

    Notes:
        - AssignmentEncoder is permutation-invariant with respect to agent order in M
          for method="sum", "attention_pool", and "k_latent".
    """

    def __init__(
        self,
        n_nodes: int,
        embedding_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        assignment_method: str = "sum",
        path_embedding_dim: int = 64,
        agent_hidden_size: int | None = None,
        num_latents: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()

        self.assignment_encoder = AssignmentEncoder(
            n_nodes=n_nodes,
            embedding_size=embedding_size,
            method=assignment_method,
            path_embedding_dim=path_embedding_dim,
            agent_hidden_size=agent_hidden_size,
            dropout=dropout,
            num_latents=num_latents,
            num_heads=num_heads,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        A_seq: torch.Tensor,
        lengths: torch.Tensor | None = None,
        return_hidden: bool = False,
        return_cell: bool = False,
        return_step_embeddings: bool = False,
    ):
        """
        Args:
            A_seq:
                Tensor of shape (B, T, N, N, M)
            lengths:
                Optional tensor of true sequence lengths, shape (B,)
            return_hidden:
                If True, also return final hidden state.
            return_cell:
                If True, also return final cell state.
            return_step_embeddings:
                If True, also return step embeddings from AssignmentEncoder.

        Returns:
            output:
                Tensor of shape (B, T, hidden_size)

            Optionally also:
                hidden:
                    Tensor of shape (num_layers, B, hidden_size)
                cell:
                    Tensor of shape (num_layers, B, hidden_size)
                step_embeddings:
                    Tensor of shape (B, T, embedding_size)
        """
        step_embeddings = self.assignment_encoder(A_seq)  # (B, T, embedding_size)

        if lengths is not None:
            lengths_cpu = lengths.detach().cpu()

            packed = pack_padded_sequence(
                step_embeddings,
                lengths=lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )

            packed_output, (hidden, cell) = self.lstm(packed)

            output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=A_seq.size(1),
            )
        else:
            output, (hidden, cell) = self.lstm(step_embeddings)

        results = [output]

        if return_hidden:
            results.append(hidden)
        if return_cell:
            results.append(cell)
        if return_step_embeddings:
            results.append(step_embeddings)

        if len(results) == 1:
            return results[0]
        return tuple(results)


class AttentionRepresentation(nn.Module):
    """
    Assignment sequence encoder:
        (B, T, N, N, M) -> AssignmentEncoder -> PositionalEncoding
        -> TransformerEncoder -> (B, T, embedding_size)

    Notes:
        - AssignmentEncoder is permutation-invariant with respect to agent order in M
          for method="sum", "attention_pool", and "k_latent".
    """

    def __init__(
        self,
        n_nodes: int,
        embedding_size: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        assignment_method: str = "sum",
        path_embedding_dim: int = 64,
        agent_hidden_size: int | None = None,
        num_latents: int = 4,
        assignment_num_heads: int = 4,
        max_len: int = 5000,
    ):
        super().__init__()

        if embedding_size % num_heads != 0:
            raise ValueError(
                f"embedding_size={embedding_size} must be divisible by num_heads={num_heads}."
            )

        self.assignment_encoder = AssignmentEncoder(
            n_nodes=n_nodes,
            embedding_size=embedding_size,
            method=assignment_method,
            path_embedding_dim=path_embedding_dim,
            agent_hidden_size=agent_hidden_size,
            dropout=dropout,
            num_latents=num_latents,
            num_heads=assignment_num_heads,
        )

        self.pos_encoder = PositionalEncoding(
            d_model=embedding_size,
            max_len=max_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    @staticmethod
    def _build_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Returns:
            mask: bool tensor of shape (B, T)
            True means 'ignore / pad'
        """
        device = lengths.device
        positions = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
        mask = positions >= lengths.unsqueeze(1)  # (B, T)
        return mask

    def forward(
        self,
        A_seq: torch.Tensor,
        lengths: torch.Tensor | None = None,
        return_step_embeddings: bool = False,
    ):
        """
        Args:
            A_seq:
                Tensor of shape (B, T, N, N, M)
            lengths:
                Optional tensor of true sequence lengths, shape (B,)
            return_step_embeddings:
                If True, also return step embeddings from AssignmentEncoder.

        Returns:
            output:
                Tensor of shape (B, T, embedding_size)

            Optionally also:
                step_embeddings:
                    Tensor of shape (B, T, embedding_size)
        """
        step_embeddings = self.assignment_encoder(A_seq)  # (B, T, embedding_size)
        embeddings_pos = self.pos_encoder(step_embeddings)

        src_key_padding_mask = None
        if lengths is not None:
            src_key_padding_mask = self._build_padding_mask(
                lengths=lengths,
                max_len=A_seq.size(1),
            )

        output = self.transformer_encoder(
            embeddings_pos,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, T, embedding_size)

        if return_step_embeddings:
            return output, step_embeddings
        return output

class gcn(nn.Module):
    """
    Graph Convolution Network (GCN) module.
    Implements diffusion graph convolution.
    """

    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()

        # Calculate input channels for the linear layer.
        # (order * support_len + 1) accounts for the original signal
        # and signals after 'k' diffusion steps for each support matrix.
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        """
        x: Input data
        support: List of adjacency matrices (e.g., original, transposed, adaptive)
        """
        out = [x]

        # For each support matrix (e.g., forward/backward directed graph)
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            # Multi-step diffusion (order k)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        # Concatenate processed signals along the channel dimension
        h = torch.cat(out, dim=1)
        # Linear layer (dimensionality reduction and feature mixing)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    """
    Main Graph WaveNet model.
    Combines Temporal Convolutional Networks (TCN) for temporal dependencies
    and Graph Convolutional Networks (GCN) for spatial dependencies.
    """

    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None,
                 in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        # Initial 1x1 convolution matching input channels to residual_channels
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        # Initialize adaptive adjacency matrix
        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                # Random initialization of node embeddings
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                # SVD-based initialization
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        # Build WaveNet (TCN) layers
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # Dilated convolutions
                # Two paths: filter and gate (LSTM/GRU-like gating mechanism)
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=(1, new_dilation)))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=(1, new_dilation)))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

                # Add GCN layer in each block if enabled
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        # Output layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):

        in_len = input.size(3)

        # Pad input if the sequence is shorter than the receptive field
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)
        skip = 0

        # Calculate adaptive adjacency matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            # adp = softmax(ReLU(E1 * E2))
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # Loop through WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + --> *input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + -------------> *skip*

            residual = x

            # Dilated convolution (temporal)
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)  # Filter activation
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate  # Gated TCN

            # Parametrized skip connection (accumulating intermediate results)
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # Spatial processing (GCN)
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    for j, a in enumerate(new_supports):
                        x = self.gconv[i](x, new_supports)
                else:
                    for j, a in enumerate(self.supports):

                        x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            # Add residual connection
            x = x + residual[:, :, :, -x.size(3):]

            # Normalization
            x = self.bn[i](x)

        # Final processing of accumulated skip connections
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


