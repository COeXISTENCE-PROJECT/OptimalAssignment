import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
from STAEformer import STAEformer




class nconv(nn.Module):
    """
    Simple graph convolution layer.
    Multiplies node features by the adjacency matrix.
    """

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, adj):
        """
        x: Input features (batch_size, num_channels, num_nodes, time_steps)
        A: Adjacency matrix (num_nodes, num_nodes)

        The einsum operation 'ncvl,vw->ncwl' effectively multiplies each
        node's features by the edge weights of its neighbors.
        """
        x = torch.einsum('ncvl,vw->ncwl', (x, adj))
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
    def __init__(self, dim_Q, dim_A, output_dim, method='concatenate'):
        super().__init__()

        canonical_method = {
            'concatenate': 'concatenate',
            'hadamard': 'Hadamard',
            'attention': 'Attention',
            'wavenet_only': 'wavenet_only',
            'assignment_only': 'assignment_only',
        }.get(method.lower(), method)

        self.method = canonical_method
        self.output_dim = output_dim

        if self.method == 'concatenate':
            self.mlp = nn.Sequential(
                nn.Linear(dim_Q + dim_A, output_dim),
                nn.ReLU(),
            )

        elif self.method == 'Hadamard':
            self.proj_Q = nn.Linear(dim_Q, output_dim)
            self.proj_A = nn.Linear(dim_A, output_dim)

        elif self.method == 'Attention':
            self.q_proj = nn.Linear(dim_Q, output_dim)
            self.a_proj = nn.Linear(dim_A, output_dim)

            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=2,
                batch_first=True,
            )

            self.out_proj = nn.Linear(output_dim, output_dim)

        elif self.method == 'wavenet_only':
            self.q_only_proj = nn.Sequential(
                nn.Linear(dim_Q, output_dim),
                nn.ReLU(),
            )

        elif self.method == 'assignment_only':
            self.a_only_proj = nn.Sequential(
                nn.Linear(dim_A, output_dim),
                nn.ReLU(),
            )

        else:
            raise ValueError(f"Unknown fuse method: {method}")

    def forward(self, Q, A):
        if Q.ndim != 2 or A.ndim != 2:
            raise ValueError(
                f"fuse expects Q and A with shape (B, D), got {Q.shape} and {A.shape}"
            )

        if self.method == 'concatenate':
            fused = torch.cat([Q, A], dim=-1)
            output = self.mlp(fused)

        elif self.method == 'Hadamard':
            q = self.proj_Q(Q)
            a = self.proj_A(A)
            output = q * a

        elif self.method == 'Attention':
            q = self.q_proj(Q)                      # (B, D)
            a = self.a_proj(A)                      # (B, D)

            query = q.unsqueeze(1)                  # (B, 1, D)
            key_value = torch.stack([q, a], dim=1)  # (B, 2, D)

            output, _ = self.attention(
                query=query,
                key=key_value,
                value=key_value,
            )

            output = output.squeeze(1)              # (B, D)
            output = output + q                     # residual
            output = self.out_proj(output)

        elif self.method == 'wavenet_only':
            output = self.q_only_proj(Q)

        elif self.method == 'assignment_only':
            output = self.a_only_proj(A)

        else:
            raise RuntimeError(
                f"fuse.forward reached unsupported method={self.method!r}"
            )

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
    Encodes a single step represented as a graph signal of shape (B, N),
    where N is the number of nodes and each node has 1 feature.

    Default encoder: diffusion GCN.
    """

    def __init__(
        self,
        n_nodes: int,
        path_embedding_dim: int,
        hidden_size: int = 64,
        dropout: float = 0.0,
        support_len: int = 3,
        order: int = 2,
        pooling: str = "mean",
    ):
        super().__init__()

        self.n_nodes = n_nodes
        self.pooling = pooling
        self.support_len = support_len

        self.gcn1 = gcn(
            c_in=1,
            c_out=hidden_size,
            dropout=dropout,
            support_len=support_len,
            order=order,
        )

        self.gcn2 = gcn(
            c_in=hidden_size,
            c_out=hidden_size,
            dropout=dropout,
            support_len=support_len,
            order=order,
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, path_embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, support: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x:
                Tensor of shape (B, N)
            support:
                list of adjacency/support matrices, each of shape (N, N)

        Returns:
            Tensor of shape (B, path_embedding_dim)
        """
        if x.ndim != 2:
            raise ValueError(
                f"x must have shape (B, N), got shape={tuple(x.shape)}"
            )

        if x.shape[1] != self.n_nodes:
            raise ValueError(
                f"Expected x.shape[1] == {self.n_nodes}, got {x.shape[1]}"
            )

        if len(support) != self.support_len:
            raise ValueError(
                f"Expected {self.support_len} support matrices, got {len(support)}"
            )

        # (B, N) -> (B, 1, N, 1)
        x = x.float().unsqueeze(1).unsqueeze(-1)

        h = self.gcn1(x, support)   # (B, hidden, N, 1)
        h = F.relu(h)

        h = self.gcn2(h, support)   # (B, hidden, N, 1)
        h = F.relu(h)

        # (B, hidden, N, 1) -> (B, hidden, N)
        h = h.squeeze(-1)

        if self.pooling == "mean":
            h = h.mean(dim=-1)              # (B, hidden)
        elif self.pooling == "max":
            h = h.max(dim=-1).values        # (B, hidden)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.out_proj(h)             # (B, path_embedding_dim)



class AssignmentEncoder(nn.Module):
    """
    Encodes a sequence of graph signals of shape (B, T, N)
    into step embeddings of shape (B, T, embedding_size).
    """

    def __init__(
        self,
        n_nodes: int,
        embedding_size: int,
        path_embedding_dim: int = 64,
        path_hidden_size: int = 64,
        dropout: float = 0.0,
        support_len: int = 3,
        order: int = 2,
        pooling: str = "mean",
    ):
        super().__init__()

        self.n_nodes = n_nodes
        self.embedding_size = embedding_size
        self.path_embedding_dim = path_embedding_dim

        self.path_encoder = PathEncoder(
            n_nodes=n_nodes,
            path_embedding_dim=path_embedding_dim,
            hidden_size=path_hidden_size,
            dropout=dropout,
            support_len=support_len,
            order=order,
            pooling=pooling,
        )

        if path_embedding_dim == embedding_size:
            self.output_projection = nn.Identity()
        else:
            self.output_projection = nn.Sequential(
                nn.Linear(path_embedding_dim, embedding_size),
                nn.ReLU(),
            )

    def forward(self, A_seq: torch.Tensor, support: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            A_seq:
                Tensor of shape (B, T, N)
            support:
                list of support matrices, each of shape (N, N)

        Returns:
            step_embeddings:
                Tensor of shape (B, T, embedding_size)
        """
        if A_seq.ndim != 3:
            raise ValueError(
                f"A_seq must have shape (B, T, N), got shape={tuple(A_seq.shape)}"
            )

        B, T, N = A_seq.shape

        if N != self.n_nodes:
            raise ValueError(
                f"Expected last dim = {self.n_nodes}, got {N}"
            )

        # (B, T, N) -> (B*T, N)
        x = A_seq.reshape(B * T, N)

        # (B*T, N) -> (B*T, path_embedding_dim)
        step_repr = self.path_encoder(x, support)

        # (B*T, path_embedding_dim) -> (B*T, embedding_size)
        step_embeddings = self.output_projection(step_repr)

        # (B*T, embedding_size) -> (B, T, embedding_size)
        step_embeddings = step_embeddings.reshape(B, T, self.embedding_size)

        return step_embeddings





class GRU_Representation(nn.Module):
    """
    Sequence encoder:
        (B, T, N) -> AssignmentEncoder(GCN) -> GRU -> (B, T, hidden_size)
    """

    def __init__(
        self,
        n_nodes: int,
        embedding_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        path_embedding_dim: int = 64,
        path_hidden_size: int = 64,
        support_len: int = 3,
        order: int = 2,
        pooling: str = "mean",
    ):
        super().__init__()

        self.assignment_encoder = AssignmentEncoder(
            n_nodes=n_nodes,
            embedding_size=embedding_size,
            path_embedding_dim=path_embedding_dim,
            path_hidden_size=path_hidden_size,
            dropout=dropout,
            support_len=support_len,
            order=order,
            pooling=pooling,
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
        a_seq: torch.Tensor,
        support: list[torch.Tensor],
        lengths: torch.Tensor | None = None,
        return_sequence: bool = False,
        return_hidden: bool = False,
        return_step_embeddings: bool = False,
    ):
        step_embeddings = self.assignment_encoder(a_seq, support)  # (B, T, embedding_size)

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
                total_length=a_seq.size(1),
            )

            last_repr = hidden[-1]  # (B, hidden_size)
        else:
            output, hidden = self.gru(step_embeddings)
            last_repr = hidden[-1]  # (B, hidden_size)

        results = [last_repr]

        if return_sequence:
            results.append(output)
        if return_hidden:
            results.append(hidden)
        if return_step_embeddings:
            results.append(step_embeddings)

        if len(results) == 1:
            return results[0]
        return tuple(results)


class LSTM_Representation(nn.Module):
    """
    Sequence encoder:
        (B, T, N) -> AssignmentEncoder(GCN) -> LSTM -> (B, T, hidden_size)
    """

    def __init__(
        self,
        n_nodes: int,
        embedding_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        path_embedding_dim: int = 64,
        path_hidden_size: int = 64,
        support_len: int = 3,
        order: int = 2,
        pooling: str = "mean",
    ):
        super().__init__()

        self.assignment_encoder = AssignmentEncoder(
            n_nodes=n_nodes,
            embedding_size=embedding_size,
            path_embedding_dim=path_embedding_dim,
            path_hidden_size=path_hidden_size,
            dropout=dropout,
            support_len=support_len,
            order=order,
            pooling=pooling,
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
        a_seq: torch.Tensor,
        supports: list[torch.Tensor],
        lengths: torch.Tensor | None = None,
        return_sequence: bool = False,
        return_hidden: bool = False,
        return_cell: bool = False,
        return_step_embeddings: bool = False,
    ):
        """
        Args:
            A_seq: (B, T, N)
            support: lista macierzy support, każda (N, N)
        """
        step_embeddings = self.assignment_encoder(a_seq, supports)  # (B, T, d_step)

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
                total_length=a_seq.size(1),
            )

            last_repr = hidden[-1]  # (B, hidden_size)
        else:
            output, (hidden, cell) = self.lstm(step_embeddings)
            last_repr = hidden[-1]  # (B, hidden_size)

        results = [last_repr]

        if return_sequence:
            results.append(output)
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
            path_embedding_dim: int = 64,
            path_hidden_size: int = 64,
            support_len: int = 3,
            order: int = 2,
            pooling: str = "mean",
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
            path_embedding_dim=path_embedding_dim,
            path_hidden_size=path_hidden_size,
            dropout=dropout,
            support_len=support_len,
            order=order,
            pooling=pooling,
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
            a_seq: torch.Tensor,
            supports: list[torch.Tensor],
            lengths: torch.Tensor | None = None,
            return_sequence: bool = False,
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
        step_embeddings = self.assignment_encoder(a_seq, supports)  # (B, T, d)
        embeddings_pos = self.pos_encoder(step_embeddings)

        src_key_padding_mask = None
        if lengths is not None:
            src_key_padding_mask = self._build_padding_mask(
                lengths=lengths,
                max_len=a_seq.size(1),
            )

        output = self.transformer_encoder(
            embeddings_pos,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, T, d)

        if lengths is not None:
            idx = lengths.to(output.device) - 1
            last_repr = output[torch.arange(output.size(0), device=output.device), idx]
        else:
            last_repr = output[:, -1, :]

        results = [last_repr]

        if return_sequence:
            results.append(output)
        if return_step_embeddings:
            results.append(step_embeddings)

        if len(results) == 1:
            return results[0]
        return tuple(results)

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
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
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
                supports = new_supports if self.addaptadj else self.supports
                x = self.gconv[i](x, supports)
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


class GraphWaveNetBackbone(gwnet):
    """
    Backbone encoder for q-sequences.

    Public input convention:
        q_seq: (B, T, N)

    Public output convention:
        q_repr: (B, repr_dim)

    Internally converts:
        (B, T, N) -> (B, 1, N, T)
    and then runs Graph WaveNet blocks.

    The returned representation corresponds to the LAST temporal position
    after temporal processing, pooled over nodes.
    """

    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        gcn_bool=True,
        addaptadj=True,
        aptinit=None,
        in_dim=1,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
        repr_dim=None,
        node_pooling="mean",
    ):
        super().__init__(
            device=device,
            num_nodes=num_nodes,
            dropout=dropout,
            supports=supports,
            gcn_bool=gcn_bool,
            addaptadj=addaptadj,
            aptinit=aptinit,
            in_dim=in_dim,
            out_dim=1,  # unused in backbone mode, but required by parent
            residual_channels=residual_channels,
            dilation_channels=dilation_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            kernel_size=kernel_size,
            blocks=blocks,
            layers=layers,
        )

        if in_dim != 1:
            raise ValueError(
                f"GraphWaveNetBackbone expects one scalar feature per node per time step, "
                f"so in_dim should be 1. Got in_dim={in_dim}."
            )

        if node_pooling not in {"mean", "max"}:
            raise ValueError(
                f"node_pooling must be 'mean' or 'max', got {node_pooling}"
            )

        self.num_nodes = num_nodes
        self.node_pooling = node_pooling
        self.repr_dim = skip_channels if repr_dim is None else repr_dim

        if self.repr_dim == skip_channels:
            self.readout = nn.Identity()
        else:
            self.readout = nn.Linear(skip_channels, self.repr_dim)

    def _prepare_input(self, q_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_seq: Tensor of shape (B, T, N)

        Returns:
            x: Tensor of shape (B, 1, N, T)
        """
        if q_seq.ndim != 3:
            raise ValueError(
                f"q_seq must have shape (B, T, N), got shape={tuple(q_seq.shape)}"
            )

        B, T, N = q_seq.shape

        if N != self.num_nodes:
            raise ValueError(
                f"Expected q_seq.shape[-1] == {self.num_nodes}, got {N}"
            )

        # (B, T, N) -> (B, N, T) -> (B, 1, N, T)
        x = q_seq.transpose(1, 2).unsqueeze(1).float()
        return x

    def forward_features(self, q_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_seq: Tensor of shape (B, T, N)

        Returns:
            features: Tensor of shape (B, skip_channels, N, T_out)
        """
        x = self._prepare_input(q_seq)  # (B, 1, N, T)

        in_len = x.size(3)

        if in_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - in_len, 0, 0, 0))

        x = self.start_conv(x)
        skip = None

        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x

            filter_out = torch.tanh(self.filter_convs[i](residual))
            gate_out = torch.sigmoid(self.gate_convs[i](residual))
            x = filter_out * gate_out

            s = self.skip_convs[i](x)
            if skip is None:
                skip = s
            else:
                skip = skip[:, :, :, -s.size(3):] + s

            if self.gcn_bool and self.supports is not None:
                supports = new_supports if self.addaptadj else self.supports
                x = self.gconv[i](x, supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        features = F.relu(skip)  # (B, skip_channels, N, T_out)
        return features

    def forward(
        self,
        q_seq: torch.Tensor,
        return_feature_map: bool = False,
        return_temporal_sequence: bool = False,
    ):
        """
        Args:
            q_seq:
                Tensor of shape (B, T, N)

            return_feature_map:
                If True, also return full feature map of shape
                (B, skip_channels, N, T_out)

            return_temporal_sequence:
                If True, also return pooled temporal sequence of shape
                (B, T_out, skip_channels)

        Returns:
            By default:
                q_repr: Tensor of shape (B, repr_dim)

            Optionally also:
                features: Tensor of shape (B, skip_channels, N, T_out)
                seq_repr: Tensor of shape (B, T_out, skip_channels)
        """
        features = self.forward_features(q_seq)  # (B, C, N, T_out)

        # take the LAST time position
        last_features = features[:, :, :, -1]  # (B, C, N)

        # pool over nodes
        if self.node_pooling == "mean":
            q_repr = last_features.mean(dim=-1)  # (B, C)
        elif self.node_pooling == "max":
            q_repr = last_features.max(dim=-1).values  # (B, C)
        else:
            raise ValueError(f"Unknown node_pooling: {self.node_pooling}")

        q_repr = self.readout(q_repr)  # (B, repr_dim)

        results = [q_repr]

        if return_feature_map:
            results.append(features)

        if return_temporal_sequence:
            if self.node_pooling == "mean":
                seq_repr = features.mean(dim=2).transpose(1, 2)  # (B, T_out, C)
            elif self.node_pooling == "max":
                seq_repr = features.max(dim=2).values.transpose(1, 2)  # (B, T_out, C)
            else:
                raise ValueError(f"Unknown node_pooling: {self.node_pooling}")

            results.append(seq_repr)

        if len(results) == 1:
            return results[0]
        return tuple(results)

class STAEformerBackbone(nn.Module):
    """
    Adapter STAEformer zgodny z GraphWaveNetBackbone.forward_features.

    Wejście:
        q: (B, T, N)

    Wyjście:
        features: (B, model_dim, N, T)
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        in_steps: int,
        node_pooling: str = "mean",
        input_embedding_dim: int = 24,
        spatial_embedding_dim: int = 0,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        steps_per_day: int = 288,
    ):
        super().__init__()

        self.encoder = STAEformer(
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=1,
            steps_per_day=steps_per_day,
            input_dim=1,
            output_dim=1,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
            spatial_embedding_dim=spatial_embedding_dim,
            adaptive_embedding_dim=adaptive_embedding_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_mixed_proj=True,
        )

        if self.encoder.model_dim % num_heads != 0:
            raise ValueError(
                f"STAEformer model_dim={self.encoder.model_dim} musi być podzielne "
                f"przez num_heads={num_heads}."
            )

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.node_pooling = node_pooling
        self.out_channels = self.encoder.model_dim

    def forward_features(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (B, T, N)
        return: (B, C, N, T)
        """
        if q.dim() != 3:
            raise ValueError(f"q must have shape (B, T, N), got {tuple(q.shape)}")

        if q.size(1) != self.in_steps:
            raise ValueError(
                f"STAEformerBackbone expected T={self.in_steps}, got T={q.size(1)}"
            )

        if q.size(2) != self.num_nodes:
            raise ValueError(
                f"STAEformerBackbone expected N={self.num_nodes}, got N={q.size(2)}"
            )

        enc = self.encoder
        batch_size = q.size(0)

        # (B, T, N) -> (B, T, N, 1)
        x = q.unsqueeze(-1)

        x = enc.input_proj(x)  # (B, T, N, input_embedding_dim)
        features = [x]

        if enc.spatial_embedding_dim > 0:
            spatial_emb = enc.node_emb.expand(
                batch_size,
                enc.in_steps,
                *enc.node_emb.shape,
            )
            features.append(spatial_emb)

        if enc.adaptive_embedding_dim > 0:
            adaptive_emb = enc.adaptive_embedding.expand(
                batch_size,
                *enc.adaptive_embedding.shape,
            )
            features.append(adaptive_emb)

        x = torch.cat(features, dim=-1)  # (B, T, N, model_dim)

        for attn in enc.attn_layers_t:
            x = attn(x, dim=1)

        for attn in enc.attn_layers_s:
            x = attn(x, dim=2)

        # GraphWaveNetBackbone.forward_features daje (B, C, N, T_out),
        # więc zwracamy taki sam układ.
        return x.permute(0, 3, 2, 1).contiguous()



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


class AGCRNBackbone(nn.Module):
    """
    Adapter AGCRN zgodny z GraphWaveNetBackbone.forward_features
    i STAEformerBackbone.forward_features.

    Wejście:
        q: (B, T, N)

    Wyjście:
        features: (B, hidden_dim, N, T)
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        input_dim: int = 1,
        hidden_dim: int = 64,
        embed_dim: int = 10,
        cheb_k: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
        node_pooling: str = "mean",
    ):
        super().__init__()

        if input_dim != 1:
            raise ValueError(
                "AGCRNBackbone w tym adapterze zakłada q jako (B, T, N), "
                "czyli jedną cechę na węzeł."
            )

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.cheb_k = cheb_k
        self.num_layers = num_layers
        self.node_pooling = node_pooling
        self.out_channels = hidden_dim

        self.node_embeddings = nn.Parameter(torch.empty(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.node_embeddings)

        cells = []
        for layer_idx in range(num_layers):
            layer_input_dim = input_dim if layer_idx == 0 else hidden_dim
            cells.append(
                AGCRNCell(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim,
                    cheb_k=cheb_k,
                    embed_dim=embed_dim,
                )
            )

        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(dropout)

    def forward_features(self, q: torch.Tensor) -> torch.Tensor:
        """
        q:      (B, T, N)
        return: (B, C, N, T)
        """
        if q.dim() != 3:
            raise ValueError(f"q must have shape (B, T, N), got {tuple(q.shape)}")

        if q.size(2) != self.num_nodes:
            raise ValueError(
                f"AGCRNBackbone expected N={self.num_nodes}, got N={q.size(2)}"
            )

        batch_size, seq_len, num_nodes = q.shape

        # (B, T, N) -> (B, T, N, 1)
        current_inputs = q.unsqueeze(-1)

        for layer_idx, cell in enumerate(self.cells):
            h = torch.zeros(
                batch_size,
                num_nodes,
                self.hidden_dim,
                device=q.device,
                dtype=q.dtype,
            )

            outputs = []
            for t in range(seq_len):
                h = cell(
                    current_inputs[:, t, :, :],
                    h,
                    self.node_embeddings,
                )
                outputs.append(h)

            # (B, T, N, hidden_dim)
            current_inputs = torch.stack(outputs, dim=1)

            if layer_idx < self.num_layers - 1:
                current_inputs = self.dropout(current_inputs)

        # (B, T, N, hidden_dim) -> (B, hidden_dim, N, T)
        return current_inputs.permute(0, 3, 2, 1).contiguous()