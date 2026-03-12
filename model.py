import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



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
            self. K = nn.Linear(dim_A, output_dim) #keys
            self. Q = nn.Linear(dim_Q, output_dim) #queries

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

            output, _ = self.attention(query = queries, kay = keys, values = values)



        return output




class recurrent(nn.Module):
    """
    Implementation of a recurrent module for processing Assignment matrices.
    Handles variable-length sequences using padding and packing.
    """

    def __init__(self, input_size, embedding_size, hidden_size, dropout, num_layers):
        super(recurrent, self).__init__()

        # Initial projection of the flattened N x N matrix into an embedding space
        self.assignment_encoder = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU()
        )

        # GRU module with batch_first=True (Batch, Seq, Feature)
        self.gru = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, A_seq, lengths):
        """
        Args:
            A_seq (Tensor): Input sequence of shape (batch_size, T, N, N)
            lengths (Tensor): Actual lengths (k) for each sequence in the batch

        Returns:
            output (Tensor): Full sequence representations (batch_size, T, hidden_size)
            final_hidden (Tensor): Aggregated representation from the last real step (batch_size, hidden_size)
        """
        batch_size, T, N, _ = A_seq.shape

        # Flatten the N x N matrices into vectors (shape: batch_size, T, N*N)
        A_seq_flat = A_seq.view(batch_size, T, -1)

        # Encode each assignment matrix in the sequence
        # Shape: (batch_size, T, embedding_size)
        raw_embeddings = self.assignment_encoder(A_seq_flat)

        # Pack the padded sequence to prevent GRU from processing zero-padding
        # lengths must be on CPU for this utility function
        lengths_cpu = lengths.cpu() if isinstance(lengths, torch.Tensor) else lengths

        embeddings = pack_padded_sequence(
            raw_embeddings,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False
        )

        # Forward pass through GRU
        # packed_output: Internal packed representation
        # hidden: Contains the state after the last 'valid' step for each layer
        packed_output, hidden = self.gru(embeddings)

        # Unpack the sequence back to a regular padded tensor
        # Shape: (batch_size, T, hidden_size)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=T)

        # Extract the hidden state of the last layer
        # from step 'k' into this hidden tensor.
        # Shape: (batch_size, hidden_size)
        final_hidden = hidden[-1]

        return output, final_hidden


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
                    x = self.gconv[i](x, new_supports)
                else:
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


