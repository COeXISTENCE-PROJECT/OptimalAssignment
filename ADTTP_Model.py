import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

from model import GraphWaveNetBackbone, AssignmentEncoder, LSTM_Representation, GRU_Representation, \
    AttentionRepresentation, fuse


class ADTTP(nn.Module):
    """
    Model:
        q -> GraphWaveNetBackbone -> q_rep
        a -> assignmentEncoder -> seq encoder -> a_rep
        q_rep, a_rep -> fuse -> mlp -> q_target

    Input:
        q: (B, T, N, 1)
        a: (B, T, N, N)
        y: (B, 1, N ,1)
    """
    def __init__(self, *,
                 num_nodes: int,
                 q_in_dim: int,
                 a_embedding_size: int = 32,
                 a_hidden_size: int = 64,
                 q_rep_dim: int = 32,
                 fused_dim: int = 64,
                 mlp_hidden_dim: int = 128, target_dim: int = 1,
                 sequence_model: str = "lstm",      # "lstm" / "gru" / "attention"
                 fuse_method: str = "Attention",    # "concatenate" / "Hadamard" / "Attention"
                 dropout: float = 0.1,
                 attention_num_heads: int = 4,
                 attention_ff_dim: int = 128,
                 gwnet_kwargs: dict | None = None,
    ):
        super().__init__()

        gwnet_kwargs = gwnet_kwargs or {}

        #Q encoder
        self.q_encoder = GraphWaveNetBackbone(
            device = "cuda",
            num_nodes = num_nodes,
            in_dim = q_in_dim,
            dropout = dropout,
            **gwnet_kwargs,
        )

        q_backbone_dim = self.q_encoder.end_conv_1.in_channels

        self.q_projector = nn.Sequential(
            nn.Linear(q_backbone_dim, q_rep_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        #A encoder
        self.sequence_model = sequence_model.lower()

        if self.sequence_model == "lstm":
            self.a_encoder = LSTM_Representation(
                n_nodes = num_nodes,
                embedding_size = a_embedding_size,
                hidden_size = a_hidden_size,
                dropout = dropout,
            )

        elif self.sequence_model == "gru":
            self.a_encoder = GRU_Representation(
                n_nodes = num_nodes,
                embedding_size = a_embedding_size,
                hidden_size = a_hidden_size,
                dropout = dropout,
            )
            a_rep_dim = a_hidden_size

        elif self.sequence_model == "attention":
            self.a_encoder = AttentionRepresentation(
                n_nodes=num_nodes,
                embedding_size=a_embedding_size,
                num_heads = attention_num_heads,
                dim_feedforward = attention_ff_dim,
                dropout=dropout,
            )
            a_rep_dim = a_hidden_size

        else:
            raise ValueError(
                f"unknown sequence model: {self.sequence_model}"
            )


        #Fuse function
        self.fuse_method = fuse_method
        self.fuser = fuse(
            dim_Q = q_rep_dim,
            dim_A = a_rep_dim,
            output_dim = fused_dim,
            method = fuse_method
        )

        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, target_dim),
        )


    @staticmethod
    def _gather_last_valid(sequence: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        """
        sequence: (B, T, D)
        lengths: (B,)
        Zwraca ostatni prawidłowy krok czasowy dla każdej sekwencji.
        """
        if lengths is None:
            return sequence[:, -1, :]

        idx = (lengths - 1).clamp_min(0).to(sequence.device)
        batch_idx = torch.arange(sequence.size(0), device=sequence.device)
        return sequence[batch_idx, idx, :]

    def encode_q(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (B, T, N)

        """
        q_features = self.q_encoder.forward_features(q, pool=False)
        q_vec = q_features.mean(dim=(2,3))
        q_repr = self.q_projector(q_vec)

        return q_repr

    def encode_a(self, a: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        a: (B, T, N, N)

        """
        if self.sequence_model == "lstm":
            output, hidden = self.a_encoder(
                a,
                lengths = lengths,
                return_hidden = True,
            )
            #output: (B,T,H)
            a_repr = hidden[-1]

        elif self.sequence_model == "gru":
            output, hidden = self.a_encoder(
                a,
                lengths=lengths,
                return_hidden=True,
            )
            a_repr = hidden[-1]
        elif self.sequence_model == "attention":
            output = self.a_encoder(a, lengths=lengths)  # (B, T, D)
            a_repr = self._gather_last_valid(output, lengths)

        else:
            raise RuntimeError("Unsupported sequence_model.")

        return a_repr


    def forward(self,
                q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
                a: torch.Tensor | None = None,
                lengths: torch.Tensor | None = None,
                return_dict: bool = False,
                ):

        if a is None:
            if not isinstance(q,tuple) or len(q) != 2:
                raise ValueError("Pass either model(q, a, ...) or model((q, a), ...).")

            q, a = q

        q_repr = self.encode_q(q)       # (B, Dq)
        a_repr = self.encode_a(a)       # (B, Da)

        if self.fuse_method == "Attention":
            # fuse.Attention wymaga tensorów 3D: (B, L, D)
            fused = self.fuser(
                q_repr.unsqueeze(1),
                a_repr.unsqueeze(1),
            ).squeeze(1)
        else:
            fused = self.fuser(q_repr, a_repr)


        #making prediction !!!

        pred = self.mlp(fused)

        if return_dict:
            return {
                "lesgoooo"
                "q_repr": q_repr,
                "a_repr": a_repr,
                "fused": fused,
            }

        return pred