from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules import (
    GraphWaveNetBackbone,
    LSTM_Representation,
    GRU_Representation,
    AttentionRepresentation,
    fuse,
)
from utilities import (
    _ensure_batch_sequence,
    _normalize_lengths
)

from Modules import STAEformerBackbone, AGCRNBackbone


class GenTTP(nn.Module):
    """
    ADTTP:
        q -> Graph neural network (Graph-WaveNet, STAEformer, AGCRN) -> q_repr
        a -> sequence encoder (LSTM / GRU / Attention) -> a_repr
        q_repr, a_repr -> fuse -> MLP -> prediction of next q

    Supported inputs:
        - dict: {"q": q, "a": a, "lengths": lengths?, "supports": supports?}
        - pair of tensors: model(q, a, ...)
        - tuple: model((q, a), ...)

    Expected shapes:
        q: (B, q_len, N) or (q_len, N)
        a: (B, a_len, N) or (a_len, N)

    Output:
        - target_dim == 1: (B, N) or (N,)
        - target_dim > 1:  (B, target_dim, N) or (target_dim, N)
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        supports: list[torch.Tensor] | None = None,
        q_in_dim: int = 1,
        a_embedding_size: int = 32,
        a_hidden_size: int = 32,
        q_rep_dim: int = 32,
        fused_dim: int = 64,
        mlp_hidden_dim: int = 64,
        target_dim: int = 1,
        sequence_model: str = "lstm",      # "lstm_concat" / "gru" / "attention"
        fuse_method: str = "concatenate",    # "concatenate" / "Hadamard" / "Attention"
        dropout: float = 0.1,
        device: str = "cpu",
        attention_num_heads: int = 4,
        attention_ff_dim: int = 64,
        q_backbone: str = "gwnet",          # "gwnet" / "staeformer" / "agcrn"
        q_len: int | None = None,
        gwnet_kwargs: dict | None = None,
        staeformer_kwargs: dict | None = None,
        agcrn_kwargs: dict | None = None,
        q_node_pooling: str = "mean",       # "max"
        default_use_gate=True,
        default_hard_gate=False,
        default_gate_threshold=0.5,
    ):
        super().__init__()

        gwnet_kwargs = dict(gwnet_kwargs or {})
        if "supports" not in gwnet_kwargs and supports is not None:
            gwnet_kwargs["supports"] = supports

        self.num_nodes = num_nodes
        self.q_in_dim = q_in_dim
        self.target_dim = target_dim
        self.supports = supports
        self.sequence_model = sequence_model.lower()

        self.default_use_gate = default_use_gate
        self.default_hard_gate = default_hard_gate
        self.default_gate_threshold = default_gate_threshold

        canonical_fuse_method = {
            "concatenate": "concatenate",
            "hadamard": "Hadamard",
            "attention": "Attention",
        }.get(fuse_method.lower(), fuse_method)

        self.fuse_method = canonical_fuse_method

        # q encoder
        if q_in_dim != 1:
            raise ValueError(
                "q dimension must be 1, change q_in_dim"
            )

        self.q_backbone = q_backbone.lower()

        if self.q_backbone in {"gwnet", "graphwavenet"}:
            self.q_encoder = GraphWaveNetBackbone(
                device=device,
                num_nodes=num_nodes,
                in_dim=1,
                dropout=dropout,
                **gwnet_kwargs,
            )

            self.q_backbone_dim = self.q_encoder.end_conv_1.in_channels

        elif self.q_backbone in {"staeformer", "steaformer"}:
            staeformer_kwargs = dict(staeformer_kwargs or {})

            if q_len is None:
                q_len = staeformer_kwargs.pop("in_steps", None)

            if q_len is None:
                raise ValueError(
                    "STAEformer backbone requires fixed q_len"
                )

            self.q_encoder = STAEformerBackbone(
                num_nodes=num_nodes,
                in_steps=q_len,
                node_pooling=q_node_pooling,
                dropout=dropout,
                **staeformer_kwargs,
            )

            self.q_backbone_dim = self.q_encoder.out_channels

        elif self.q_backbone in {"agcrn", "adaptive_gcrn"}:
            agcrn_kwargs = dict(agcrn_kwargs or {})

            self.q_encoder = AGCRNBackbone(
                num_nodes=num_nodes,
                input_dim=1,
                dropout=dropout,
                node_pooling=q_node_pooling,
                **agcrn_kwargs,
            )

            self.q_backbone_dim = self.q_encoder.out_channels

        else:
            raise ValueError(f"Unknown q_backbone: {q_backbone}")

        self.q_projector = nn.Sequential(
            nn.Linear(self.q_backbone_dim, q_rep_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Assignments encoder

        if self.sequence_model == "lstm":
            self.a_encoder = LSTM_Representation(
                n_nodes=num_nodes,
                embedding_size=a_embedding_size,
                hidden_size=a_hidden_size,
                dropout=dropout,
            )
            a_rep_dim = a_hidden_size

        elif self.sequence_model == "gru":
            self.a_encoder = GRU_Representation(
                n_nodes=num_nodes,
                embedding_size=a_embedding_size,
                hidden_size=a_hidden_size,
                dropout=dropout,
            )
            a_rep_dim = a_hidden_size

        elif self.sequence_model == "attention":
            self.a_encoder = AttentionRepresentation(
                n_nodes=num_nodes,
                embedding_size=a_embedding_size,
                num_heads=attention_num_heads,
                dim_feedforward=attention_ff_dim,
                dropout=dropout,
            )
            a_rep_dim = a_embedding_size

        else:
            raise ValueError(f"Unknown sequence model: {sequence_model}")

        self.fuser = fuse(
            dim_Q=q_rep_dim,
            dim_A=a_rep_dim,
            output_dim=fused_dim,
            method=canonical_fuse_method,
        )

        self.reg_head = nn.Sequential(
            nn.Linear(fused_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_nodes * target_dim),
        )

        self.gate_head = nn.Sequential(
            nn.Linear(fused_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_nodes),
        )


    def _resolve_supports(
        self,
        supports: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        supports = supports if supports is not None else self.supports
        if supports is None:
            raise ValueError(
                "Required support matrix"
            )
        return supports


    # Initialisation of the last layers close to zero in order to reduce impact of large number of empty nodes

    def _init_output_heads(self):
        reg_last = self.reg_head[-1]
        gate_last = self.gate_head[-1]

        if not isinstance(reg_last, nn.Linear):
            raise TypeError("Expected last layer of reg_head to be nn.Linear")
        if not isinstance(gate_last, nn.Linear):
            raise TypeError("Expected last layer of gate_head to be nn.Linear")

        # low weights for regression head
        nn.init.normal_(reg_last.weight, mean=0.0, std=1e-3)
        nn.init.constant_(reg_last.bias, -4.0)

        # neutral start for gate head
        nn.init.normal_(gate_last.weight, mean=0.0, std=1e-3)
        nn.init.constant_(gate_last.bias, -1.0)

    def encode_q(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (B, T, N)
        """
        q_features = self.q_encoder.forward_features(q)  # (B, C, N, T_out)
        q_last = q_features[:, :, :, -1]  # (B, C, N)

        if getattr(self.q_encoder, "node_pooling", "mean") == "max":
            q_vec = q_last.max(dim=-1).values
        else:
            q_vec = q_last.mean(dim=-1)

        q_repr = self.q_projector(q_vec)
        return q_repr

    def encode_a(
        self,
        a: torch.Tensor,
        lengths: torch.Tensor | None = None,
        supports: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:

        """
        a: (B, T, N)
        """

        supports = self._resolve_supports(supports)

        if self.sequence_model == "lstm":
            a_repr = self.a_encoder(
                a_seq=a,
                supports=supports,
                lengths=lengths,
            )

        elif self.sequence_model == "gru":
            a_repr = self.a_encoder(
                a_seq=a,
                support=supports,
                lengths=lengths,
            )

        elif self.sequence_model == "attention":
            a_repr = self.a_encoder(
                a_seq=a,
                supports=supports,
                lengths=lengths,
            )

        else:
            raise RuntimeError("Unsupported sequence_model.")

        return a_repr

    def forward(
            self,
            q: torch.Tensor,
            a: torch.Tensor,
            lengths: torch.Tensor | None = None,
            supports: list[torch.Tensor] | None = None,
            return_dict: bool = False,
            use_gate: bool | None = None,
            hard_gate: bool | None = None,
            gate_threshold: float | None = None,
    ):

        if use_gate is None:
            use_gate = self.default_use_gate
        if hard_gate is None:
            hard_gate = self.default_hard_gate
        if gate_threshold is None:
            gate_threshold = self.default_gate_threshold

        q, q_was_unbatched = _ensure_batch_sequence(q, "q")
        a, a_was_unbatched = _ensure_batch_sequence(a, "a")

        if q.size(0) != a.size(0):
            raise ValueError(
                f"Batch size mismatch: q has batch={q.size(0)}, a has batch={a.size(0)}"
            )

        lengths = _normalize_lengths(
            lengths,
            batch_size=a.size(0),
            device=a.device,
        )

        q_repr = self.encode_q(q)
        a_repr = self.encode_a(a, lengths=lengths, supports=supports)
        fused = self.fuser(q_repr, a_repr)

        reg_raw = self.reg_head(fused)
        gate_logits = self.gate_head(fused)
        gate_prob = torch.sigmoid(gate_logits)

        if self.target_dim == 1:
            reg_raw = reg_raw.view(reg_raw.size(0), self.num_nodes)
            reg_pred = F.softplus(reg_raw)

            if not use_gate:
                pred = reg_pred
            elif hard_gate:
                gate_mask = (gate_prob >= gate_threshold).float()
                pred = gate_mask * reg_pred
            else:
                pred = gate_prob * reg_pred

        else:
            reg_raw = reg_raw.view(reg_raw.size(0), self.target_dim, self.num_nodes)
            reg_pred = F.softplus(reg_raw)

            if not use_gate:
                pred = reg_pred
            elif hard_gate:
                gate_mask = (gate_prob >= gate_threshold).float().unsqueeze(1)
                pred = gate_mask * reg_pred
            else:
                pred = gate_prob.unsqueeze(1) * reg_pred

        was_unbatched = q_was_unbatched and a_was_unbatched

        if was_unbatched:
            pred = pred.squeeze(0)
            reg_raw = reg_raw.squeeze(0)
            reg_pred = reg_pred.squeeze(0)
            gate_logits = gate_logits.squeeze(0)
            gate_prob = gate_prob.squeeze(0)
            q_repr = q_repr.squeeze(0)
            a_repr = a_repr.squeeze(0)
            fused = fused.squeeze(0)

        if return_dict:
            return {
                "pred": pred,
                "reg_raw": reg_raw,
                "reg_pred": reg_pred,
                "gate_logits": gate_logits,
                "gate_prob": gate_prob,
                "q_repr": q_repr,
                "a_repr": a_repr,
                "fused": fused,
            }

        return pred
