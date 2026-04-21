import torch
import torch.nn as nn
import torch.optim as optim
import util
from ADTTP_Model import ADTTP


class TrainerADTTP:
    def __init__(
        self,
        scaler,
        in_dim,
        num_nodes,
        nhid,
        dropout,
        lrate,
        wdecay,
        device,
        supports,
        gcn_bool,
        addaptadj,
        aptinit,
        kernel_size=2,
        blocks=4,
        layers=3,
        target_dim=1,
        sequence_model="lstm",
        fuse_method="attention",
        a_embedding_size=32,
        a_hidden_size=32,
        q_rep_dim=32,
        fused_dim=64,
        mlp_hidden_dim=64,
        attention_num_heads=4,
        attention_ff_dim=64,
        loss_name="mae",
        alpha = "1"
    ):
        self.device = device

        gwnet_kwargs = {
            "supports": supports,
            "gcn_bool": gcn_bool,
            "addaptadj": addaptadj,
            "aptinit": aptinit,
            "residual_channels": nhid,
            "dilation_channels": nhid,
            "skip_channels": nhid * 8,
            "end_channels": nhid * 16,
            "kernel_size": kernel_size,
            "blocks": blocks,
            "layers": layers,
        }

        self.model = ADTTP(
            num_nodes=num_nodes,
            supports=supports,
            q_in_dim=in_dim,
            a_embedding_size=a_embedding_size,
            a_hidden_size=a_hidden_size,
            q_rep_dim=q_rep_dim,
            fused_dim=fused_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            target_dim=target_dim,
            sequence_model=sequence_model,
            fuse_method=fuse_method,
            dropout=dropout,
            attention_num_heads=attention_num_heads,
            attention_ff_dim=attention_ff_dim,
            gwnet_kwargs=gwnet_kwargs,
        ).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lrate, weight_decay=wdecay
        )

        self.loss_name = loss_name.lower()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.loss = self._get_loss_fn(self.loss_name)

        self.scaler = scaler
        self.clip = 5

    def _prepare_target(self, pred, real_val):
        real = real_val.to(self.device)

        if real.shape != pred.shape:
            real = real.reshape_as(pred)

        return real

    def _mae(self, pred, real):
        return torch.mean(torch.abs(pred - real))

    def _mape(self, pred, real, eps=1e-8):
        denom = torch.clamp(torch.abs(real), min=eps)
        return torch.mean(torch.abs((pred - real) / denom))

    def _rmse(self, pred, real):
        return torch.sqrt(torch.mean((pred - real) ** 2))

    def _adj_mape(self, pred, real, offset=1.0):
        return torch.mean(torch.abs(pred - real) / (real + offset))

    def _mae_with_adj_mape(self, pred, real):
        return (
            self.alpha * self._mae(pred, real)
            + (1 - self.alpha) * self._adj_mape(pred, real)
        )

    def _flow_cons(self, pred, real, offset=1.0):
        if pred.dim() not in (2, 3):
            raise ValueError(f"Unsupported shape for flow_cons: {tuple(pred.shape)}")

        pred_sum = pred.sum(dim=-1)
        real_sum = real.sum(dim=-1)

        return torch.mean(torch.abs(pred_sum - real_sum) / (torch.abs(real_sum) + offset))

    def _get_loss_fn(self, loss_name):
        if loss_name == "mae":
            return self._mae_with_adj_mape
        elif loss_name == "mape":
            return self._mape
        elif loss_name == "rmse":
            return self._rmse
        elif loss_name == "adj_mape":
            return self._adj_mape
        elif loss_name == "flow_cons":
            return self._flow_cons
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

    def _compute_metrics(self, pred, real, loss_value):
        return {
            "loss": loss_value.item(),
            "mae": self._mae(pred, real).item(),
            "mape": self._mape(pred, real).item(),
            "rmse": self._rmse(pred, real).item(),
            "adj_mape": self._adj_mape(pred, real).item(),
            "flow_cons": self._flow_cons(pred, real).item(),
            "mae_adj_mape_loss": self._mae_with_adj_mape(pred, real).item(),
        }

    def _unpack_batch(self, batch_or_q, a=None, real_val=None, lengths=None):
        if isinstance(batch_or_q, dict) and "x" in batch_or_q and "y" in batch_or_q:
            x = batch_or_q["x"]
            q = x["q"]
            a = x["a"]
            real_val = batch_or_q["y"]

            if lengths is None:
                lengths = x.get("lengths", batch_or_q.get("lengths", None))

            return q, a, real_val, lengths

        return batch_or_q, a, real_val, lengths

    def train(self, batch_or_q, a=None, real_val=None, lengths=None):
        self.model.train()
        self.optimizer.zero_grad()

        q, a, real_val, lengths = self._unpack_batch(
            batch_or_q, a=a, real_val=real_val, lengths=lengths
        )

        q = q.to(self.device)
        a = a.to(self.device)
        real_val = real_val.to(self.device)

        if lengths is not None:
            lengths = lengths.to(self.device)

        pred = self.model(q, a, lengths=lengths)

        if self.scaler is not None:
            pred = self.scaler.inverse_transform(pred)

        real = self._prepare_target(pred, real_val)

        loss = self.loss(pred, real)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        return self._compute_metrics(pred, real, loss)

    @torch.no_grad()
    def eval(self, batch_or_q, a=None, real_val=None, lengths=None):
        self.model.eval()

        q, a, real_val, lengths = self._unpack_batch(
            batch_or_q, a=a, real_val=real_val, lengths=lengths
        )

        q = q.to(self.device)
        a = a.to(self.device)
        real_val = real_val.to(self.device)

        if lengths is not None:
            lengths = lengths.to(self.device)

        pred = self.model(q, a, lengths=lengths)

        if self.scaler is not None:
            pred = self.scaler.inverse_transform(pred)

        real = self._prepare_target(pred, real_val)

        loss = self.loss(pred, real)

        return self._compute_metrics(pred, real, loss)