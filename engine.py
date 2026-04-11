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
        kernel_size,
        blocks,
        layers,
        target_dim = 1,
        sequence_model = "lstm",
        fuse_method = "Attention",
        a_embedding_size = 32,
        a_hidden_size = 64,
        q_rep_dim = 32,
        fused_dim = 64,
        mlp_hidden_dim = 128,
        attention_num_heads = 4,
        attention_ff_dim = 128
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
            supports = supports,  
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
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def _prepare_target(self, pred, real_val):
        real = real_val.to(self.device)

        if real.shape != pred.shape:
            real = real.reshape_as(pred)

        return real

    def _unpack_batch(self, batch_or_q, a=None, real_val=None, lengths=None):
        # nowy format z DataLoadera:
        # {"x": {"q": ..., "a": ...}, "y": ...}
        if isinstance(batch_or_q, dict) and "x" in batch_or_q and "y" in batch_or_q:
            x = batch_or_q["x"]
            q = x["q"]
            a = x["a"]
            real_val = batch_or_q["y"]

            if lengths is None:
                lengths = x.get("lengths", batch_or_q.get("lengths", None))

            return q, a, real_val, lengths

        # stary format: train(q, a, real_val, lengths)
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

        loss = self.loss(pred, real, 0.0)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        mape = util.masked_mape(pred, real, 0.0).item()
        rmse = util.masked_rmse(pred, real, 0.0).item()

        return loss.item(), mape, rmse

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

        loss = self.loss(pred, real, 0.0)
        mape = util.masked_mape(pred, real, 0.0).item()
        rmse = util.masked_rmse(pred, real, 0.0).item()

        return loss.item(), mape, rmse

