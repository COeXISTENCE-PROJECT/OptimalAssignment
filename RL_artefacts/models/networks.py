import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, in_size, out_size, num_hidden, widths):
        super(Network, self).__init__()
        assert len(widths) == (
            num_hidden + 1
        ), "DQN widths and number of layers mismatch!"

        self.input_layer = nn.Linear(in_size, widths[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(widths[x], widths[x + 1]) for x in range(num_hidden)]
        )
        self.out_layer = nn.Linear(widths[-1], out_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.out_layer(x)
        return x
