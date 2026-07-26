import torch.nn as nn
import torch

class WinPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 32
        self.layers = 1

        self.gru = nn.GRU(
            input_size=9,
            hidden_size=self.hidden_size,
            num_layers=self.layers,
            batch_first=True
        )

        self.final = nn.Linear(self.hidden_size, 2)

        # super().__init__()

        # self.input_layer = nn.Linear(in_features=9, out_features=128)
        # self.hidden_layer = nn.Linear(in_features=128, out_features=64)
        # self.out_layer = nn.Linear(in_features=64, out_features=2)

        # self.activation = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.randn(self.layers, batch_size, self.hidden_size, device=x.device)

        out, hn = self.gru(x, h0)

        final_time_step = out[:, -1, :]
        logits = self.final(final_time_step)


        # out = self.input_layer(x)
        # out = self.hidden_layer(self.activation(out))
        # logits = self.out_layer(self.activation(out))
        return logits