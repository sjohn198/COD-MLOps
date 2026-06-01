import torch.nn as nn

class WinPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(in_features=9, out_features=64)
        self.hidden_layer = nn.Linear(in_features=64, out_features=32)
        self.out_layer = nn.Linear(in_features=32, out_features=2)

        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layer(self.activation(out))
        logits = self.out_layer(self.activation(out))
        return logits