from torch import nn, optim
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

class WinPredictor(nn.module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(in_features=10, out_features=64)
        self.hidden_layer = nn.Linear(in_features=64, out_features=32)
        self.out_layer = nn.Linear(in_features=32, out_features=2)

        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layer(self.activation(out))
        logits = self.output_layer(self.activation(out))
        return logits
    
if __name__ == "__main__":
    df = pd.read_parquet("dataset/baseball.parquet")

    x_dataset = df[["balls", "inning_topbot", "strikes", "on_3b", "on_2b", "on_1b", "outs_when_up", "inning", "home_score", "away_score"]]
    y_dataset = df["winner"]

    x_tensor = torch.tensor(x_dataset.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_dataset.values, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(x_tensor, y_tensor)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = WinPredictor().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 25