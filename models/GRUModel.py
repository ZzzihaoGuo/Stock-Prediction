import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=3, num_layers=3):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x[..., -1] = x[..., -1]/222
        x[..., -2] = x[..., -2]/1000
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
