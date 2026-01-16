# src/models/model.py
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn.squeeze(0))


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        hn = hn.transpose(0, 1).contiguous().view(hn.size(1), -1)
        return self.fc(hn)


class GRUModel(nn.Module):
    # similar implementation...


class BiLSTMAttentionModel(nn.Module):
    # implement attention layer...


def get_model_class(arch_name: str):
    mapping = {
        "lstm": LSTMModel,
        "bilstm": BiLSTMModel,
        "gru": GRUModel,
        "bilstm_attention": BiLSTMAttentionModel,
    }
    if arch_name not in mapping:
        raise ValueError(f"No implementation for {arch_name}")
    return mapping[arch_name]
