# src/models/components.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Simple additive attention (can be used in bilstm_attention)
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size)
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)      # (batch, seq_len, hidden)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))   # (batch, seq_len, hidden)
        energy = energy.permute(0, 2, 1)                           # (batch, hidden, seq_len)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)              # (batch, 1, hidden)
        attention = torch.bmm(v, energy).squeeze(1)                # (batch, seq_len)
        return F.softmax(attention, dim=1)


class LuongAttention(nn.Module):
    """Luong-style attention – optional more advanced variant"""
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, keys):
        # query: (batch, hidden)
        # keys: (batch, seq_len, hidden)
        scores = torch.bmm(keys, self.Wa(query).unsqueeze(2)).squeeze(2)  # (batch, seq_len)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)   # (batch, hidden)
        return context, attn_weights
