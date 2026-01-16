# src/models/model.py
import torch
import torch.nn as nn


class TrajectoryModel(nn.Module):
    """
    Flexible trajectory prediction model supporting multiple RNN types.
    
    Supported architectures:
      - 'lstm'              (standard LSTM)
      - 'bilstm'            (bidirectional LSTM)
      - 'gru'               (standard GRU)
      - 'bilstm_attention'  (bidirectional LSTM + simple attention)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        architecture: str = "lstm",
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.architecture = architecture.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Recurrent layer selection
        if self.architecture == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            rnn_output_dim = hidden_size

        elif self.architecture == "bilstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            rnn_output_dim = hidden_size * 2

        elif self.architecture == "gru":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            rnn_output_dim = hidden_size

        elif self.architecture == "bilstm_attention":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            rnn_output_dim = hidden_size * 2
            # Simple attention layer
            self.attention = nn.Sequential(
                nn.Linear(rnn_output_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
        else:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                "Supported: 'lstm', 'bilstm', 'gru', 'bilstm_attention'"
            )

        # Final projection layer
        self.fc = nn.Linear(rnn_output_dim, output_size)

        # Optional dropout after RNN
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        x: shape (batch_size, seq_len, input_size)
        returns: shape (batch_size, output_size)
        """
        # RNN forward pass
        if self.architecture == "bilstm_attention":
            # Full sequence output + attention
            rnn_out, _ = self.rnn(x)                        # (batch, seq_len, hidden*2)
            attn_scores = self.attention(rnn_out)           # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = torch.sum(rnn_out * attn_weights, dim=1)  # (batch, hidden*2)
            out = self.fc(self.dropout_layer(context))
        else:
            # Only last hidden state
            _, (h_n, _) = self.rnn(x)                       # h_n: (num_layers * num_directions, batch, hidden)
            
            # Select last layer hidden state
            if self.architecture in ["bilstm", "bilstm_attention"]:
                # Bidirectional → concat forward & backward from last layer
                h_n = torch.cat((h_n[-2], h_n[-1]), dim=-1)     # (batch, hidden*2)
            else:
                h_n = h_n[-1]                                   # (batch, hidden)

            out = self.fc(self.dropout_layer(h_n))

        return out
