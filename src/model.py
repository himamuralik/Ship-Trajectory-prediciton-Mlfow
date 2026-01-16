import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Attention, Input, Reshape
from tensorflow.keras.models import Sequential


def build_model(arch, input_shape, output_shape=(6, 2), hidden_units=64):
    model = Sequential()
    model.add(Input(shape=input_shape))

    if arch == "lstm":
        model.add(LSTM(hidden_units, return_sequences=False))
    elif arch == "gru":
        model.add(GRU(hidden_units, return_sequences=False))
    elif arch == "bilstm":
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=False)))
    elif arch == "bilstm_attention":
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
        model.add(Attention())
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model.add(Dense(output_shape[0] * output_shape[1]))
    model.add(Reshape(output_shape))

    model.compile(optimizer='adam', loss='mse')
    return model
