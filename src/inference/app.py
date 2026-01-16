# src/inference/app.py
"""
Simple Gradio interface for ship trajectory prediction demo
"""

import gradio as gr
import torch
import yaml
import numpy as np

from src.models.model import get_model_class
from src.utils.seed import set_seed


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(arch_name, config_path="config.yaml"):
    config = load_config(config_path)
    set_seed(config["seed"])

    ModelClass = get_model_class(arch_name)
    model = ModelClass(
        input_size=len(config["feature_cols"]),
        hidden_size=config["hidden_size"],
        output_size=len(config["target_cols"])
    )

    # Load the latest/best weights for this architecture
    # (you should adjust path according to where you save models)
    model_path = f"artifacts/model_{arch_name}.pth"
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(model_path))

    model.eval()
    return model


def predict(arch_name, input_sequence):
    """
    input_sequence: string like "lat1,lon1,sog1,cog1;lat2,lon2,...;"
    Returns predicted future positions
    """
    try:
        # Parse input
        points = [list(map(float, p.split(","))) for p in input_sequence.strip().split(";") if p.strip()]
        if len(points) == 0:
            return "Error: empty input"

        input_tensor = torch.tensor([points], dtype=torch.float32)  # shape: (1, seq_len, features)

        model = load_model(arch_name)
        with torch.no_grad():
            prediction = model(input_tensor).squeeze(0).numpy()

        # Format output nicely
        output = []
        for i, (lat, lon) in enumerate(prediction):
            output.append(f"Step {i+1}: lat={lat:.4f}, lon={lon:.4f}")

        return "\n".join(output)

    except Exception as e:
        return f"Error: {str(e)}"


# Gradio interface
def create_interface():
    config = load_config()
    arch_choices = ["lstm", "bilstm", "gru", "bilstm_attention"]

    iface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Dropdown(choices=arch_choices, label="Model Architecture", value="lstm"),
            gr.Textbox(
                lines=5,
                label="Input sequence (format: lat1,lon1,sog1,cog1 ; lat2,lon2,...)",
                placeholder="10.12,76.25,12.5,180; 10.13,76.26,12.6,185; ..."
            )
        ],
        outputs=gr.Textbox(label="Predicted future positions"),
        title="Ship Trajectory Prediction Demo",
        description="Predict future lat/lon positions using different RNN models",
        examples=[
            ["lstm", "10.12,76.25,12.5,180; 10.13,76.26,12.6,185; 10.14,76.27,12.7,190"],
            ["bilstm", "9.85,76.30,15.0,270; 9.84,76.29,14.8,265"]
        ],
        allow_flagging="never"
    )
    return iface


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
