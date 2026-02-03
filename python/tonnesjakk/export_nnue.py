"""
Eksporter NNUE-vekter til JSON for Rust.
"""

import json
import torch
from pathlib import Path


def export_to_json(model_path: str, output_path: str):
    """
    Eksporter en trent NNUE-modell til JSON som Rust kan lese.

    Format:
    {
        "hidden1": 64,
        "hidden2": 32,
        "weights": {
            "fc1_weight": [[...], ...],  # shape: [hidden1, 144]
            "fc1_bias": [...],           # shape: [hidden1]
            "fc2_weight": [[...], ...],  # shape: [hidden2, hidden1]
            "fc2_bias": [...],           # shape: [hidden2]
            "fc3_weight": [[...], ...],  # shape: [1, hidden2]
            "fc3_bias": [...]            # shape: [1]
        }
    }
    """
    from nnue import TonnesjakkNNUE

    # Last modell
    model = TonnesjakkNNUE()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Hent vekter
    state_dict = model.state_dict()

    # Konverter til lister
    weights = {}
    for name, tensor in state_dict.items():
        # PyTorch navngir lagene: net.0.weight, net.0.bias, net.2.weight, etc.
        # Konverter til mer lesbare navn
        if "0.weight" in name:
            weights["fc1_weight"] = tensor.tolist()
        elif "0.bias" in name:
            weights["fc1_bias"] = tensor.tolist()
        elif "2.weight" in name:
            weights["fc2_weight"] = tensor.tolist()
        elif "2.bias" in name:
            weights["fc2_bias"] = tensor.tolist()
        elif "4.weight" in name:
            weights["fc3_weight"] = tensor.tolist()
        elif "4.bias" in name:
            weights["fc3_bias"] = tensor.tolist()

    # Bestem arkitektur fra vekter
    hidden1 = len(weights["fc1_bias"])
    hidden2 = len(weights["fc2_bias"])

    output = {
        "hidden1": hidden1,
        "hidden2": hidden2,
        "weights": weights
    }

    # Skriv til fil
    with open(output_path, "w") as f:
        json.dump(output, f)

    print(f"Eksportert modell til {output_path}")
    print(f"  Arkitektur: 144 -> {hidden1} -> {hidden2} -> 1")
    print(f"  Filstorrelse: {Path(output_path).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Bruk: python export_nnue.py <model.pt> [output.json]")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else model_path.replace(".pt", ".json")

    export_to_json(model_path, output_path)
