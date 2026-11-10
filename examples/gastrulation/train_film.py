"""Fit the COATI gastrulation FiLM map before trajectory training."""
import argparse
from pathlib import Path

import torch

from .film import train_film_map


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA / "gastrulation_embeddings.h5ad")
    parser.add_argument("--primal-norm", type=Path, default=DATA / "primal_norm_params.pt")
    parser.add_argument("--secondary-norm", type=Path, default=DATA / "secondary_norm_params.pt")
    parser.add_argument("--output", type=Path, default=Path("outputs/gastrulation-film/T_FiLM.pt"))
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint, _ = train_film_map(
        args.data, args.output, args.primal_norm, args.secondary_norm,
        device=device, epochs=args.epochs,
    )
    print("FiLM checkpoint:", args.output)
    print("Selected epoch:", checkpoint["meta"]["epoch"])


if __name__ == "__main__":
    main()
