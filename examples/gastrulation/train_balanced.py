"""Train the archived balanced full-data gastrulation configuration."""
from .train import main


if __name__ == "__main__":
    main(default_config="balanced.json", run_name="gastrulation-balanced")
