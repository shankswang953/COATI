"""Train the 20,000-step balanced RNA-only gastrulation baseline."""
from .train import main


if __name__ == "__main__":
    main(default_config="balanced_rna_only.json", run_name="gastrulation-balanced-rna-only")
