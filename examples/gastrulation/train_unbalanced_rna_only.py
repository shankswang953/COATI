"""Train the 20,000-step unbalanced RNA-only gastrulation baseline."""
from .train import main


if __name__ == "__main__":
    main(default_config="unbalanced_rna_only.json", run_name="gastrulation-unbalanced-rna-only")
