"""Train the archived unbalanced full-data gastrulation configuration."""
from .train import main


if __name__ == "__main__":
    main(default_config="unbalanced.json", run_name="gastrulation-unbalanced")
