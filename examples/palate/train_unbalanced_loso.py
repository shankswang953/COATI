"""Unbalanced palate LOSO entry point."""
from .train_loso import main

if __name__ == "__main__":
    main(default_config="unbalanced_loso.json", run_name="palate-unbalanced-loso")
