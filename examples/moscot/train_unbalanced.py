"""Unbalanced full-data pancreas entry point."""
from .train import main

if __name__ == "__main__":
    main(default_config="unbalanced.json", run_name="moscot-unbalanced")
