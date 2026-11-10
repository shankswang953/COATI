"""Unbalanced full-data palate entry point."""
from .train import main

if __name__ == "__main__":
    main(default_config="unbalanced.json", run_name="palate-unbalanced")
