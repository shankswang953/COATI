"""Run after `python -m pip install -e .`: python examples/run_toy.py --dataset paired --output outputs/paired"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key,"1")
from coati.cli import main
if __name__ == "__main__":
    import sys
    main(["toy", *sys.argv[1:]])
