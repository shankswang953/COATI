"""Read-only numerical comparison with a user-specified original TraInf checkout.

python scripts/verify_equivalence.py --original /path/to/TraInf --output outputs/equivalence
The original is never written, including bytecode caches. No original code is patched on disk.
"""
import argparse, contextlib, hashlib, importlib, json, sys, types
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from coati import Config, datasets, fit

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--original",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    cli=parser.parse_args();cli.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((Path(__file__).resolve().parents[1]/"provenance/core_sha256.json").read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((cli.original/"src"/name).read_bytes()).hexdigest()==digest, f"Original changed since copy: {name}"
    original_pkg=types.ModuleType("src");original_pkg.__path__=[str(cli.original/"src")];sys.modules["src"]=original_pkg
    unused_args=types.ModuleType("Args");unused_args.get_args=lambda:None;sys.modules["Args"]=unused_args
    original=importlib.import_module("src.Training")
    # Force CPU in this comparison harness, because the original auto-selects MPS on macOS.
    original._resolve_device=lambda args:torch.device("cpu")
    report=[]
    for name,kind,extra in [("balanced","gaussian",{}),("kernel_sync","paired",dict(sync_loss=True,sync_weight=.35)),
                            ("unbalanced","split",dict(unbalancedModel=True,alpha_growth=1.,mass_loss=True,mass_coefficient=1.)),
                            ("diagnostic_100","paired",dict(sync_loss=True,sync_weight=.35,sync_density_loss=True)),
                            ("adaptive","gaussian",dict(adaptive_lambda=True,extra={"dual_warmup":0,"dual_update_freq":1}))]:
        data=getattr(datasets,kind)(n=40)
        steps=100 if name=="diagnostic_100" else 4
        cfg=Config(niters=steps,num_samples=32,floor_n_sample=32,median_n_sample=40,hidden_dim=16,**extra)
        new=fit(data,cfg,output_dir=cli.output/(name+"-coati"))
        directory=(cli.output/(name+"-reference")).resolve();directory.mkdir()
        keys=data.save_npz(directory/"inputs");(directory/"checkpoints").mkdir()
        values=cfg.to_dict();values.update(data_path=str(directory/"inputs/primary.npz"),sync_data_path=str(directory/"inputs/secondary.npz"),
            time_labels=keys,time_points=data.times.tolist(),otdim=2,train_dir=str(directory/"checkpoints"),results_dir=str(directory/"logs"))
        with (directory/"training.log").open("w") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            old=original.run_training(types.SimpleNamespace(**values))
        differences=[float((value-old["func"].state_dict()[key]).abs().max()) for key,value in new.model.state_dict().items()]
        delta=max(differences)
        assert delta==0., f"{name}: weights differ by {delta}"
        report.append(dict(mode=name,iterations=steps,max_parameter_difference=delta))
    for name,digest in manifest.items():
        assert hashlib.sha256((cli.original/"src"/name).read_bytes()).hexdigest()==digest
    (cli.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
