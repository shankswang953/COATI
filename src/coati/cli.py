"""Command-line entry point sharing the Python API."""
import argparse
import json
from pathlib import Path
from . import datasets, fit, Config, TemporalData

def main(argv=None):
    parser=argparse.ArgumentParser(description="COATI trajectory inference")
    sub=parser.add_subparsers(dest="command",required=True)
    toy=sub.add_parser("toy",help="Run a small synthetic example")
    toy.add_argument("--dataset",choices=["gaussian","split","paired"],default="gaussian")
    toy.add_argument("--niters",type=int,default=200)
    toy.add_argument("--seed",type=int,default=0)
    toy.add_argument("--unbalanced",action="store_true")
    toy.add_argument("--device",default="cpu")
    toy.add_argument("--output",required=True)
    train=sub.add_parser("fit",help="Train from explicitly ordered NPZ snapshots")
    train.add_argument("input")
    train.add_argument("--secondary")
    train.add_argument("--keys",nargs="+",required=True)
    train.add_argument("--times",nargs="+",type=float,required=True)
    train.add_argument("--config",required=True)
    train.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    if args.command=="toy":
        data=getattr(datasets,args.dataset)(seed=args.seed)
        cfg=Config(niters=args.niters,seed=args.seed,device=args.device,sync_loss=args.dataset=="paired",
                   sync_weight=.35 if args.dataset=="paired" else 0.,unbalancedModel=args.unbalanced,
                   mass_loss=args.unbalanced,mass_coefficient=1. if args.unbalanced else 0.,
                   alpha_growth=1. if args.unbalanced else 0.)
    else:
        data=TemporalData.from_npz(args.input,keys=args.keys,times=args.times,secondary_path=args.secondary)
        cfg=Config.load(args.config)
    result=fit(data,cfg,output_dir=args.output)
    scores=result.evaluate()
    (result.output_dir/"metrics.json").write_text(json.dumps(scores,indent=2)+"\n")
    result.plot().figure.savefig(result.output_dir/"trajectories.png",dpi=150,bbox_inches="tight")
    print(json.dumps(scores,indent=2))
    print(f"Results: {result.output_dir}")

if __name__=="__main__": main()
