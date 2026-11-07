"""Run a synthetic example from the COATI repository, outside the installed package."""
import argparse
import json
from pathlib import Path
import toy_data as datasets
from coati import Config, MapConfig, train_vector_field


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["gaussian","split","paired"],default="gaussian")
    parser.add_argument("--niters",type=int,default=200)
    parser.add_argument("--seed",type=int,default=0)
    mass=parser.add_mutually_exclusive_group()
    mass.add_argument("--unbalanced",dest="balanced",action="store_false")
    mass.add_argument("--balanced",dest="balanced",action="store_true")
    parser.set_defaults(balanced=True)
    parser.add_argument("--multimodal",action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output",required=True)
    parser.add_argument("--map-type", choices=["film", "mlp", "kernel", "accurate_kernel"], default="film")
    parser.add_argument("--map-epochs", type=int, default=300)
    args=parser.parse_args(argv)
    data=getattr(datasets,args.dataset)(seed=args.seed)
    multimodal=args.dataset=="paired" if args.multimodal is None else args.multimodal
    cfg=Config(iterations=args.niters,seed=args.seed,device=args.device,
               multimodal=multimodal,balanced=args.balanced,map_type=args.map_type)
    map_cfg = MapConfig(epochs=args.map_epochs) if cfg.multimodal and args.map_type in {"film", "mlp"} else None
    result=train_vector_field(data,cfg,output_dir=args.output,map_config=map_cfg)
    if result.map_report is not None:
        print(f"Map validation: {result.map_report['status']}. Report: {result.output_dir / 'map/report.txt'}")
    scores=result.evaluate()
    (result.output_dir/"metrics.json").write_text(json.dumps(scores,indent=2)+"\n")
    result.plot_trajectory().figure.savefig(result.output_dir/"trajectories.png",dpi=150,bbox_inches="tight")
    print(json.dumps(scores,indent=2))
    print(f"Results: {result.output_dir}")


if __name__=="__main__":
    main()
