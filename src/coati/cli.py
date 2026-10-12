"""Command-line entry point sharing the Python API."""
import argparse
import json
from pathlib import Path
from . import datasets, train_vector_field, load_result, Config, MapConfig, TemporalData

def main(argv=None):
    parser=argparse.ArgumentParser(description="COATI trajectory inference")
    sub=parser.add_subparsers(dest="command",required=True)
    toy=sub.add_parser("toy",help="Run a small synthetic example")
    toy.add_argument("--dataset",choices=["gaussian","split","paired"],default="gaussian")
    toy.add_argument("--niters",type=int,default=200)
    toy.add_argument("--seed",type=int,default=0)
    mass=toy.add_mutually_exclusive_group()
    mass.add_argument("--unbalanced",dest="balanced",action="store_false")
    mass.add_argument("--balanced",dest="balanced",action="store_true")
    toy.set_defaults(balanced=True)
    toy.add_argument("--multimodal",action=argparse.BooleanOptionalAction,default=None)
    toy.add_argument("--device",default="cpu")
    toy.add_argument("--output",required=True)
    toy.add_argument("--map-type", choices=["film", "mlp", "kernel", "accurate_kernel"], default="film")
    toy.add_argument("--map-epochs", type=int, default=300)
    train=sub.add_parser("train-vector-field",aliases=["fit"],help="Train a vector field from ordered NPZ snapshots")
    train.add_argument("input")
    train.add_argument("--secondary")
    train.add_argument("--keys",nargs="+",required=True)
    train.add_argument("--times",nargs="+",type=float,required=True)
    train.add_argument("--config",help="Optional JSON config; otherwise use single-modality balanced defaults")
    train.add_argument("--output",required=True)
    train.add_argument("--map-config", help="JSON MapConfig settings for automatic FiLM/MLP fitting")
    train.add_argument("--multimodal",action=argparse.BooleanOptionalAction,default=None)
    mass=train.add_mutually_exclusive_group()
    mass.add_argument("--balanced",dest="balanced",action="store_true")
    mass.add_argument("--unbalanced",dest="balanced",action="store_false")
    train.set_defaults(balanced=None)
    train.add_argument("--iterations",type=int)
    train.add_argument("--batch-size",type=int)
    train.add_argument("--learning-rate",type=float)
    plot=sub.add_parser("plot-velocity",help="Plot a saved vector field on an existing AnnData embedding")
    plot.add_argument("run")
    plot.add_argument("adata")
    plot.add_argument("--use-rep",default="X_pca",help="Training representation in obsm; X selects adata.X")
    plot.add_argument("--basis",default="umap")
    plot.add_argument("--time-key",default="time")
    plot.add_argument("--time",type=float)
    plot.add_argument("--time-mapping",help="JSON file mapping stage labels to training times")
    plot.add_argument("--color")
    plot.add_argument("--style",choices=["stream","grid","arrows"],default="stream")
    plot.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    if args.command=="plot-velocity":
        import anndata
        mapping=json.loads(Path(args.time_mapping).read_text()) if args.time_mapping else None
        result=load_result(args.run)
        adata=anndata.read_h5ad(args.adata)
        ax=result.plot_velocity(adata,use_rep=None if args.use_rep=="X" else args.use_rep,
                               basis=args.basis,time_key=args.time_key,time=args.time,time_mapping=mapping,
                               color=args.color,style=args.style)
        ax.figure.savefig(args.output,dpi=180,bbox_inches="tight")
        print(f"Saved velocity plot: {args.output}")
        return
    if args.command=="toy":
        data=getattr(datasets,args.dataset)(seed=args.seed)
        multimodal=args.dataset=="paired" if args.multimodal is None else args.multimodal
        cfg=Config(iterations=args.niters,seed=args.seed,device=args.device,
                   multimodal=multimodal,balanced=args.balanced,map_type=args.map_type)
        map_cfg = MapConfig(epochs=args.map_epochs) if cfg.multimodal and args.map_type in {"film", "mlp"} else None
    else:
        data=TemporalData.from_npz(args.input,keys=args.keys,times=args.times,secondary_path=args.secondary)
        cfg=Config.load(args.config) if args.config else Config()
        overrides={key:getattr(args,key) for key in ["multimodal","balanced","iterations","batch_size","learning_rate"] if getattr(args,key) is not None}
        cfg=cfg.with_parameters(**overrides)
        map_cfg = MapConfig(**json.loads(Path(args.map_config).read_text())) if args.map_config else None
    result=train_vector_field(data,cfg,output_dir=args.output,map_config=map_cfg)
    if result.map_report is not None:
        print(f"Map validation: {result.map_report['status']}. Report: {result.output_dir / 'map/report.txt'}")
    scores=result.evaluate()
    (result.output_dir/"metrics.json").write_text(json.dumps(scores,indent=2)+"\n")
    result.plot_trajectory().figure.savefig(result.output_dir/"trajectories.png",dpi=150,bbox_inches="tight")
    print(json.dumps(scores,indent=2))
    print(f"Results: {result.output_dir}")

if __name__=="__main__": main()
