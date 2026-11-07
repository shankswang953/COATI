"""Generate AnnData, fit T and compare fresh balanced fields using the original Args."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import coati as ct
from gaussian_fresh import generate_fresh_anndata
from gaussian_map import fit_bump_map, evaluate_bump_map, plot_map_training, plot_map_performance
from gaussian_parameters import make_flow_config
from gaussian_comparison import make_shared_initial_checkpoint, plot_gaussian_sync, plot_bump_height, compare_gaussian_runs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--map-steps",type=int,default=6000)
    parser.add_argument("--iterations",type=int,default=3000)
    parser.add_argument("--secondary-weight",type=float,default=.5)
    parser.add_argument("--map-only",action="store_true",help="Inspect the freshly fitted T before training fields")
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose a fresh result directory.")
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    plt.rcParams.update({"font.family":"Arial","font.size":11})
    cells,map_pairs=generate_fresh_anndata()
    T,history,fit=fit_bump_map(map_pairs,steps=args.map_steps,progress_path=args.output/"progress.json")
    metrics,grid=evaluate_bump_map(T,map_pairs)
    history.to_csv(args.output/"map_training.csv",index=False)
    metrics.to_csv(args.output/"map_performance.csv")
    for name,figure in [("map_training",plot_map_training(history)),("map_performance",plot_map_performance(grid))]:
        figure.savefig(args.output/(name+".png"),dpi=180,bbox_inches="tight")
        plt.close(figure)
    summary={**fit,"grid_z_correlation":grid["z_correlation"],"center_predicted_height":grid["center_predicted_height"]}
    (args.output/"map_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(metrics.to_string());print(json.dumps(summary,indent=2))
    if args.map_only:
        return
    config=make_flow_config(iterations=args.iterations, secondary_weight=args.secondary_weight)
    initial=make_shared_initial_checkpoint(config,args.output/"initial.pt")
    results,predictions={},{}
    for name,cfg in {"Without sync":config.with_parameters(multimodal=False,secondary_weight=0.),"With sync":config}.items():
        print("Training:",name,flush=True)
        result=ct.train_vector_field(cells,cfg,use_rep="X_latent",secondary_obsm_key="X_secondary",
            T_model=T if cfg.multimodal else None,output_dir=args.output/("with_sync" if cfg.multimodal else "without_sync"),warm_start_path=initial)
        prediction=result.predict_trajectory(times=np.linspace(0.,1.,51),method="rk4")
        with torch.no_grad(): prediction["secondary"]=T(torch.tensor(prediction["primary"])).numpy()
        assert all(np.isfinite(value).all() for value in prediction.values())
        results[name],predictions[name]=result,prediction
        print(name,"completed",flush=True)
    data=ct.TemporalData.from_anndata(cells,secondary_obsm_key="X_secondary")
    for name,figure in [("comparison",plot_gaussian_sync(data,T,results,predictions)),("bump_height",plot_bump_height(predictions))]:
        figure.savefig(args.output/(name+".png"),dpi=180,bbox_inches="tight");plt.close(figure)
    scores=compare_gaussian_runs(results,predictions)
    scores.to_csv(args.output/"comparison.csv")
    print(scores.to_string())
    print("Results:",args.output.resolve())


if __name__=="__main__":
    main()
