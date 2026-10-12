"""Train a toy vector field and plot it on a real UMAP of synthetic snapshots.

python examples/run_velocity.py --multimodal --unbalanced --output outputs/velocity
No biological data are used. Install with pip install -e '.[visualization]'.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(key,"1")
import argparse
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import anndata as ad
from umap import UMAP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import coati as ct


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multimodal",action=argparse.BooleanOptionalAction,default=False)
    mass=parser.add_mutually_exclusive_group()
    mass.add_argument("--unbalanced",dest="balanced",action="store_false")
    mass.add_argument("--balanced",dest="balanced",action="store_true")
    parser.set_defaults(balanced=True)
    parser.add_argument("--iterations",type=int,default=200)
    parser.add_argument("--cells",type=int,default=128)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    data=ct.datasets.paired(n=args.cells,seed=7) if args.multimodal else ct.datasets.split(n=args.cells,seed=7)
    result=ct.train_vector_field(data,multimodal=args.multimodal,balanced=args.balanced,
             iterations=args.iterations,batch_size=min(64,args.cells),seed=7,map_type="kernel",output_dir=args.output)
    x=np.concatenate(data.primary)
    times=np.repeat(data.times,[len(p) for p in data.primary])
    adata=ad.AnnData(x,obs=pd.DataFrame({"time":times,"stage":[f"t={t:g}" for t in times]},index=[f"toy-{i}" for i in range(len(x))]))
    adata.obsm["X_pca"]=x.copy()  # the toy model is trained directly in these coordinates
    reducer=UMAP(n_neighbors=30,min_dist=.35,random_state=7,n_jobs=1)
    adata.obsm["X_umap"]=reducer.fit_transform(x)
    result.write_velocity(adata,use_rep="X_pca",time_key="time")
    adata.write_h5ad(result.output_dir/"velocity.h5ad")
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for ax,style in zip(axes,["arrows","grid","stream"]):
        result.plot_velocity(adata,use_rep="X_pca",time_key="time",color="stage",style=style,ax=ax,
                             title=f"COATI · {style}",point_size=9,alpha=.65)
    fig.savefig(result.output_dir/"velocity_umap.png",dpi=180)
    fig.savefig(result.output_dir/"velocity_umap.pdf")
    result.plot_trajectory().figure.savefig(result.output_dir/"trajectories.png",dpi=150,bbox_inches="tight")
    scores=result.evaluate()
    (result.output_dir/"metrics.json").write_text(json.dumps(scores,indent=2)+"\n")
    print(json.dumps(scores,indent=2));print(f"Saved: {result.output_dir/'velocity_umap.png'}")


if __name__=="__main__":main()
