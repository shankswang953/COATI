from pathlib import Path

import anndata as ad

from gastrulation.nmp_umap import (
    ATAC_UMAP_PARAMETERS,
    EXPECTED_UMAP_VERSION,
    RNA_UMAP_PARAMETERS,
    select_nmp_related,
)


def test_nmp_umap_inputs_and_parameters_are_frozen():
    data = Path(__file__).parents[1] / "examples/gastrulation/data/gastrulation_embeddings.h5ad"
    cells = ad.read_h5ad(data, backed="r")
    mask, labels = select_nmp_related(cells)
    assert int(mask.sum()) == len(labels) == 7931
    assert EXPECTED_UMAP_VERSION == "0.5.9.post2"
    assert RNA_UMAP_PARAMETERS == {
        "n_neighbors": 100,
        "min_dist": 0.8,
        "n_components": 2,
        "metric": "euclidean",
        "random_state": 42,
        "transform_seed": 42,
        "init": "spectral",
        "spread": 1.0,
        "n_jobs": 1,
        "angular_rp_forest": False,
    }
    assert ATAC_UMAP_PARAMETERS == {
        "n_neighbors": 150,
        "min_dist": 0.98,
        "n_components": 2,
        "metric": "euclidean",
        "random_state": 0,
        "transform_seed": 42,
        "init": "spectral",
        "spread": 1.0,
        "n_jobs": 1,
        "angular_rp_forest": True,
    }
    cells.file.close()
