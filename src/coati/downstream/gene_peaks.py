"""Gene-centered RNA and candidate peak readouts along continuous model paths."""
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import pandas as pd

from .features import FeatureDecoder, FeatureDynamics, compute_feature_dynamics


def find_gene_peaks(gene, peak_gene_links, *, gene_key="gene", peak_key="peak",
                    region_type=None, max_peaks=None):
    """Look up candidate peaks in a supplied DataFrame or CSV annotation table.

    Matching is exact and case-sensitive. Optional region_type uses the table's
    region_type column. Rank by absolute TSS distance when available, otherwise
    retain table order; no curve or outcome is used to choose peaks. Duplicate
    peak IDs are collapsed after sorting. Return canonical gene/peak columns
    and the original link metadata. These annotations do not prove regulation.
    """
    if not isinstance(gene, str) or not gene.strip():
        raise ValueError("gene must be a nonempty gene name.")
    if max_peaks is not None and (not isinstance(max_peaks, int) or isinstance(max_peaks, bool) or max_peaks < 1):
        raise ValueError("max_peaks must be a positive integer or None.")
    links = pd.read_csv(peak_gene_links) if isinstance(peak_gene_links, (str, Path)) else peak_gene_links
    if not isinstance(links, pd.DataFrame):
        raise TypeError("peak_gene_links must be a DataFrame or CSV path.")
    if gene_key == peak_key or not {gene_key, peak_key} <= set(links.columns):
        raise ValueError("The link table needs distinct gene and peak columns.")
    links = links.loc[links[gene_key].eq(gene)].copy()
    if links.empty:
        raise ValueError(f"No candidate peak links for gene {gene!r} in the supplied table.")
    if region_type is not None:
        if "region_type" not in links:
            raise ValueError("region_type filtering requires a region_type column.")
        links = links.loc[links["region_type"].eq(region_type)].copy()
    if links.empty:
        raise ValueError(f"No candidate peaks for {gene!r} with region_type={region_type!r}.")
    if links[peak_key].isna().any() or links[peak_key].astype(str).str.strip().eq("").any():
        raise ValueError("Candidate peak IDs must be nonempty.")
    links[peak_key] = links[peak_key].astype(str)
    if "distance" in links:
        # Stable sorting makes tied distances independent of correlation scores.
        order = np.argsort(pd.to_numeric(links["distance"], errors="raise").abs().to_numpy(), kind="stable")
        links = links.iloc[order]
    links = links.drop_duplicates(peak_key).head(max_peaks)
    if (gene_key != "gene" and "gene" in links) or (peak_key != "peak" and "peak" in links):
        raise ValueError("Custom link columns conflict with canonical gene/peak columns.")
    return links.rename(columns={gene_key: "gene", peak_key: "peak"}).reset_index(drop=True)


def _select_decoder(decoder, names, modality):
    if not isinstance(decoder, FeatureDecoder):
        raise TypeError(f"{modality} readout must be a fitted FeatureDecoder.")
    missing = [name for name in names if name not in decoder.features]
    if missing:
        raise ValueError(f"Missing {modality} readout features: {missing}. Fit the decoder on these observed features first.")
    indices = [decoder.features.index(name) for name in names]
    return replace(decoder, observed=decoder.observed[:, indices], features=tuple(names))


@dataclass
class GenePeakDynamics:
    """One gene and its candidate peaks, keeping each modality's units separate."""
    gene: str
    rna: FeatureDynamics
    atac: FeatureDynamics
    links: pd.DataFrame

    def plot(self, *, layout="combined", show_observed=False, observed=None, show_iqr=True):
        """Plot together or separately; measured snapshot means are optional."""
        return plot_gene_peak_dynamics(self, layout=layout, show_observed=show_observed,
                                      observed=observed, show_iqr=show_iqr)


def compute_gene_peak_dynamics(gene, prediction, *, gene_decoder, peak_decoder,
                               peak_gene_links, region_type=None, max_peaks=None,
                               selection=None, weighting="uniform"):
    """Read one gene and automatically look up its candidate peaks along paths.

    Fit RNA and ATAC FeatureDecoders once in their corresponding model spaces.
    The decoders must contain the named gene and all selected candidate peaks;
    missing features raise instead of being silently skipped. prediction must
    contain primary RNA and secondary ATAC paths. Both use the same selection
    and weighting, with support screened separately in each modality.
    A gene name selects existing observations and annotations; no de novo
    linking, molecular prediction model or new vector-field training is added.
    """
    links = find_gene_peaks(gene, peak_gene_links, region_type=region_type, max_peaks=max_peaks)
    rna = _select_decoder(gene_decoder, [gene], "RNA")
    atac = _select_decoder(peak_decoder, links["peak"].tolist(), "ATAC")
    primary, secondary = np.asarray(prediction["primary"]), np.asarray(prediction["secondary"])
    if primary.ndim != 3 or secondary.ndim != 3 or primary.shape[:2] != secondary.shape[:2]:
        raise ValueError("RNA and ATAC trajectories must have matching time and path dimensions.")
    return GenePeakDynamics(
        gene, compute_feature_dynamics(rna, prediction, selection=selection, weighting=weighting),
        compute_feature_dynamics(atac, prediction, modality="secondary", selection=selection, weighting=weighting),
        links,
    )


def _observed_table(observed):
    if not isinstance(observed, pd.DataFrame) or not {"modality", "feature", "time", "mean"} <= set(observed.columns):
        raise ValueError("Observed overlays require a table with modality, feature, time and mean columns.")
    if not observed["modality"].isin(["RNA", "ATAC"]).all() or observed["feature"].isna().any():
        raise ValueError("Observed rows need RNA/ATAC modality and nonmissing feature names.")
    numeric = observed[["time", "mean"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Observed time and mean values must be finite.")
    keys = ["modality", "feature", "time"] + (["cohort"] if "cohort" in observed else [])
    if observed.duplicated(keys).any():
        raise ValueError("Observed means must be unique per modality, feature, time and cohort.")
    return observed


def plot_gene_peak_dynamics(dynamics, *, layout="combined", show_observed=False,
                            observed=None, show_iqr=True):
    """Plot a gene and candidate peaks with shared time and separate native units.

    combined places the gene above the peaks in one figure. separate creates
    a gene figure and a peak figure. Return a dict of Axes arrays keyed by gene
    and peaks in either layout. show_observed defaults to False; when enabled,
    observed must contain modality (RNA/ATAC), feature, time, mean and optionally
    cohort. Scatter points are measured snapshot-cohort means, not the fitted
    trajectory or longitudinal observations of the same cells. IQR describes
    across-path variation. The plot does not normalize expression and peaks onto
    a common y scale or infer temporal precedence.
    """
    import matplotlib.pyplot as plt
    if not isinstance(dynamics, GenePeakDynamics):
        raise TypeError("dynamics must be GenePeakDynamics.")
    if layout not in ("combined", "separate"):
        raise ValueError("layout must be 'combined' or 'separate'.")
    table = _observed_table(observed) if show_observed else None
    n_peaks = len(dynamics.atac.features)
    if layout == "combined":
        figure, axes = plt.subplots(1+n_peaks, 1, sharex=True, squeeze=False,
                                   figsize=(8, 2.4*(1+n_peaks)), layout="constrained")
        axes = axes[:, 0]
        groups = {"gene": axes[:1], "peaks": axes[1:]}
        figure.suptitle(f"{dynamics.gene} and candidate peaks", fontsize=14)
    else:
        groups = {}
        for name, count in [("gene", 1), ("peaks", n_peaks)]:
            _, axes = plt.subplots(count, 1, sharex=True, squeeze=False,
                                   figsize=(8, 2.4*count), layout="constrained")
            groups[name] = axes[:, 0]
    for key, modality, readout, color in [("gene", "RNA", dynamics.rna, "#0072b2"),
                                         ("peaks", "ATAC", dynamics.atac, "#e69f00")]:
        for ax, feature in zip(groups[key], readout.features):
            sub = readout.summary.loc[readout.summary.feature == feature]
            ax.plot(sub.time, sub["mean"], color=color, lw=2, label="kNN trajectory mean")
            if show_iqr:
                ax.fill_between(sub.time.to_numpy(), sub.q25.to_numpy(), sub.q75.to_numpy(),
                                color=color, alpha=.16, label="Across-path IQR")
            if table is not None:
                rows = table.loc[(table.modality == modality) & (table.feature == feature) &
                                 table.time.between(readout.times[0], readout.times[-1])]
                if readout.binary and ((rows["mean"] < 0) | (rows["mean"] > 1)).any():
                    raise ValueError("Observed binary peak means must be fractions in [0, 1].")
                if "cohort" in rows:
                    for index, (cohort, cohort_rows) in enumerate(rows.groupby("cohort", sort=False, dropna=False)):
                        ax.scatter(cohort_rows.time, cohort_rows["mean"], color="#243445",
                                   marker=["D", "s", "^"][index % 3], s=34, zorder=3,
                                   label=f"Observed: {cohort}")
                elif not rows.empty:
                    ax.scatter(rows.time, rows["mean"], color="#243445", marker="D", s=34,
                               zorder=3, label="Observed snapshot mean")
            ax.set(title=feature, ylabel=readout.units, xlim=readout.times[[0, -1]])
            ax.title.set_fontsize(11)
            ax.spines[["top", "right"]].set_visible(False)
            if readout.binary:
                ax.set_ylim(0, 1)
            if table is not None:
                ax.legend(frameon=False, fontsize=8, loc="best")
        if layout == "separate" or key == "peaks":
            groups[key][-1].set_xlabel("Model time")
    return groups
