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

    def plot(self, *, layout="combined", normalization="zscore", show_observed=False, observed=None, show_iqr=False):
        """Overlay separately standardized RNA/ATAC profiles, or use separate figures."""
        return plot_gene_peak_dynamics(self, layout=layout, normalization=normalization, show_observed=show_observed,
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


def _profile_transform(mean, normalization):
    """Fit one display transform to the finite trajectory mean over time."""
    values = np.asarray(mean, dtype=float)
    finite = np.isfinite(values)
    if normalization is None:
        return lambda x: np.asarray(x, dtype=float), False
    if not finite.any():
        return lambda x: np.full(np.asarray(x).shape, np.nan), True
    center, scale = float(values[finite].mean()), float(values[finite].std(ddof=0))
    if scale <= 1e-12:
        # A flat mean has no temporal standard deviation. Keep its mean flat at
        # zero, but do not invent a scale for observed points or across-path IQR.
        return lambda x: np.full(np.asarray(x).shape, np.nan), True
    return lambda x: (np.asarray(x, dtype=float)-center)/scale, False


def plot_gene_peak_dynamics(dynamics, *, layout="combined", normalization="zscore",
                            show_observed=False, observed=None, show_iqr=False):
    """Plot RNA and linked peaks on one coordinate system, or on separate panels.

    combined overlays the gene and every linked peak on one Axes; each temporal
    mean is standardized separately as (value - temporal mean) / temporal std,
    with ddof=0 and finite trajectory time points. separate creates a gene figure
    and a peak figure, using the same normalization. normalization=None keeps
    native units only in separate mode. This display transform leaves the stored
    readouts unchanged and does not apply an additional log/count transform.

    Return Axes arrays keyed by gene and peaks. In combined mode all entries
    refer to the same Axes. show_observed is optional and requires a table with
    modality (RNA/ATAC), feature, time, mean and optionally cohort. Observed means
    use the matching model profile's center and scale, not their own fitted
    normalization. show_iqr transforms the bounds with that same scale; it
    describes across-path variation, not a molecular confidence interval.

    Unsupported times remain NaN. A constant mean becomes a zero line with
    undefined IQR/observed normalization omitted. Profiles with no supported
    times remain missing. Standardized shapes do not establish temporal precedence.
    """
    import matplotlib.pyplot as plt
    if not isinstance(dynamics, GenePeakDynamics):
        raise TypeError("dynamics must be GenePeakDynamics.")
    if layout not in ("combined", "separate"):
        raise ValueError("layout must be 'combined' or 'separate'.")
    if normalization not in ("zscore", None):
        raise ValueError("normalization must be 'zscore' or None.")
    if layout == "combined" and normalization is None:
        raise ValueError("Combined RNA/ATAC curves require normalization='zscore'; use separate for native units.")
    table = _observed_table(observed) if show_observed else None
    n_peaks = len(dynamics.atac.features)
    if layout == "combined":
        figure, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
        groups = {"gene": np.array([ax]), "peaks": np.array([ax]*n_peaks)}
        ax.set_title(f"{dynamics.gene} and candidate peaks", fontsize=13)
    else:
        groups = {}
        for name, count in [("gene", 1), ("peaks", n_peaks)]:
            _, axes = plt.subplots(count, 1, sharex=True, squeeze=False,
                                   figsize=(8, 2.7*count), layout="constrained")
            groups[name] = axes[:, 0]
    colors = ["#e69f00", "#cc79a7", "#d55e00", "#009e73", "#8b6eaf"]
    for key, modality, readout in [("gene", "RNA", dynamics.rna), ("peaks", "ATAC", dynamics.atac)]:
        for index, (ax, feature) in enumerate(zip(groups[key], readout.features)):
            sub = readout.summary.loc[readout.summary.feature == feature]
            raw_mean = sub["mean"].to_numpy(dtype=float)
            transform, flat = _profile_transform(raw_mean, normalization)
            curve = transform(raw_mean)
            if flat and np.isfinite(raw_mean).any():
                curve = np.where(np.isfinite(raw_mean), 0., np.nan)
            color = "#0072b2" if modality == "RNA" else colors[index % len(colors)]
            style = "-" if modality == "RNA" else ["--", ":", "-."][index % 3]
            label = f"{modality} · {feature}" + (" (flat profile)" if flat and np.isfinite(raw_mean).any() else "")
            ax.plot(sub.time, curve, color=color, ls=style, lw=2.2, label=label)
            if show_iqr and not flat:
                ax.fill_between(sub.time.to_numpy(), transform(sub.q25.to_numpy()), transform(sub.q75.to_numpy()),
                                color=color, alpha=.12)
            if table is not None:
                rows = table.loc[(table.modality == modality) & (table.feature == feature) &
                                 table.time.between(readout.times[0], readout.times[-1])]
                if readout.binary and ((rows["mean"] < 0) | (rows["mean"] > 1)).any():
                    raise ValueError("Observed binary peak means must be fractions in [0, 1] before normalization.")
                if not flat:
                    cohorts = rows.groupby("cohort", sort=False, dropna=False) if "cohort" in rows else [("snapshot", rows)]
                    for cohort, cohort_rows in cohorts:
                        if not cohort_rows.empty:
                            ax.scatter(cohort_rows.time, transform(cohort_rows["mean"]), color=color,
                                       marker="o" if modality == "RNA" else "D", s=32, zorder=3,
                                       label=f"Observed {modality} · {feature} · {cohort}")
            ylabel = "Standardized profile (z-score)" if normalization else readout.units
            ax.set(ylabel=ylabel, xlim=readout.times[[0, -1]])
            ax.spines[["top", "right"]].set_visible(False)
            if layout == "separate":
                ax.set_title(feature, fontsize=11)
            if readout.binary and normalization is None:
                ax.set_ylim(0, 1)
    unique_axes = list(dict.fromkeys([*groups["gene"], *groups["peaks"]]))
    for ax in unique_axes:
        ax.legend(frameon=False, fontsize=8, loc="best")
    if layout == "combined":
        unique_axes[0].set_xlabel("Model time")
    else:
        for axes in groups.values():
            axes[-1].set_xlabel("Model time")
    return groups
