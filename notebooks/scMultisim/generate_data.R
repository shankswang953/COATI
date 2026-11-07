#!/usr/bin/env Rscript

# Generate paired scRNA-seq and scATAC-seq data with scMultiSim.
#
# Usage:
#   Rscript generate_data.R [output_directory]
#
# The default output directory is "scmultisim_data". The script keeps the
# simulation parameters fixed so that the dataset can be reproduced exactly.

required_packages <- c("scMultiSim", "ggplot2")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    "Install the required R packages before running this script: ",
    paste(missing_packages, collapse = ", ")
  )
}

suppressPackageStartupMessages({
  library(scMultiSim)
  library(ggplot2)
})

arguments <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(arguments) >= 1) arguments[[1]] else "scmultisim_data"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

seed <- 42L
set.seed(seed)

data(GRN_params_100, package = "scMultiSim")

simulation_options <- list(
  GRN = GRN_params_100,
  num.cells = 10000L,
  num.cifs = 30L,
  tree = Phyla3(),
  diff.cif.fraction = 0.9,
  cif.sigma = 0.1,
  do.velocity = TRUE,
  discrete.cif = FALSE
)

message("Simulating paired scRNA-seq and scATAC-seq data...")
started_at <- Sys.time()
simulation <- sim_true_counts(simulation_options)
elapsed_minutes <- as.numeric(difftime(Sys.time(), started_at, units = "mins"))

rna_counts <- simulation$counts
atac_counts <- simulation$atac_counts
velocity <- simulation$velocity
unspliced_counts <- simulation$unspliced_counts
cell_metadata <- as.data.frame(simulation$cell_meta)

stopifnot(
  ncol(rna_counts) == ncol(atac_counts),
  ncol(rna_counts) == ncol(velocity),
  ncol(rna_counts) == ncol(unspliced_counts),
  ncol(rna_counts) == nrow(cell_metadata)
)

pseudotime <- as.numeric(simulation$cell_time)
if (length(pseudotime) != ncol(rna_counts)) {
  stop("simulation$cell_time must contain one value per cell.")
}

# Use the simulated pseudotime when it is informative. PCA is a deterministic
# fallback for scMultiSim outputs with a constant or degenerate cell_time.
if (any(!is.finite(pseudotime)) || diff(range(pseudotime, na.rm = TRUE)) < 1e-6) {
  message("cell_time is degenerate; deriving pseudotime from RNA PC1.")
  rna_pca_fallback <- prcomp(t(log2(rna_counts + 1)), scale. = TRUE)
  pseudotime <- rna_pca_fallback$x[, 1]

  edge_size <- min(100L, floor(length(pseudotime) / 2L))
  first_edge <- seq_len(edge_size)
  last_edge <- seq.int(length(pseudotime) - edge_size + 1L, length(pseudotime))
  if (mean(pseudotime[first_edge]) > mean(pseudotime[last_edge])) {
    pseudotime <- -pseudotime
  }
}

# Split the pseudotime range into the four equal-width intervals used in the
# original workflow. Cell counts are allowed to differ between snapshots.
time_breaks <- seq(
  min(pseudotime, na.rm = TRUE),
  max(pseudotime, na.rm = TRUE),
  length.out = 5L
)

time_bin <- cut(
  pseudotime,
  breaks = time_breaks,
  include.lowest = TRUE,
  labels = paste0("time_", seq_len(4L))
)
time_index <- as.integer(time_bin)

cell_ids <- colnames(rna_counts)
if (is.null(cell_ids)) {
  cell_ids <- sprintf("cell_%05d", seq_len(ncol(rna_counts)))
}

rownames(cell_metadata) <- cell_ids
cell_metadata$barcode <- cell_ids
cell_metadata$pseudotime <- pseudotime
cell_metadata$time <- time_index
cell_metadata$time_label <- as.character(time_bin)

write_matrix <- function(matrix, filename) {
  write.csv(
    matrix,
    file = file.path(output_dir, filename),
    row.names = TRUE,
    quote = FALSE
  )
}

message("Writing complete matrices and metadata...")
write_matrix(rna_counts, "complete_scRNA.csv")
write_matrix(atac_counts, "complete_scATAC.csv")
write_matrix(velocity, "complete_velocity.csv")
write_matrix(unspliced_counts, "complete_unspliced.csv")
write.csv(
  cell_metadata,
  file = file.path(output_dir, "complete_meta.csv"),
  row.names = TRUE,
  quote = FALSE
)

for (time_point in seq_len(4L)) {
  selected <- which(time_index == time_point)
  prefix <- paste0("time_", time_point)

  write_matrix(rna_counts[, selected, drop = FALSE], paste0(prefix, "_scRNA.csv"))
  write_matrix(atac_counts[, selected, drop = FALSE], paste0(prefix, "_scATAC.csv"))
  write_matrix(velocity[, selected, drop = FALSE], paste0(prefix, "_velocity.csv"))
  write_matrix(
    unspliced_counts[, selected, drop = FALSE],
    paste0(prefix, "_unspliced.csv")
  )
  write.csv(
    cell_metadata[selected, , drop = FALSE],
    file = file.path(output_dir, paste0(prefix, "_meta.csv")),
    row.names = TRUE,
    quote = FALSE
  )
}

build_pca_frame <- function(counts, metadata) {
  fit <- prcomp(t(log2(counts + 1)), scale. = TRUE)
  explained <- summary(fit)$importance[2, 1:2] * 100
  frame <- data.frame(
    PC1 = fit$x[, 1],
    PC2 = fit$x[, 2],
    time = factor(metadata$time),
    celltype = factor(metadata$pop)
  )
  list(frame = frame, explained = explained)
}

plot_pca <- function(pca_result, color_by, title, filename) {
  plot <- ggplot(
    pca_result$frame,
    aes(x = PC1, y = PC2, color = .data[[color_by]])
  ) +
    geom_point(size = 0.45, alpha = 0.55) +
    labs(
      title = title,
      color = color_by,
      x = sprintf("PC1 (%.1f%%)", pca_result$explained[[1]]),
      y = sprintf("PC2 (%.1f%%)", pca_result$explained[[2]])
    ) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "right")

  ggsave(
    filename = file.path(output_dir, filename),
    plot = plot,
    width = 6.4,
    height = 5.0,
    dpi = 180
  )
}

message("Computing PCA quality-control plots...")
rna_pca <- build_pca_frame(rna_counts, cell_metadata)
atac_pca <- build_pca_frame(atac_counts, cell_metadata)

plot_pca(rna_pca, "time", "scRNA-seq PCA by time", "scrna_pca_time.png")
plot_pca(rna_pca, "celltype", "scRNA-seq PCA by cell type", "scrna_pca_celltype.png")
plot_pca(atac_pca, "time", "scATAC-seq PCA by time", "scatac_pca_time.png")
plot_pca(
  atac_pca,
  "celltype",
  "scATAC-seq PCA by cell type",
  "scatac_pca_celltype.png"
)

manifest <- data.frame(
  seed = seed,
  grn = "GRN_params_100",
  tree = "Phyla3",
  num_cifs = simulation_options$num.cifs,
  diff_cif_fraction = simulation_options$diff.cif.fraction,
  cif_sigma = simulation_options$cif.sigma,
  velocity = simulation_options$do.velocity,
  discrete_cif = simulation_options$discrete.cif,
  cells = ncol(rna_counts),
  genes = nrow(rna_counts),
  peaks = nrow(atac_counts),
  snapshots = 4L,
  elapsed_minutes = elapsed_minutes
)
write.csv(
  manifest,
  file = file.path(output_dir, "simulation_manifest.csv"),
  row.names = FALSE,
  quote = FALSE
)
writeLines(
  capture.output(sessionInfo()),
  con = file.path(output_dir, "session_info.txt")
)

message(
  sprintf(
    "Finished: %d genes, %d peaks, and %d cells written to %s (%.2f minutes).",
    nrow(rna_counts),
    nrow(atac_counts),
    ncol(rna_counts),
    normalizePath(output_dir, mustWork = FALSE),
    elapsed_minutes
  )
)
