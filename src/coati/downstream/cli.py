"""Command-line entry points for reusable downstream analyses."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd

from .analyses import (
    paired_modality_terminal_trajectory_difference,
    rna_atac_disagreement_by_time,
    same_space_terminal_trajectory_difference,
    time_resolved_cross_modal_consistency,
)
from .io_utils import (
    concat_by_time,
    ensure_dir,
    load_labels_by_time,
    load_npz_by_time,
    load_trajectory,
)
from .t_mapping import load_plain_mlp, apply_t, summarize_t_error, t_error_table


def _load_reference(data_path, norm_path, labels_path, time_keys, stage_names):
    data = load_npz_by_time(data_path, norm_path, time_keys)
    labels = load_labels_by_time(labels_path, time_keys) if labels_path else None
    return concat_by_time(data, labels, time_keys, stage_names)


def cmd_same_space(args):
    ensure_dir(args.out_dir)
    ref = _load_reference(args.data, args.norm, args.labels, args.time_keys, args.stage_names)
    reference_traj = load_trajectory(args.reference_traj)
    query_traj = load_trajectory(args.query_traj)
    initial = None
    if args.initial_labels and ref.labels is not None:
        z = load_labels_by_time(args.labels, args.time_keys)
        initial = z[args.time_keys[0]]
    df = same_space_terminal_trajectory_difference(
        reference_traj,
        query_traj,
        ref.x,
        labels=ref.labels,
        initial_labels=initial,
        k=args.k,
        batch_size=args.batch_size,
    )
    df.to_csv(os.path.join(args.out_dir, "same_space_terminal_trajectory_difference.csv"), index=False)


def cmd_paired_modality(args):
    ensure_dir(args.out_dir)
    primary = _load_reference(args.primary_data, args.primary_norm, args.labels, args.time_keys, args.stage_names)
    secondary = _load_reference(args.secondary_data, args.secondary_norm, args.labels, args.time_keys, args.stage_names)
    primary_traj = load_trajectory(args.primary_traj)
    secondary_traj = load_trajectory(args.secondary_traj)
    initial = None
    if args.labels:
        initial = load_labels_by_time(args.labels, args.time_keys)[args.time_keys[0]]
    df = paired_modality_terminal_trajectory_difference(
        primary_traj,
        secondary_traj,
        primary.x,
        secondary.x,
        labels=primary.labels,
        initial_labels=initial,
        k=args.k,
        batch_size=args.batch_size,
    )
    df.to_csv(os.path.join(args.out_dir, "paired_modality_terminal_trajectory_difference.csv"), index=False)


def cmd_disagreement(args):
    ensure_dir(args.out_dir)
    primary = _load_reference(args.primary_data, args.primary_norm, args.labels, args.time_keys, args.stage_names)
    secondary = _load_reference(args.secondary_data, args.secondary_norm, args.labels, args.time_keys, args.stage_names)
    initial = load_labels_by_time(args.labels, args.time_keys)[args.time_keys[0]]
    df = rna_atac_disagreement_by_time(
        load_trajectory(args.primary_traj),
        load_trajectory(args.secondary_traj),
        primary.x,
        secondary.x,
        primary.labels,
        initial_labels=initial,
        time_indices=args.time_indices,
        time_names=args.time_names,
        k=args.k,
    )
    df.to_csv(os.path.join(args.out_dir, "rna_atac_disagreement_by_time.csv"), index=False)
    top = (
        df.sort_values(["time_index", "tv_distance"], ascending=[True, False])
        .groupby("time_index", group_keys=False)
        .head(max(1, args.top_n))
    )
    top.to_csv(os.path.join(args.out_dir, "rna_atac_disagreement_top_cells.csv"), index=False)


def cmd_consistency(args):
    ensure_dir(args.out_dir)
    primary = _load_reference(args.primary_data, args.primary_norm, args.labels, args.time_keys, args.stage_names)
    secondary = _load_reference(args.secondary_data, args.secondary_norm, args.labels, args.time_keys, args.stage_names)
    summary, per_cell = time_resolved_cross_modal_consistency(
        load_trajectory(args.primary_traj),
        load_trajectory(args.secondary_traj),
        primary.x,
        secondary.x,
        primary.labels,
        k=args.k,
    )
    summary.to_csv(os.path.join(args.out_dir, "cross_modal_consistency_summary.csv"), index=False)
    if args.write_per_cell:
        per_cell.to_csv(os.path.join(args.out_dir, "cross_modal_consistency_per_cell.csv"), index=False)


def cmd_t_diagnostics(args):
    ensure_dir(args.out_dir)
    primary = _load_reference(args.primary_data, args.primary_norm, args.labels, args.time_keys, args.stage_names)
    secondary = _load_reference(args.secondary_data, args.secondary_norm, args.labels, args.time_keys, args.stage_names)
    model, ckpt = load_plain_mlp(args.t_checkpoint, device=args.device)
    df = t_error_table(
        primary.x,
        secondary.x,
        labels=primary.labels,
        stages=primary.stage,
        model=model,
        k=args.k,
        device=args.device,
    )
    df.to_csv(os.path.join(args.out_dir, "t_mapping_error_per_cell.csv"), index=False)
    if primary.labels is not None:
        summarize_t_error(df, ["cell_type"]).to_csv(os.path.join(args.out_dir, "t_mapping_error_by_cell_type.csv"), index=False)
    summarize_t_error(df, ["stage"]).to_csv(os.path.join(args.out_dir, "t_mapping_error_by_stage.csv"), index=False)
    with open(os.path.join(args.out_dir, "t_checkpoint_meta.json"), "w") as f:
        json.dump({"config": ckpt.get("config", {}), "meta": ckpt.get("meta", {})}, f, indent=2)


def add_common_reference_args(p, paired=False):
    p.add_argument("--time_keys", nargs="+", default=["time_0", "time_1", "time_2"])
    p.add_argument("--stage_names", nargs="+", default=None)
    p.add_argument("--labels", default=None, help="celltype_sub_by_stage.npz-like file")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--out_dir", required=True)
    if paired:
        p.add_argument("--primary_data", required=True)
        p.add_argument("--secondary_data", required=True)
        p.add_argument("--primary_norm", default=None)
        p.add_argument("--secondary_norm", default=None)
    else:
        p.add_argument("--data", required=True)
        p.add_argument("--norm", default=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reusable TraInf downstream analyses")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("same-space-distance", help="Compare two trajectories in one reference space")
    add_common_reference_args(p, paired=False)
    p.add_argument("--reference_traj", required=True)
    p.add_argument("--query_traj", required=True)
    p.add_argument("--initial_labels", action="store_true")
    p.add_argument("--batch_size", type=int, default=256)
    p.set_defaults(func=cmd_same_space)

    p = sub.add_parser("paired-modality-distance", help="Compare primary/secondary trajectories using paired cell indices")
    add_common_reference_args(p, paired=True)
    p.add_argument("--primary_traj", required=True)
    p.add_argument("--secondary_traj", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.set_defaults(func=cmd_paired_modality)

    p = sub.add_parser("rna-atac-disagreement", help="Find cells with different primary/secondary soft labels")
    add_common_reference_args(p, paired=True)
    p.add_argument("--primary_traj", required=True)
    p.add_argument("--secondary_traj", required=True)
    p.add_argument("--time_indices", nargs="+", type=int, default=None)
    p.add_argument("--time_names", nargs="+", default=None)
    p.add_argument("--top_n", type=int, default=300)
    p.set_defaults(func=cmd_disagreement)

    p = sub.add_parser("cross-modal-consistency", help="Time-resolved JS label consistency")
    add_common_reference_args(p, paired=True)
    p.add_argument("--primary_traj", required=True)
    p.add_argument("--secondary_traj", required=True)
    p.add_argument("--write_per_cell", action="store_true")
    p.set_defaults(func=cmd_consistency)

    p = sub.add_parser("t-diagnostics", help="Evaluate a primary-to-secondary T checkpoint")
    add_common_reference_args(p, paired=True)
    p.add_argument("--t_checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_t_diagnostics)

    args = parser.parse_args(argv)
    if args.stage_names is None:
        args.stage_names = args.time_keys
    if hasattr(args, "time_names") and args.time_names is not None and args.time_indices is not None:
        if len(args.time_names) != len(args.time_indices):
            raise ValueError("--time_names and --time_indices must have the same length")
    args.func(args)


if __name__ == "__main__":
    main()
