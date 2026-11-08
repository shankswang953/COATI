"""Prepare the original Gaussian-bump observations without calling COATI training."""
import argparse
from pathlib import Path
import numpy as np


def sample_truncated_gaussian(rng, mean, cov, n, low=0.0, high=1.0):
    points = []
    remaining = n
    while remaining > 0:
        batch = rng.multivariate_normal(mean, cov, size=max(remaining * 2, 256))
        keep = batch[(batch >= low).all(axis=1) & (batch <= high).all(axis=1)]
        if keep.size == 0:
            continue
        points.append(keep[:remaining])
        remaining -= min(remaining, keep.shape[0])
    return np.concatenate(points, axis=0)


def make_data(seed=2026, n_initial=1000):
    rng = np.random.default_rng(seed)

    initial_covariance = 0.004 * np.eye(2)
    terminal_covariance = 0.003 * np.eye(2)

    time0 = sample_truncated_gaussian(
        rng,
        mean=np.array([0.30, 0.30]),
        cov=initial_covariance,
        n=n_initial,
    )

    n_each_terminal = n_initial
    terminal_a = sample_truncated_gaussian(
        rng,
        mean=np.array([0.56, 0.82]),
        cov=terminal_covariance,
        n=n_each_terminal,
    )
    terminal_b = sample_truncated_gaussian(
        rng,
        mean=np.array([0.83, 0.56]),
        cov=terminal_covariance,
        n=n_each_terminal,
    )
    time1 = np.concatenate([terminal_a, terminal_b], axis=0)

    return time0.astype(np.float32), time1.astype(np.float32)


def gaussian_bump_z(xy, sigma=0.08, amplitude=1.0):
    center = np.array([0.5, 0.5], dtype=np.float32)
    diff = xy - center
    r2 = np.sum(diff * diff, axis=-1)
    z = amplitude * np.exp(-r2 / (2.0 * sigma * sigma))
    return z.astype(np.float32)


def map_to_secondary(xy, sigma=0.08, amplitude=1.0, force_z_zero=False):
    z = gaussian_bump_z(xy, sigma=sigma, amplitude=amplitude)
    if force_z_zero:
        z = np.zeros_like(z)
    return np.concatenate([xy.astype(np.float32), z[:, None]], axis=1).astype(np.float32)


def sample_truncated_normal(rng, mean, std, n, low=0.0, high=1.0):
    chunks = []
    remaining = n
    while remaining > 0:
        batch = rng.normal(loc=mean, scale=std, size=(max(remaining * 2, 256), 2))
        keep = batch[(batch >= low).all(axis=1) & (batch <= high).all(axis=1)]
        if keep.size == 0:
            continue
        take = min(remaining, keep.shape[0])
        chunks.append(keep[:take])
        remaining -= take
    return np.concatenate(chunks, axis=0).astype(np.float32)


def make_support(rng, n_uniform, n_center, center_std, sigma, amplitude):
    uniform = rng.uniform(0.02, 0.98, size=(n_uniform, 2)).astype(np.float32)
    center = sample_truncated_normal(
        rng,
        mean=np.array([0.5, 0.5], dtype=np.float32),
        std=center_std,
        n=n_center,
        low=0.0,
        high=1.0,
    )
    support2d = np.concatenate([uniform, center], axis=0).astype(np.float32)
    support3d = map_to_secondary(
        support2d,
        sigma=sigma,
        amplitude=amplitude,
        force_z_zero=False,
    )
    return support2d, support3d


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True,help="A new output directory")
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--initial-cells",type=int,default=1000)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty output directory; existing data are never overwritten.")
    if args.initial_cells<1:
        parser.error("--initial-cells must be positive")
    args.output.mkdir(parents=True,exist_ok=True)
    primary=make_data(seed=args.seed,n_initial=args.initial_cells)
    sigma,amplitude=.08,1.
    secondary=[map_to_secondary(x,sigma=sigma,amplitude=amplitude,force_z_zero=True) for x in primary]
    support2d,support3d=make_support(np.random.default_rng(args.seed),4000,3000,.13,sigma,amplitude)
    np.savez(args.output/"rna_2d_by_time.npz",time0=primary[0],time1=primary[1])
    np.savez_compressed(args.output/"secondary_3d_bump_by_time.npz",time0=secondary[0],time1=secondary[1],
                        sigma=np.array([sigma],dtype=np.float32),amplitude=np.array([amplitude],dtype=np.float32),
                        zero_observed_z=np.array([True]))
    np.savez_compressed(args.output/"bump_support_2d_3d.npz",support2d=support2d,support3d=support3d,
                        sigma=np.array([sigma],dtype=np.float32),amplitude=np.array([amplitude],dtype=np.float32),
                        center=np.array([.5,.5],dtype=np.float32))
    print("Prepared Gaussian observations:",args.output.resolve())


if __name__=="__main__":
    main()
