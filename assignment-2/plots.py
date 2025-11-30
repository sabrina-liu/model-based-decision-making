"""
plots.py

Visualisation utilities for Assignment 2 threshold experiments.

Reads the "_all" CSV files produced by main.py
and generates:

For a chosen (tau, seed_fraction) slice:
    - mean adoption curves with variance bands
    - early adoption curves (zoomed window)
    - final adoption bar chart (with std)
    - convergence speed bar chart (n_steps)
    - time-to-50%-adoption (t_50) bar chart
    - time-to-90%-adoption (t_90) bar chart
    - AUC bar chart
    - scatter plot: AUC vs t_90 (per run, coloured by strategy)
    - boxplot of t_90 per strategy

Over the whole parameter grid:
    - heatmaps of final adoption vs (tau, seed_fraction) per strategy
    - heatmaps of t_90 vs (tau, seed_fraction) per strategy
    - optional 3D surface plots (if flag enabled)

If the node-level adoption-time file exists:
    - degree vs adoption time plot.

Usage
-----
python3 plots.py \
    --prefix ass2 \
    --output-dir plots \
    --tau 0.2 \
    --seed-fraction 0.01

Author: Sabrina Liu
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


# ============================================================================
# Loading utilities
# ============================================================================

def load_results(prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the combined result CSVs for a given prefix.

    Returns
    -------
    per_run_all, aggregate_all, ts_all, mean_curves_all
    """
    per_run_path = f"{prefix}_per_run_all.csv"
    aggregate_path = f"{prefix}_aggregate_all.csv"
    ts_path = f"{prefix}_ts_all.csv"
    mean_curves_path = f"{prefix}_mean_curves_all.csv"

    for path in [per_run_path, aggregate_path, ts_path, mean_curves_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    LOGGER.info("Loading per-run results from %s", per_run_path)
    per_run_all = pd.read_csv(per_run_path)

    LOGGER.info("Loading aggregate results from %s", aggregate_path)
    aggregate_all = pd.read_csv(aggregate_path)

    LOGGER.info("Loading time series from %s", ts_path)
    ts_all = pd.read_csv(ts_path)

    LOGGER.info("Loading mean curves from %s", mean_curves_path)
    mean_curves_all = pd.read_csv(mean_curves_path)

    return per_run_all, aggregate_all, ts_all, mean_curves_all


# ============================================================================
# Slice by (tau, seed_fraction)
# ============================================================================

def select_slice(
    per_run_all: pd.DataFrame,
    ts_all: pd.DataFrame,
    mean_curves_all: pd.DataFrame,
    tau: float | None,
    seed_fraction: float | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:
    """
    Select the subset of results corresponding to a single (tau, seed_fraction).

    If tau or seed_fraction are None, use the smallest values present.
    """
    available_taus = sorted(per_run_all["tau"].unique())
    available_seeds = sorted(per_run_all["seed_fraction"].unique())

    if tau is None:
        tau = float(available_taus[0])
    if seed_fraction is None:
        seed_fraction = float(available_seeds[0])

    LOGGER.info("Using slice tau=%.3f, seed_fraction=%.3f", tau, seed_fraction)

    mask = (per_run_all["tau"] == tau) & (per_run_all["seed_fraction"] == seed_fraction)
    per_run_slice = per_run_all[mask].copy()

    mask_ts = (ts_all["tau"] == tau) & (ts_all["seed_fraction"] == seed_fraction)
    ts_slice = ts_all[mask_ts].copy()

    mask_mc = (mean_curves_all["tau"] == tau) & (mean_curves_all["seed_fraction"] == seed_fraction)
    mean_curves_slice = mean_curves_all[mask_mc].copy()

    return per_run_slice, ts_slice, mean_curves_slice, tau, seed_fraction


# ============================================================================
# Per-slice plotting
# ============================================================================

def plot_mean_adoption_curves(
    mean_curves_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Mean adoption curves with variance bands."""
    plt.figure(figsize=(7, 4))

    for strategy, sub in mean_curves_df.groupby("strategy"):
        sub_sorted = sub.sort_values("time_step")
        plt.plot(
            sub_sorted["time_step"],
            sub_sorted["mean_fraction"],
            label=strategy,
        )
        plt.fill_between(
            sub_sorted["time_step"],
            sub_sorted["mean_fraction"] - sub_sorted["std_fraction"],
            sub_sorted["mean_fraction"] + sub_sorted["std_fraction"],
            alpha=0.2,
        )

    plt.xlabel("Time step")
    plt.ylabel("Fraction of active nodes")
    plt.title("Adoption over time by seeding strategy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_early_adoption_curves(
    mean_curves_df: pd.DataFrame,
    output_path: str,
    max_time: int = 20,
) -> None:
    """Zoomed view on the early phase of adoption."""
    plt.figure(figsize=(7, 4))

    for strategy, sub in mean_curves_df.groupby("strategy"):
        sub_small = sub[sub["time_step"] <= max_time].sort_values("time_step")
        plt.plot(
            sub_small["time_step"],
            sub_small["mean_fraction"],
            label=strategy,
        )
        plt.fill_between(
            sub_small["time_step"],
            sub_small["mean_fraction"] - sub_small["std_fraction"],
            sub_small["mean_fraction"] + sub_small["std_fraction"],
            alpha=0.2,
        )

    plt.xlabel("Time step")
    plt.ylabel("Fraction of active nodes")
    plt.title(f"Early adoption (t ≤ {max_time}) by seeding strategy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _bar_with_error(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: str,
) -> None:
    """Generic bar chart with std error bars."""
    grouped = (
        df
        .groupby("strategy")
        .agg(
            mean_val=(metric, "mean"),
            std_val=(metric, "std"),
        )
        .reset_index()
    )

    strategies = grouped["strategy"].tolist()
    means = grouped["mean_val"].values
    stds = grouped["std_val"].values

    plt.figure(figsize=(6, 4))
    x_pos = np.arange(len(strategies))
    plt.bar(x_pos, means, yerr=stds, capsize=5)
    plt.xticks(x_pos, strategies)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_final_fraction_bar(per_run_df: pd.DataFrame, output_path: str) -> None:
    _bar_with_error(
        df=per_run_df,
        metric="final_fraction",
        ylabel="Final fraction of active nodes",
        title="Final adoption by seeding strategy",
        output_path=output_path,
    )


def plot_steps_bar(per_run_df: pd.DataFrame, output_path: str) -> None:
    _bar_with_error(
        df=per_run_df,
        metric="n_steps",
        ylabel="Number of steps to steady state",
        title="Convergence speed by seeding strategy",
        output_path=output_path,
    )


def plot_t50_bar(per_run_df: pd.DataFrame, output_path: str) -> None:
    _bar_with_error(
        df=per_run_df,
        metric="t_50",
        ylabel="Steps until 50% adoption",
        title="Time to half adoption by seeding strategy",
        output_path=output_path,
    )


def plot_t90_bar(per_run_df: pd.DataFrame, output_path: str) -> None:
    _bar_with_error(
        df=per_run_df,
        metric="t_90",
        ylabel="Steps until 90% adoption",
        title="Time to 90% adoption by seeding strategy",
        output_path=output_path,
    )


def plot_auc_bar(per_run_df: pd.DataFrame, output_path: str) -> None:
    _bar_with_error(
        df=per_run_df,
        metric="auc",
        ylabel="Area under adoption curve",
        title="Overall diffusion efficiency (AUC) by strategy",
        output_path=output_path,
    )


def plot_auc_vs_t90_scatter(per_run_df: pd.DataFrame, output_path: str) -> None:
    """Scatter of AUC vs t_90, one point per run."""
    plt.figure(figsize=(6, 4))
    strategies = per_run_df["strategy"].unique()
    for s in strategies:
        sub = per_run_df[per_run_df["strategy"] == s]
        plt.scatter(sub["t_90"], sub["auc"], label=s, alpha=0.8)

    plt.xlabel("t_90 (steps until 90% adoption)")
    plt.ylabel("Area under adoption curve (AUC)")
    plt.title("Speed vs efficiency of diffusion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_t90_boxplot(per_run_df: pd.DataFrame, output_path: str) -> None:
    """Boxplot of t_90 per strategy (distributional view)."""
    plt.figure(figsize=(6, 4))
    strategies = per_run_df["strategy"].unique().tolist()
    data = [per_run_df[per_run_df["strategy"] == s]["t_90"].values for s in strategies]

    plt.boxplot(data, labels=strategies)
    plt.ylabel("Steps until 90% adoption")
    plt.title("Distribution of t_90 by seeding strategy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================================
# Heatmaps & optional 3D surfaces over parameter grid
# ============================================================================

def plot_heatmap_metric_over_params(
    aggregate_all: pd.DataFrame,
    metric_col: str,
    strategy: str,
    output_path: str,
) -> None:
    """
    Heatmap of a strategy-specific metric over (tau, seed_fraction).
    """
    sub = aggregate_all[aggregate_all["strategy"] == strategy]

    pivot = sub.pivot_table(
        index="tau",
        columns="seed_fraction",
        values=metric_col,
    )

    plt.figure(figsize=(6, 4))
    im = plt.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(im, label=metric_col)
    plt.xticks(range(len(pivot.columns)), [f"{c:.3f}" for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [f"{r:.3f}" for r in pivot.index])
    plt.xlabel("Seed fraction")
    plt.ylabel("Tau")
    plt.title(f"{metric_col} for {strategy} seeding")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_3d_surface_metric_over_params(
    aggregate_all: pd.DataFrame,
    metric_col: str,
    strategy: str,
    output_path: str,
) -> None:
    """
    Optional 3D surface plot of a metric over (tau, seed_fraction).
    """
    sub = aggregate_all[aggregate_all["strategy"] == strategy]
    pivot = sub.pivot_table(
        index="tau",
        columns="seed_fraction",
        values=metric_col,
    )

    taus = pivot.index.values
    seeds = pivot.columns.values
    Tau, Seed = np.meshgrid(seeds, taus)
    Z = pivot.values

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Tau, Seed, Z, cmap="viridis", edgecolor="none")
    fig.colorbar(surf, shrink=0.5, aspect=10, label=metric_col)
    ax.set_xlabel("Seed fraction")
    ax.set_ylabel("Tau")
    ax.set_zlabel(metric_col)
    ax.set_title(f"{metric_col} surface for {strategy} seeding")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================================
# Degree vs adoption time (node-level)
# ============================================================================

def maybe_plot_degree_vs_adoption_time(prefix: str, output_dir: str) -> None:
    """
    If the node-level adoption-time file exists, plot degree vs adoption time.
    """
    path = f"{prefix}_adoption_times_representative.csv"
    if not os.path.exists(path):
        LOGGER.info("No representative adoption-time file found (%s); skipping.", path)
        return

    df = pd.read_csv(path)
    LOGGER.info("Loaded representative adoption times from %s", path)

    plt.figure(figsize=(7, 4))
    for strategy, sub in df.groupby("strategy"):
        plt.hexbin(
            sub["degree"],
            sub["adoption_time"],
            gridsize=40,
            mincnt=1,
            bins="log",
            alpha=0.7,
            label=strategy,
        )

    plt.xscale("log")
    plt.xlabel("Degree (log scale)")
    plt.ylabel("Adoption time (steps)")
    plt.title("Degree vs adoption time (hexbin density)")
    cb = plt.colorbar()
    cb.set_label("log10(count)")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{prefix}_degree_vs_adoption_time.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    LOGGER.info("Saved degree vs adoption-time plot to %s", out_path)


# ============================================================================
# CLI + main
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise threshold diffusion results for Assignment 2.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix used for the CSV files, e.g. 'ass2'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save PNG figures (default: 'plots').",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Tau value for slice plots (default: smallest available).",
    )
    parser.add_argument(
        "--seed-fraction",
        type=float,
        default=None,
        help="Seed fraction for slice plots (default: smallest available).",
    )
    parser.add_argument(
        "--early-max-time",
        type=int,
        default=20,
        help="Max time step for early adoption plot.",
    )
    parser.add_argument(
        "--make-3d-surfaces",
        action="store_true",
        help="If set, also create 3D surface plots over (tau, seed_fraction).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    per_run_all, aggregate_all, ts_all, mean_curves_all = load_results(args.prefix)
    per_run_slice, ts_slice, mean_curves_slice, tau_used, sf_used = select_slice(
        per_run_all,
        ts_all,
        mean_curves_all,
        tau=args.tau,
        seed_fraction=args.seed_fraction,
    )

    LOGGER.info("Aggregate metrics for selected slice:\n%s",
                per_run_slice.groupby("strategy").agg(["mean", "std"]))

    # Per-slice plots.
    plot_mean_adoption_curves(
        mean_curves_slice,
        os.path.join(args.output_dir, f"{args.prefix}_tau{tau_used:.3f}_sf{sf_used:.3f}_adoption_curves.png"),
    )
    plot_early_adoption_curves(
        mean_curves_slice,
        os.path.join(args.output_dir, f"{args.prefix}_tau{tau_used:.3f}_sf{sf_used:.3f}_early_adoption.png"),
        max_time=args.early_max_time,
    )
    plot_final_fraction_bar(
        per_run_slice,
        os.path.join(args.output_dir, f"{args.prefix}_tau{tau_used:.3f}_sf{sf_used:.3f}_final_adoption.png"),
    )
    plot_steps_bar(
        per_run_slice,
        os.path.join(args.output_dir, f"{args.prefix}_tau{tau_used:.3f}_sf{sf_used:.3f}_steps_to_steady.png"),
    )
    plot_t50_bar(
        per_run_slice,
        os.path.join(args.output_dir, f"{args.prefix}_tau{tau_used:.3f}_sf{sf_used:.3f}_t50.png"),
    )
    plot_t90_bar(
        per_run_slice,
        os.path.join(args.output_dir, f"{args.prefix}_tau{tau_used:.3f}_sf{sf_used:.3f}_t90.png"),
    )
    plot_auc_bar(
        per_run_slice,
        os.path.join(args.output_dir, f"{args.prefix}_tau{tau_used:.3f}_sf{sf_used:.3f}_auc.png"),
    )
    plot_auc_vs_t90_scatter(
        per_run_slice,
        os.path.join(args.output_dir, f"{args.prefix}_tau{tau_used:.3f}_sf{sf_used:.3f}_auc_vs_t90.png"),
    )
    plot_t90_boxplot(
        per_run_slice,
        os.path.join(args.output_dir, f"{args.prefix}_tau{tau_used:.3f}_sf{sf_used:.3f}_t90_boxplot.png"),
    )

    # Heatmaps over parameter grid (for final_fraction and t_90).
    for strategy in aggregate_all["strategy"].unique():
        plot_heatmap_metric_over_params(
            aggregate_all,
            metric_col="mean_final_fraction",
            strategy=strategy,
            output_path=os.path.join(
                args.output_dir,
                f"{args.prefix}_heatmap_final_tau_seed_{strategy}.png",
            ),
        )
        plot_heatmap_metric_over_params(
            aggregate_all,
            metric_col="mean_t_90",
            strategy=strategy,
            output_path=os.path.join(
                args.output_dir,
                f"{args.prefix}_heatmap_t90_tau_seed_{strategy}.png",
            ),
        )

        if args.make_3d_surfaces:
            plot_3d_surface_metric_over_params(
                aggregate_all,
                metric_col="mean_t_90",
                strategy=strategy,
                output_path=os.path.join(
                    args.output_dir,
                    f"{args.prefix}_surface_t90_tau_seed_{strategy}.png",
                ),
            )

    # Degree vs adoption time (if file exists).
    maybe_plot_degree_vs_adoption_time(args.prefix, args.output_dir)

    LOGGER.info("All plots saved in directory: %s", args.output_dir)


if __name__ == "__main__":
    main()
