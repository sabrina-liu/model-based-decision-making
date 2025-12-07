"""
main.py

Assignment 2 — Dynamics on Networks: Thresholds and Spreading
14244861    Liu, Sabrina    30 November 2025

Granovetter-style threshold dynamics on the UvA co-authorship network.

Features
--------
- Builds co-authorship network on the full dataset from Assignment 1 (LLC only, uva_dare_year_authors.csv)
- Sweeps over:
    - threshold values tau (list)
    - seed fractions (list)
- For each (tau, seed fraction) combination:
    - runs parallel simulations for each seeding strategy
    - computes:
        - n_steps              : number of steps to steady state
        - final_fraction       : final adoption fraction
        - t_50                 : time to reach 50% adoption
        - t_90                 : time to reach 90% adoption
        - auc                  : area under the adoption curve
- Saves combined CSVs across all parameter combinations:
    - <prefix>_per_run_all.csv     : per-run data
    - <prefix>_aggregate_all.csv   : aggregate statistics (mean, std) per (tau, seed fraction, strategy)
    - <prefix>_ts_all.csv          : mean adoption curves per (tau, seed fraction, strategy)
    - <prefix>_mean_curves_all.csv : mean adoption curves averaged over all (tau, seed fraction) per strategy
    - (optional) <prefix>_adoption_times_representative.csv : node-level adoption times and degrees for one representative setting
- Logs timings and progress

Usage
-----
python main.py \
  --uva-csv-path uva_dare_year_authors.csv \
  --tau-values 0.05,0.10,0.15,0.20,0.25,0.30 \
  --seed-fractions 0.0025,0.005,0.01,0.02,0.05 \
  --n-runs 10 \
  --max-steps 200 \
  --n-workers 14 \
  --betweenness-k 200 \
  --output-prefix ass2 \
  --save-representative-adoption-times


"""


from __future__ import annotations

import argparse
import itertools
import logging
import math
import os
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace, dataclass
from typing import Dict, Hashable, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------
# Logging configuration
# ---------------------

# Basic logger for entire file. Prints timestamps, log levels, and messages to stdout.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


# -----------------------
# Dataclass configuration
# -----------------------

@dataclass
class ThresholdExperimentConfig:
    """
    Container for experiment configuration parameters.

    seed_fraction    : float
        Fraction of nodes used as initial seeds in each run
    tau              : float
        Global threshold τ when `threshold_mode == "fixed"`
    threshold_mode   : str
        How individual τ_i are assigned:
            - "fixed"   : all nodes share the same τ
            - "uniform" : τ_i ~ U(0, 1) independently
    n_runs           : int
        Number of independent runs per seeding strategy
    max_steps        : int
        Maximum number of synchronous update steps per simulation
    n_workers        : int
        Number of worker processes to use in the process pool
    base_random_seed : int
        Base seed used to derive run-specific RNG seeds for reproducibility
    betweenness_k     : int | None
        Sample size k nodes for approximate betweenness centrality
    """

    seed_fraction   : float = 0.01
    tau             : float = 0.2
    threshold_mode  : str = "fixed"
    n_runs          : int = 10
    max_steps       : int = 200
    n_workers       : int = max(1, (os.cpu_count() or 2) - 1)
    base_random_seed: int = 42
    betweenness_k   : int | None = 200


# ---------------------------
# Build co-authorship network
# ---------------------------

def build_coauthorship_graph(csv_path: str) -> nx.Graph:
    """
    Construct an undirected co-authorship network from `uva_dare_year_authors.csv`

    CSV format:
        - "year"    : publication year (int)
        - "authors" : comma-separated list of author identifiers

    For each publication:
        - Parse authors
        - Add authors as nodes
        - Add undirected edges between every pair of co-authors, with a 'weight' counting the number of shared publications

    We then extract the largest connected component (LCC)

    Parameters
    ----------
    csv_path : str
        Path to `uva_dare_year_authors.csv`

    Returns
    -------
    G : nx.Graph
        Co-authorship network (LCC)

    """
    t0 = time.perf_counter()
    LOGGER.info("Reading UvA DARE csv from %s", csv_path)
    df = pd.read_csv(csv_path)

    LOGGER.info("Building co-authorship graph from %d publications", len(df))
    G = nx.Graph()

    for _, row in df.iterrows():
        authors_raw = str(row["authors"]).strip().strip(",")
        if not authors_raw:
            continue

        authors = [a.strip() for a in authors_raw.split(",") if a.strip()]
        if not authors:
            continue

        for a in authors:
            if not G.has_node(a):
                G.add_node(a)

        if len(authors) >= 2:
            for u, v in itertools.combinations(authors, 2):
                if G.has_edge(u, v):
                    G[u][v]["weight"] = G[u][v].get("weight", 1) + 1
                else:
                    G.add_edge(u, v, weight=1)

    LOGGER.info(
        "Initial graph: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )

    if not nx.is_connected(G):
        LOGGER.info("Extracting LCC")
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc_nodes).copy()

    LOGGER.info(
        "LCC: %d nodes, %d edges, density = %.5f",
        G.number_of_nodes(),
        G.number_of_edges(),
        nx.density(G),
    )
    LOGGER.info("Graph building + LLC extraction took %.2f seconds",
                time.perf_counter() - t0)
    
    return G


# -------------------------------------------
# Threshold assignment and seeding strategies
# -------------------------------------------

def assign_thresholds(
    G: nx.Graph,
    mode: str = "fixed",
    tau: float = 0.2,
    random_state: int | None = None,
) -> Dict[Hashable, float]:
    """
    Assign adoption threshold τ_i to each node

    Parameters
    ----------
    G            : nx.Graph
    mode         : {"fixed", "uniform"}
    tau          : float
    random_state : int or None

    Returns
    -------
    thresholds : dict[node, float]

    """
    rng = np.random.default_rng(random_state)
    thresholds: Dict[Hashable, float] = {}

    if mode == "fixed":
        for node in G.nodes():
            thresholds[node] = float(tau)
    elif mode == "uniform":
        for node in G.nodes():
            thresholds[node] = float(rng.uniform(0.0, 1.0))
    else:
        raise ValueError(f"Unsupported threshold mode: {mode}")

    return thresholds


def compute_degree_ranking(G: nx.Graph) -> List[Hashable]:
    """
    Return nodes sorted by degree in descending order

    """
    degree_pairs = sorted(G.degree(), key=lambda x: x[1], reverse=True)

    return [node for node, _deg in degree_pairs]


def compute_betweenness_ranking(
    G: nx.Graph,
    k: int | None = 200,
    random_state: int | None = 0,
) -> List[Hashable]:
    """
    Compute (approximate) betweenness centrality ranking

    Parameters
    ----------
    G            : nx.Graph
    k            : int or None
        Sample size for approximation (None = exact)
    random_state : int or None

    Returns
    -------
    ranked_nodes : list

    """
    t0 = time.perf_counter()
    LOGGER.info(
        "Computing betweenness centrality with k=%s (n=%d, m=%d).",
        k, G.number_of_nodes(), G.number_of_edges(),
    )
    btwn_dict = nx.betweenness_centrality(G, k=k, normalized=True, seed=random_state)
    LOGGER.info("Betweenness centrality finished in %.2f seconds.",
                time.perf_counter() - t0)
    ranked = sorted(btwn_dict.items(), key=lambda x: x[1], reverse=True)

    return [node for node, _score in ranked]


def initialise_seeds(
    G: nx.Graph,
    n_seeds: int,
    strategy: str,
    degree_ranking: List[Hashable] | None = None,
    betweenness_ranking: List[Hashable] | None = None,
    rng: np.random.Generator | None = None,
) -> List[Hashable]:
    """
    Select seed nodes according to a specified strategy

    """
    if rng is None:
        rng = np.random.default_rng()

    nodes = list(G.nodes())
    n_seeds = max(1, min(n_seeds, len(nodes)))

    if strategy == "random":
        seeds = list(rng.choice(nodes, size=n_seeds, replace=False))
    elif strategy == "degree":
        if degree_ranking is None:
            raise ValueError("Degree ranking must be provided for 'degree'")
        seeds = degree_ranking[:n_seeds]
    elif strategy == "betweenness":
        if betweenness_ranking is None:
            raise ValueError("Betweenness ranking must be provided for 'betweeness'")
        seeds = betweenness_ranking[:n_seeds]
    else:
        raise ValueError(f"Unknown seeding strategy: {strategy}")

    return seeds


# ----------------------------
# Threshold diffusion dynamics
# ----------------------------

def run_threshold_diffusion(
    G: nx.Graph,
    thresholds: Dict[Hashable, float],
    seeds: Iterable[Hashable],
    max_steps: int,
    return_adoption_times: bool = False,
) -> Tuple[List[float], int, Dict[Hashable, int] | None]:
    """
    Run a single Granovetter-style threshold diffusion simulation

    Parameters
    ----------
    G                     : nx.Graph
    thresholds            : dict[node, float]
    seeds                 : iterable[node]
    max_steps             : int
    return_adoption_times : bool
        If True, also return a mapping node (time step when it first became active (0 for seeds, -1 if never adopts))

    Returns
    -------
    adoption_history : list of float
    n_steps_realised : int
    adoption_times   : dict of None

    """
    active_set = set(seeds)
    n_nodes = G.number_of_nodes()

    adoption_times: Dict[Hashable, int] | None = None
    if return_adoption_times:
        adoption_times = {node: 0 for node in active_set}

    adoption_history: List[float] = [len(active_set) / n_nodes]

    for t in range(1, max_steps + 1):
        new_active_set = set(active_set)

        for node in G.nodes():
            if node in active_set:
                continue

            neighbours = list(G.neighbors(node))
            if not neighbours:
                continue

            n_active_neighbours = sum(1 for nbr in neighbours if nbr in active_set)
            frac_active = n_active_neighbours / len(neighbours)

            if frac_active >= thresholds[node]:
                new_active_set.add(node)
                if adoption_times is not None and node not in adoption_times:
                    adoption_times[node] = t

        if new_active_set == active_set:
            active_set = new_active_set
            adoption_history.append(len(active_set) / n_nodes)
            n_steps_realised = t
            break

        active_set = new_active_set
        adoption_history.append(len(active_set) / n_nodes)
        n_steps_realised = t

    if return_adoption_times and adoption_times is not None:
        for node in G.nodes():
            if node not in adoption_times:
                adoption_times[node] = -1

    return adoption_history, n_steps_realised, adoption_times


# -----------------------
# Adoption curves metrics
# -----------------------

def compute_curve_metrics(
    adoption_history: List[float],
    t_50_target: float = 0.5,
    t_90_target: float = 0.9,
) -> Dict[str, float]:
    """
    Compute summary metrics from an adoption time series

    Returns
    -------
    dict with keys:
        - final_fraction
        - t_50
        - t_90
        - auc

    """
    final_fraction = float(adoption_history[-1])

    def first_time_at_or_above(target: float) -> int:
        for t_idx, frac in enumerate(adoption_history):
            if frac >= target:
                return t_idx
        return -1

    t_50 = first_time_at_or_above(t_50_target)
    t_90 = first_time_at_or_above(t_90_target)
    auc = float(sum(adoption_history))

    return {
        "final_fraction": final_fraction,
        "t_50": float(t_50),
        "t_90": float(t_90),
        "auc": auc,
    }


# ---------------
# Parallel worker
# ---------------

def _single_run_worker(
    strategy: str,
    run_id: int,
    G: nx.Graph,
    thresholds: Dict[Hashable, float],
    degree_ranking: List[Hashable],
    betweenness_ranking: List[Hashable],
    config: ThresholdExperimentConfig,
) -> Dict[str, object]:
    """
    Worker function for one simulation run

    """
    try:
        run_seed = config.base_random_seed + run_id * 1000 + hash(strategy) % 997
        rng = np.random.default_rng(run_seed)

        n_nodes = G.number_of_nodes()
        n_seeds = max(1, math.floor(config.seed_fraction * n_nodes))

        seeds = initialise_seeds(
            G=G,
            n_seeds=n_seeds,
            strategy=strategy,
            degree_ranking=degree_ranking,
            betweenness_ranking=betweenness_ranking,
            rng=rng,
        )

        adoption_history, n_steps, _ = run_threshold_diffusion(
            G=G,
            thresholds=thresholds,
            seeds=seeds,
            max_steps=config.max_steps,
            return_adoption_times=False,
        )

        metrics = compute_curve_metrics(adoption_history)

        return {
            "strategy"         : strategy,
            "run_id"           : run_id,
            "adoption_history" : adoption_history,
            "n_steps"          : n_steps,
            "final_fraction"   : metrics["final_fraction"],
            "t_50"             : metrics["t_50"],
            "t_90"             : metrics["t_90"],
            "auc"              : metrics["auc"],
        }
    
    except Exception as e:
        LOGGER.error(f"Error in worker (strategy={strategy}, run_id={run_id}): {e}")
        raise


# -----------------------------------------------------
# Experiments for a single (tau, seed_fraction) setting
# -----------------------------------------------------

def run_parallel_experiments_single_setting(
    G: nx.Graph,
    base_config: ThresholdExperimentConfig,
    degree_ranking: List[Hashable],
    betweenness_ranking: List[Hashable],
    tau: float,
    seed_fraction: float,
    strategies: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run experiments for one combination of (tau, seed_fraction)

    Returns
    -------
    per_run_df, aggregate_df, ts_df, mean_curves_df

    """
    if strategies is None:
        strategies = ["random", "degree", "betweenness"]

    config = replace(base_config, tau=tau, seed_fraction=seed_fraction)

    LOGGER.info(
        "Running experiments for tau=%.3f, seed_fraction=%.3f",
        tau, seed_fraction,
    )

    thresholds = assign_thresholds(
        G=G,
        mode=config.threshold_mode,
        tau=config.tau,
        random_state=config.base_random_seed,
    )

    tasks = list(itertools.product(strategies, range(config.n_runs)))
    per_run_results: List[Dict[str, object]] = []
    ts_rows: List[Dict[str, object]] = []

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
        future_to_task = {}
        for strategy, run_id in tasks:
            future = executor.submit(
                _single_run_worker,
                strategy,
                run_id,
                G,
                thresholds,
                degree_ranking,
                betweenness_ranking,
                config,
            )
            future_to_task[future] = (strategy, run_id)

        for future in as_completed(future_to_task):
            strategy, run_id = future_to_task[future]
            res = future.result()
            per_run_results.append(res)

            adoption_history = res["adoption_history"]
            for t_idx, frac in enumerate(adoption_history):
                ts_rows.append(
                    {
                        "strategy": strategy,
                        "run_id": run_id,
                        "time_step": t_idx,
                        "fraction_active": frac,
                    }
                )

    LOGGER.info(
        "Finished tau=%.3f, seed_fraction=%.3f in %.2f seconds.",
        tau, seed_fraction, time.perf_counter() - t0,
    )

    # per-run summary
    summary_rows = []
    for res in per_run_results:
        summary_rows.append(
            {
                "strategy"      : res["strategy"],
                "run_id"        : res["run_id"],
                "n_steps"       : res["n_steps"],
                "final_fraction": res["final_fraction"],
                "t_50"          : res["t_50"],
                "t_90"          : res["t_90"],
                "auc"           : res["auc"],
            }
        )
    per_run_df = pd.DataFrame(summary_rows)
    ts_df = pd.DataFrame(ts_rows)

    # aggregate by strategy
    aggregate_df = (
        per_run_df
        .groupby("strategy")
        .agg(
            mean_final_fraction=("final_fraction", "mean"),
            std_final_fraction=("final_fraction", "std"),
            mean_n_steps=("n_steps", "mean"),
            std_n_steps=("n_steps", "std"),
            mean_t_50=("t_50", "mean"),
            std_t_50=("t_50", "std"),
            mean_t_90=("t_90", "mean"),
            std_t_90=("t_90", "std"),
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
        )
        .reset_index()
    )

    # mean adoption curves by strategy
    mean_curves_df = (
        ts_df
        .groupby(["strategy", "time_step"])
        .agg(
            mean_fraction=("fraction_active", "mean"),
            std_fraction=("fraction_active", "std"),
        )
        .reset_index()
    )

    mean_curves_df["std_fraction"] = mean_curves_df["std_fraction"].fillna(0.0)

    return per_run_df, aggregate_df, ts_df, mean_curves_df


# ----------------------------------------------------------------
# Representative adoption times (degree vs adoption time analysis)
# ----------------------------------------------------------------

def compute_representative_adoption_times(
    G: nx.Graph,
    config: ThresholdExperimentConfig,
    degree_ranking: List[Hashable],
    betweenness_ranking: List[Hashable],
    tau: float,
    seed_fraction: float,
    output_prefix: str,
) -> None:
    """
    For a single (tau, seed_fraction), run one simulation per strategy with adoption-time tracking, and save node-level adoption times + degree

    Output
    ------
    <output_prefix>_adoption_times_representative.csv
        Columns: node, degree, strategy, adoption_time, tau, seed_fraction
        
    """
    LOGGER.info(
        "Computing representative adoption times for tau=%.3f, seed_fraction=%.3f",
        tau, seed_fraction,
    )

    config_rep = replace(config, tau=tau, seed_fraction=seed_fraction)
    thresholds = assign_thresholds(
        G=G,
        mode=config_rep.threshold_mode,
        tau=config_rep.tau,
        random_state=config_rep.base_random_seed,
    )
    deg_dict = dict(G.degree())

    strategies = ["random", "degree", "betweenness"]
    rows: List[Dict[str, object]] = []

    for strategy in strategies:
        rng = np.random.default_rng(config_rep.base_random_seed + hash(strategy) % 997)
        n_nodes = G.number_of_nodes()
        n_seeds = max(1, math.floor(config_rep.seed_fraction * n_nodes))

        seeds = initialise_seeds(
            G=G,
            n_seeds=n_seeds,
            strategy=strategy,
            degree_ranking=degree_ranking,
            betweenness_ranking=betweenness_ranking,
            rng=rng,
        )

        _, _, adoption_times = run_threshold_diffusion(
            G=G,
            thresholds=thresholds,
            seeds=seeds,
            max_steps=config_rep.max_steps,
            return_adoption_times=True,
        )

        assert adoption_times is not None
        for node, t_adopt in adoption_times.items():
            rows.append(
                {
                    "node": node,
                    "degree": deg_dict.get(node, 0),
                    "strategy": strategy,
                    "adoption_time": t_adopt,
                    "tau": tau,
                    "seed_fraction": seed_fraction,
                }
            )

    df = pd.DataFrame(rows)
    out_path = f"{output_prefix}_adoption_times_representative.csv"
    df.to_csv(out_path, index=False)
    LOGGER.info("Saved representative adoption times to %s", out_path)


# ----------
# CLI + main
# ----------

def parse_float_list(s: str) -> List[float]:
    """
    Parse a comma-separated string of floats into a list

    """
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the script

    """
    parser = argparse.ArgumentParser(
        description="Run threshold diffusion on the UvA co-authorship network",
    )
    parser.add_argument(
        "--uva-csv-path",
        type=str,
        default="uva_dare_year_authors.csv",
        help="Path to uva_dare_year_authors.csv (default: ./uva_dare_year_authors.csv)",
    )
    parser.add_argument(
        "--tau-values",
        type=parse_float_list,
        default=parse_float_list("0.2"),
        help="Comma-separated list of tau values",
    )
    parser.add_argument(
        "--seed-fractions",
        type=parse_float_list,
        default=parse_float_list("0.01"),
        help="Comma-separated list of seed fractions",
    )
    parser.add_argument(
        "--threshold-mode",
        type=str,
        choices=["fixed", "uniform"],
        default="fixed",
        help="Threshold assignment mode",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of runs per strategy per parameter setting",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max update steps per simulation",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--betweenness-k",
        type=int,
        default=200,
        help="Sample size k for approximate betweenness (default: 200)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="ass2",
        help="Prefix for output CSVs (default: ass2)",
    )
    parser.add_argument(
        "--save-representative-adoption-times",
        action="store_true",
        help="If set, also save node-level adoption times for one setting",
    )
    return parser.parse_args()


def main() -> None:

    args = parse_args()
    LOGGER.info("Starting Assignment 2 threshold experiments")
    LOGGER.info("Tau values: %s", args.tau_values)
    LOGGER.info("Seed fractions: %s", args.seed_fractions)

    # 1. Build network
    G = build_coauthorship_graph(
        csv_path=args.uva_csv_path,
        min_year=args.min_year,
        max_year=args.max_year,
    )

    # 2. Compute centrality rankings
    LOGGER.info("Precomputing centrality rankings")
    degree_ranking = compute_degree_ranking(G)
    betweenness_ranking = compute_betweenness_ranking(
        G,
        k=args.betweenness_k,
        random_state=42,
    )

    # 3. Base configuration
    base_config = ThresholdExperimentConfig(
        seed_fraction=args.seed_fractions[0], 
        tau=args.tau_values[0],             
        threshold_mode=args.threshold_mode,
        n_runs=args.n_runs,
        max_steps=args.max_steps,
        n_workers=args.n_workers,
        base_random_seed=42,
        betweenness_k=args.betweenness_k,
    )

    # 4. Parameter sweep
    per_run_all: List[pd.DataFrame] = []
    aggregate_all: List[pd.DataFrame] = []
    ts_all: List[pd.DataFrame] = []
    mean_curves_all: List[pd.DataFrame] = []

    for tau in args.tau_values:
        for sf in args.seed_fractions:
            per_run_df, agg_df, ts_df, mc_df = run_parallel_experiments_single_setting(
                G,
                base_config,
                degree_ranking,
                betweenness_ranking,
                tau=tau,
                seed_fraction=sf,
            )

            per_run_df["tau"] = tau
            per_run_df["seed_fraction"] = sf
            agg_df["tau"] = tau
            agg_df["seed_fraction"] = sf
            ts_df["tau"] = tau
            ts_df["seed_fraction"] = sf
            mc_df["tau"] = tau
            mc_df["seed_fraction"] = sf

            per_run_all.append(per_run_df)
            aggregate_all.append(agg_df)
            ts_all.append(ts_df)
            mean_curves_all.append(mc_df)

    per_run_all_df = pd.concat(per_run_all, ignore_index=True)
    aggregate_all_df = pd.concat(aggregate_all, ignore_index=True)
    ts_all_df = pd.concat(ts_all, ignore_index=True)
    mean_curves_all_df = pd.concat(mean_curves_all, ignore_index=True)

    # 5. Save combined outputs
    per_run_path = f"{args.output_prefix}_per_run_all.csv"
    aggregate_path = f"{args.output_prefix}_aggregate_all.csv"
    ts_path = f"{args.output_prefix}_ts_all.csv"
    mean_curves_path = f"{args.output_prefix}_mean_curves_all.csv"

    per_run_all_df.to_csv(per_run_path, index=False)
    aggregate_all_df.to_csv(aggregate_path, index=False)
    ts_all_df.to_csv(ts_path, index=False)
    mean_curves_all_df.to_csv(mean_curves_path, index=False)

    LOGGER.info("Saved per-run results to: %s", per_run_path)
    LOGGER.info("Saved aggregate results to: %s", aggregate_path)
    LOGGER.info("Saved time series to: %s", ts_path)
    LOGGER.info("Saved mean curves to: %s", mean_curves_path)

    # 6. Save a small JSON with configuration + graph summary
    config_info = {
        "tau_values"     : args.tau_values,
        "seed_fractions" : args.seed_fractions,
        "threshold_mode" : args.threshold_mode,
        "n_runs"         : args.n_runs,
        "max_steps"      : args.max_steps,
        "n_workers"      : args.n_workers,
        "betweenness_k"  : args.betweenness_k,
        "min_year"       : args.min_year,
        "max_year"       : args.max_year,
        "n_nodes"        : G.number_of_nodes(),
        "n_edges"        : G.number_of_edges(),
        "density"        : nx.density(G),
    }
    json_path = f"{args.output_prefix}_config.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2)
    LOGGER.info("Saved configuration to: %s", json_path)

    # 7. Optional representative node-level adoption times
    if args.save_representative_adoption_times:
        rep_tau = args.tau_values[0]
        rep_sf = args.seed_fractions[0]
        compute_representative_adoption_times(
            G,
            base_config,
            degree_ranking,
            betweenness_ranking,
            tau=rep_tau,
            seed_fraction=rep_sf,
            output_prefix=args.output_prefix,
        )

    LOGGER.info("All experiments completed successfully.")


if __name__ == "__main__":
    main()
