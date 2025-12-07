# Threshold Diffusion on the UvA Co-authorship Network

This repository implements Granovetter-style threshold dynamics on the UvA co-authorship network constructed in Assignment 1, as part of:

> **Assignment 2 — Dynamics on Networks: Thresholds and Spreading**
> Complex Systems & Policy (Model Based Decisions, 2025) — Universiteit van Amsterdam
> Lecturer: Michael Lees

## Brief Abstract

Nodes represent UvA authors, and undirected edges connect pairs of authors who have co-authored at least one publication. Each author adopts a behaviour when the fraction of active neighbours exceeds a personal threshold.

The script:

* Builds the largest connected component of the co-authorship network from `uva_dare_year_authors.csv`.
* Sweeps over lists of fhreshold values τ and seed fractions.
* For each (τ, seed fraction, seeding strategy) combination:

  * Runs multiple independent simulations in parallel.
  * Records adoption curves over time and summary metrics.
* Exports CSVs with per-run metrics, time series, and aggregated statistics.

The analysis quantifies how network structure and seeding strategy (random, high-degree, high-betweenness) shape diffusion speed and final adoption.

---

## Implementation Notes

### Network Construction

* Input: `uva_dare_year_authors.csv` with columns:

  * `year` – publication year
  * `authors` – comma-separated author identifiers
* For each publication:

  * Add all authors as nodes.
  * Add undirected edges between every pair of co-authors.
  * Maintain an edge attribute `weight` counting shared publications.
* After scanning all rows, the script extracts the largest connected component (LCC):

  * This removes isolated authors and tiny components that cannot meaningfully participate in large cascades.
* Logged summary: number of nodes, edges, and network density.

### Threshold Assignment

Two modes are supported:

* `fixed`: every node has τᵢ = τ (global threshold parameter).
* `uniform`: τᵢ ~ U(0, 1) independently for each node.

Thresholds are assigned once per experiment setting using a seeded NumPy RNG for reproducibility.

### Seeding Strategies

For a given seed fraction `s` and number of nodes `N`, the number of seeds is `⌊sN⌋` (at least 1). Three seeding strategies are implemented:

1. **Random** – nodes sampled uniformly at random.
2. **Degree** – top-degree nodes using a pre-computed degree ranking.
3. **Betweenness** – top nodes by (approximate) betweenness centrality:

   * Uses `networkx.betweenness_centrality` with a configurable sample size `k` for scalability on the large UvA network.

Centrality rankings are computed once and reused across all parameter combinations.

### Threshold Dynamics

The diffusion model is a standard Granovetter threshold process:

* Binary state: nodes are either inactive or active.
* Initial condition: seed nodes are active at time step 0.
* Synchronous updates:

  * At each step, inactive nodes compare the fraction of active neighbours to their threshold τᵢ.
  * If `(# active neighbours / degree) ≥ τᵢ`, the node becomes active.
  * Adoption is irreversible.
* Stopping rule:

  * Simulation ends when the active set no longer changes or `max_steps` is reached.
  * Adoption history (fraction of active nodes per time step) is recorded.

### Metrics

For each single simulation run, the script computes:

* `n_steps` — realised number of update steps until steady state.
* `final_fraction` — final fraction of active nodes.
* `t_50` — first time step at which adoption ≥ 50% (or −1 if never reached).
* `t_90` — first time step at which adoption ≥ 90% (or −1 if never reached).
* `auc` — area under the adoption curve (sum of fractions over time).

These metrics are later aggregated (mean, standard deviation) over `n_runs` for each `(τ, seed_fraction, strategy)`.

### Parallelisation

* Simulations are embarrassingly parallel across runs and strategies.
* The script uses a `ProcessPoolExecutor` with a configurable number of workers (`--n-workers`) to distribute `_single_run_worker` calls across CPU cores.
* Each worker run:

  * Derives its own RNG seed from a base seed plus run id and strategy hash.
  * Selects seeds, runs the diffusion, and returns metrics and adoption history.

---

## Usage

Example command:

```bash
python main.py \
  --uva-csv-path uva_dare_year_authors.csv \
  --tau-values 0.05,0.10,0.15,0.20,0.25,0.30 \
  --seed-fractions 0.0025,0.005,0.01,0.02,0.05 \
  --threshold-mode fixed \
  --n-runs 10 \
  --max-steps 200 \
  --n-workers 14 \
  --betweenness-k 200 \
  --output-prefix ass2 \
  --save-representative-adoption-times
```

### Key Arguments

* `--uva-csv-path`
  Path to `uva_dare_year_authors.csv`.

* `--tau-values`
  Comma-separated list of τ values to sweep.

* `--seed-fractions`
  Comma-separated list of seed fractions (e.g. 0.01 = 1% of nodes).

* `--threshold-mode`
  `fixed` or `uniform`.

* `--n-runs`
  Number of repetitions per strategy per parameter combination.

* `--max-steps`
  Maximum synchronous update steps per simulation.

* `--n-workers`
  Number of worker processes for parallel simulations.

* `--betweenness-k`
  Sample size for approximate betweenness centrality (trade-off between speed and accuracy).

* `--output-prefix`
  Prefix for all output files.

* `--save-representative-adoption-times`
  If set, runs an additional set of simulations to record **node-level adoption times** for one representative `(τ, seed_fraction)` pair.

---

## Outputs

Given an `output-prefix` (e.g. `ass2`), the script writes:

* `ass2_per_run_all.csv`
  Per-run metrics across all τ, seed fractions, and strategies.
  Columns include: `strategy`, `run_id`, `n_steps`, `final_fraction`, `t_50`, `t_90`, `auc`, `tau`, `seed_fraction`.

* `ass2_aggregate_all.csv`
  Aggregated metrics by `(strategy, tau, seed_fraction)` with means and standard deviations.

* `ass2_ts_all.csv`
  Time series table with one row per `(strategy, run_id, time_step)`:

  * `fraction_active`, `tau`, `seed_fraction`.

* `ass2_mean_curves_all.csv`
  Mean adoption curves over runs:

  * `strategy`, `time_step`, `mean_fraction`, `std_fraction`, `tau`, `seed_fraction`.

* `ass2_config.json`
  Configuration and network summary (parameter lists, runs, steps, number of nodes/edges, density).

---

## Repository Structure

Minimal structure for this assignment:

```text
.
├── main.py                     # Core script: network construction + experiments
├── uva_dare_year_authors.csv   # Input data
├── ass2_per_run_all.csv        # Generated results (example)
├── ass2_aggregate_all.csv
├── ass2_ts_all.csv
├── ass2_mean_curves_all.csv
└── README.md                   # This file
```

---

## Limitations & Scope

* Focuses on **threshold contagion on a fixed empirical network** (co-authorship at UvA).
* Thresholds are either global or i.i.d. uniform; more realistic heterogeneity (e.g. degree-dependent thresholds) is not explored here.
* No temporal evolution of the network; edges are treated as static.
* Interpretation is limited to behaviour adoption / information diffusion analogies rather than specific empirical behaviours.

---



VM setup:

n2-highcpu-16 16 vCPU, 8 core, 16 GB memory

sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y

python3 -m venv env
source env/bin/activate

pip install --upgrade pip
pip install networkx numpy pandas matplotlib tqdm





