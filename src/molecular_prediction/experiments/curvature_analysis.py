# src/molecular_prediction/experiments/curvature_analysis.py
#
# Ollivier-Ricci curvature analysis for over-squashing study.
#
# Computes per-edge Ollivier-Ricci curvature on the topological (covalent)
# molecular graph using the GraphRicciCurvature library, then derives a
# per-molecule bottleneck score based on the fraction of edges with highly
# negative curvature.

import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot
import torch
from torch_geometric.data import Dataset

from configs.config import Config
from src.molecular_prediction.data.dataset import load_splits
from src.molecular_prediction.experiments.main_comparison import (
    MODEL_NAMES,
    TARGET_NAMES,
)
from src.molecular_prediction.experiments.per_molecule_eval import (
    run_per_molecule_evaluation,
)

# Default threshold: edges with curvature below this are considered bottlenecks
DEFAULT_CURVATURE_THRESHOLD: float = -0.5


def pyg_to_networkx(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    """Convert a PyG edge_index to an undirected NetworkX graph.

    Args:
        edge_index: Edge connectivity of shape [2, num_edges].
        num_nodes: Total number of nodes in the graph.

    Returns:
        Undirected NetworkX graph.
    """
    g: nx.Graph = nx.Graph()
    g.add_nodes_from(range(num_nodes))

    src: list[int] = edge_index[0].tolist()
    dst: list[int] = edge_index[1].tolist()

    for s, d in zip(src, dst):
        if s < d:  # avoid duplicate undirected edges
            g.add_edge(s, d)

    return g


def _node_distribution(
    g: nx.Graph,
    node: int,
    alpha: float,
) -> tuple[list[int], np.ndarray]:
    """Build the lazy random walk distribution centred at a node.

    Places mass alpha on the node itself and distributes (1 - alpha)
    uniformly among its neighbours.

    Args:
        g: Undirected NetworkX graph.
        node: Centre node.
        alpha: Laziness parameter (mass on the node itself).

    Returns:
        Tuple of (support nodes as list, probability weights as array).
    """
    neighbours: list[int] = list(g.neighbors(node))
    deg: int = len(neighbours)

    if deg == 0:
        return [node], np.array([1.0])

    support: list[int] = [node] + neighbours
    weights: np.ndarray = np.zeros(len(support))
    weights[0] = alpha
    weights[1:] = (1.0 - alpha) / deg

    return support, weights


def compute_ollivier_ricci_curvature(
    g: nx.Graph,
    alpha: float = 0.5,
) -> dict[tuple[int, int], float]:
    """Compute Ollivier-Ricci curvature for every edge in the graph.

    For each edge (u, v), builds lazy random walk distributions mu_u and mu_v,
    computes the Wasserstein-1 distance W(mu_u, mu_v) using the shortest-path
    metric, and returns kappa(u,v) = 1 - W(mu_u, mu_v) / d(u,v).

    Since the graph is unweighted, d(u,v) = 1 for all edges.

    Uses the POT library (Python Optimal Transport) for the EMD computation.

    Args:
        g: Undirected NetworkX graph.
        alpha: Laziness parameter for the ORC computation.
            alpha=0.5 is standard (equal weight to staying vs moving).

    Returns:
        Dict mapping (u, v) edge tuple to its curvature value.
    """
    if g.number_of_edges() == 0:
        return {}

    # Precompute all-pairs shortest paths (cheap for small molecular graphs)
    nodes: list[int] = list(g.nodes())
    node_to_idx: dict[int, int] = {n: i for i, n in enumerate(nodes)}
    n: int = len(nodes)

    sp_lengths: dict[int, dict[int, int]] = dict(nx.all_pairs_shortest_path_length(g))

    # Build full distance matrix
    dist_matrix: np.ndarray = np.zeros((n, n))
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if ni in sp_lengths and nj in sp_lengths[ni]:
                dist_matrix[i, j] = float(sp_lengths[ni][nj])
            else:
                # Disconnected components — shouldn't happen in molecules
                dist_matrix[i, j] = float(n)

    curvatures: dict[tuple[int, int], float] = {}

    for u, v in g.edges():
        # Build distributions
        support_u: list[int]
        weights_u: np.ndarray
        support_u, weights_u = _node_distribution(g, u, alpha)

        support_v: list[int]
        weights_v: np.ndarray
        support_v, weights_v = _node_distribution(g, v, alpha)

        # Build cost matrix between supports using precomputed shortest paths
        idx_u: list[int] = [node_to_idx[s] for s in support_u]
        idx_v: list[int] = [node_to_idx[s] for s in support_v]
        cost: np.ndarray = dist_matrix[np.ix_(idx_u, idx_v)]

        # Compute Wasserstein-1 distance via EMD
        wasserstein: float = float(ot.emd2(weights_u, weights_v, cost))

        # d(u, v) = 1 for unweighted graphs
        kappa: float = 1.0 - wasserstein
        curvatures[(u, v)] = kappa

    return curvatures


def compute_bottleneck_score(
    curvatures: dict[tuple[int, int], float],
    threshold: float = DEFAULT_CURVATURE_THRESHOLD,
) -> float:
    """Compute the bottleneck score for a molecule.

    The bottleneck score is the fraction of edges with curvature below
    the given threshold.

    Args:
        curvatures: Dict mapping edge -> curvature value.
        threshold: Curvature threshold below which an edge is a bottleneck.

    Returns:
        Fraction of bottleneck edges in [0, 1].
    """
    if len(curvatures) == 0:
        return 0.0

    n_bottleneck: int = sum(1 for c in curvatures.values() if c < threshold)
    return n_bottleneck / len(curvatures)


def compute_curvature_stats_for_dataset(
    test_dataset: Dataset,
    threshold: float = DEFAULT_CURVATURE_THRESHOLD,
) -> dict[str, list]:
    """Compute curvature statistics for every molecule in the test set.

    Args:
        test_dataset: Test dataset (PyG).
        threshold: Curvature threshold for bottleneck classification.

    Returns:
        Dict with keys:
            'bottleneck_scores': list[float] — one per molecule
            'mean_curvatures': list[float] — mean edge curvature per molecule
            'min_curvatures': list[float] — minimum edge curvature per molecule
            'num_edges': list[int] — number of undirected edges per molecule
            'num_bottleneck_edges': list[int] — edges below threshold
    """
    n: int = len(test_dataset)
    bottleneck_scores: list[float] = []
    mean_curvatures: list[float] = []
    min_curvatures: list[float] = []
    num_edges: list[int] = []
    num_bottleneck_edges: list[int] = []

    for i in range(n):
        if (i + 1) % 2000 == 0 or i == 0:
            print(f"  Computing curvature for molecule {i + 1}/{n}...")

        data = test_dataset[i]
        g: nx.Graph = pyg_to_networkx(data.edge_index, data.num_nodes)

        curvatures: dict[tuple[int, int], float] = compute_ollivier_ricci_curvature(g)

        if len(curvatures) == 0:
            bottleneck_scores.append(0.0)
            mean_curvatures.append(0.0)
            min_curvatures.append(0.0)
            num_edges.append(0)
            num_bottleneck_edges.append(0)
            continue

        curv_values: list[float] = list(curvatures.values())
        score: float = compute_bottleneck_score(curvatures, threshold)
        n_bottleneck: int = sum(1 for c in curv_values if c < threshold)

        bottleneck_scores.append(score)
        mean_curvatures.append(float(np.mean(curv_values)))
        min_curvatures.append(float(np.min(curv_values)))
        num_edges.append(len(curvatures))
        num_bottleneck_edges.append(n_bottleneck)

    return {
        "bottleneck_scores": bottleneck_scores,
        "mean_curvatures": mean_curvatures,
        "min_curvatures": min_curvatures,
        "num_edges": num_edges,
        "num_bottleneck_edges": num_bottleneck_edges,
    }


def assign_quartiles(scores: list[float]) -> np.ndarray:
    """Assign each molecule to a quartile based on bottleneck score.

    Quartile 0 = lowest bottleneck (least over-squashing risk).
    Quartile 3 = highest bottleneck (most over-squashing risk).

    Molecules with score == 0.0 are always in quartile 0.
    Remaining molecules are split into quartiles by percentile.

    Args:
        scores: List of bottleneck scores, one per molecule.

    Returns:
        Array of quartile assignments (0–3), shape [N].
    """
    arr: np.ndarray = np.array(scores)
    quartiles: np.ndarray = np.zeros(len(arr), dtype=int)

    # Many molecules may have score 0 (no bottleneck edges),
    # so we use quantile-based binning on nonzero scores.
    nonzero_mask: np.ndarray = arr > 0.0

    if nonzero_mask.sum() == 0:
        return quartiles

    nonzero_scores: np.ndarray = arr[nonzero_mask]
    # Molecules with score 0 stay in quartile 0.
    # Split nonzero into 3 bins (quartiles 1–3).
    percentiles: list[float] = [33.3, 66.7]
    bins: np.ndarray = np.percentile(nonzero_scores, percentiles)

    nonzero_quartiles: np.ndarray = np.digitize(nonzero_scores, bins) + 1  # 1, 2, or 3
    quartiles[nonzero_mask] = nonzero_quartiles

    return quartiles


def compute_mae_by_quartile(
    per_molecule_mae: dict[str, torch.Tensor],
    quartiles: np.ndarray,
) -> dict[str, dict[str, list[float]]]:
    """Compute mean MAE per quartile per model per target.

    Args:
        per_molecule_mae: Dict mapping model name -> tensor [N, num_targets].
        quartiles: Array of quartile assignments [N].

    Returns:
        Dict mapping model name -> dict with:
            'combined': list of 4 floats (one per quartile)
            per target name: list of 4 floats (one per quartile)
    """
    results: dict[str, dict[str, list[float]]] = {}

    for model_name, mae_tensor in per_molecule_mae.items():
        model_results: dict[str, list[float]] = {"combined": []}
        for tname in TARGET_NAMES:
            model_results[tname] = []

        for q in range(4):
            mask: np.ndarray = quartiles == q
            if mask.sum() == 0:
                model_results["combined"].append(float("nan"))
                for tname in TARGET_NAMES:
                    model_results[tname].append(float("nan"))
                continue

            subset: torch.Tensor = mae_tensor[mask]
            model_results["combined"].append(float(subset.mean()))

            for t_idx, tname in enumerate(TARGET_NAMES):
                model_results[tname].append(float(subset[:, t_idx].mean()))

        results[model_name] = model_results

    return results


def run_curvature_analysis(
    config: Config,
    device: str,
    threshold: float = DEFAULT_CURVATURE_THRESHOLD,
) -> dict:
    """Run the full curvature-based over-squashing analysis.

    Steps:
        1. Evaluate all models per-molecule on the test set.
        2. Compute Ollivier-Ricci curvature for each test molecule.
        3. Assign quartiles by bottleneck score.
        4. Compute MAE per quartile per model.

    Args:
        config: Full experiment config dict.
        device: PyTorch device string.
        threshold: Curvature threshold for bottleneck edges.

    Returns:
        Dict with keys:
            'curvature_stats': per-molecule curvature data
            'quartiles': quartile assignments
            'mae_by_quartile': MAE per quartile per model per target
            'quartile_sizes': number of molecules per quartile
    """
    # Step 1: per-molecule evaluation
    print("\n=== Step 1: Per-molecule evaluation ===")
    per_molecule_mae: dict[str, torch.Tensor] = run_per_molecule_evaluation(
        config, device
    )

    # Step 2: curvature computation
    print("\n=== Step 2: Computing Ollivier-Ricci curvature ===")
    _, _, test_dataset = load_splits(config)
    curvature_stats: dict[str, list] = compute_curvature_stats_for_dataset(
        test_dataset, threshold
    )

    # Step 3: quartile assignment
    print("\n=== Step 3: Assigning quartiles ===")
    quartiles: np.ndarray = assign_quartiles(curvature_stats["bottleneck_scores"])
    quartile_sizes: list[int] = [int((quartiles == q).sum()) for q in range(4)]
    print(f"  Quartile sizes: {quartile_sizes}")

    # Step 4: MAE by quartile
    print("\n=== Step 4: Computing MAE by quartile ===")
    mae_by_quartile: dict[str, dict[str, list[float]]] = compute_mae_by_quartile(
        per_molecule_mae, quartiles
    )

    print_quartile_table(mae_by_quartile, quartile_sizes)

    return {
        "curvature_stats": curvature_stats,
        "quartiles": quartiles.tolist(),
        "mae_by_quartile": mae_by_quartile,
        "quartile_sizes": quartile_sizes,
    }


def print_quartile_table(
    mae_by_quartile: dict[str, dict[str, list[float]]],
    quartile_sizes: list[int],
) -> None:
    """Print formatted tables of MAE by quartile.

    Args:
        mae_by_quartile: MAE per quartile per model per target.
        quartile_sizes: Number of molecules in each quartile.
    """
    quartile_labels: list[str] = [f"Q{q} (n={quartile_sizes[q]})" for q in range(4)]

    # Combined MAE
    print(f"\n{'Combined MAE by Bottleneck Quartile':}")
    header: str = f"{'Quartile':<20}" + "".join(f"{m:>12}" for m in MODEL_NAMES)
    print(header)
    print("-" * len(header))
    for q in range(4):
        row: str = f"{quartile_labels[q]:<20}"
        for model_name in MODEL_NAMES:
            val: float = mae_by_quartile[model_name]["combined"][q]
            row += f"{val:>12.4f}"
        print(row)

    # Per-target tables
    for tname in TARGET_NAMES:
        print(f"\n{tname} MAE by Bottleneck Quartile:")
        print(header)
        print("-" * len(header))
        for q in range(4):
            row = f"{quartile_labels[q]:<20}"
            for model_name in MODEL_NAMES:
                val = mae_by_quartile[model_name][tname][q]
                row += f"{val:>12.4f}"
            print(row)

    print()


def save_curvature_results(results: dict, output_dir: str) -> None:
    """Save curvature analysis results to disk.

    Args:
        results: Full results dict from run_curvature_analysis.
        output_dir: Directory where files will be written.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save everything except numpy arrays (convert to lists)
    serialisable: dict = {
        "curvature_stats": results["curvature_stats"],
        "quartiles": results["quartiles"],
        "mae_by_quartile": results["mae_by_quartile"],
        "quartile_sizes": results["quartile_sizes"],
    }

    filepath: str = os.path.join(output_dir, "curvature_analysis_results.json")
    with open(filepath, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Curvature results saved to {filepath}")


def plot_mae_by_quartile(
    mae_by_quartile: dict[str, dict[str, list[float]]],
    quartile_sizes: list[int],
    output_dir: str,
) -> None:
    """Plot grouped bar charts of MAE by bottleneck quartile.

    Produces one figure for combined MAE and one per target.

    Args:
        mae_by_quartile: MAE per quartile per model per target.
        quartile_sizes: Number of molecules per quartile.
        output_dir: Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    colors: list[str] = ["#4C72B0", "#55A868", "#C44E52"]
    quartile_labels: list[str] = [f"Q{q}\n(n={quartile_sizes[q]})" for q in range(4)]
    x: np.ndarray = np.arange(4)
    width: float = 0.25

    # Combined plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model_name in enumerate(MODEL_NAMES):
        vals: list[float] = mae_by_quartile[model_name]["combined"]
        ax.bar(x + i * width, vals, width, label=model_name, color=colors[i])

    ax.set_xlabel("Bottleneck Quartile (Q0=low, Q3=high)")
    ax.set_ylabel("Mean MAE")
    ax.set_title("Combined MAE by Bottleneck Quartile")
    ax.set_xticks(x + width)
    ax.set_xticklabels(quartile_labels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    filepath: str = os.path.join(output_dir, "mae_by_quartile_combined.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Combined quartile plot saved to {filepath}")

    # Per-target plots
    for tname in TARGET_NAMES:
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, model_name in enumerate(MODEL_NAMES):
            vals = mae_by_quartile[model_name][tname]
            ax.bar(x + i * width, vals, width, label=model_name, color=colors[i])

        ax.set_xlabel("Bottleneck Quartile (Q0=low, Q3=high)")
        ax.set_ylabel("Mean MAE")
        ax.set_title(f"{tname} MAE by Bottleneck Quartile")
        ax.set_xticks(x + width)
        ax.set_xticklabels(quartile_labels)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        filepath = os.path.join(output_dir, f"mae_by_quartile_{tname}.png")
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"{tname} quartile plot saved to {filepath}")


def plot_curvature_distribution(
    curvature_stats: dict[str, list],
    output_dir: str,
) -> None:
    """Plot distribution of bottleneck scores and edge curvatures.

    Args:
        curvature_stats: Per-molecule curvature data.
        output_dir: Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bottleneck score distribution
    ax: plt.Axes = axes[0]
    scores: list[float] = curvature_stats["bottleneck_scores"]
    ax.hist(scores, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Bottleneck Score (fraction of edges with κ < -0.5)")
    ax.set_ylabel("Number of Molecules")
    ax.set_title("Distribution of Bottleneck Scores")
    ax.grid(True, alpha=0.3)

    # Mean curvature distribution
    ax = axes[1]
    means: list[float] = curvature_stats["mean_curvatures"]
    ax.hist(means, bins=50, color="#55A868", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Mean Edge Curvature")
    ax.set_ylabel("Number of Molecules")
    ax.set_title("Distribution of Mean Ollivier-Ricci Curvature")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath: str = os.path.join(output_dir, "curvature_distributions.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Curvature distribution plots saved to {filepath}")


def plot_relative_improvement(
    mae_by_quartile: dict[str, dict[str, list[float]]],
    quartile_sizes: list[int],
    output_dir: str,
) -> None:
    """Plot relative MAE improvement of EGNN over GIN by quartile.

    Shows (MAE_GIN - MAE_EGNN) / MAE_GIN as a percentage.
    The hypothesis predicts this ratio increases with quartile.

    Args:
        mae_by_quartile: MAE per quartile per model per target.
        quartile_sizes: Number of molecules per quartile.
        output_dir: Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    quartile_labels: list[str] = [f"Q{q}\n(n={quartile_sizes[q]})" for q in range(4)]

    targets_to_plot: list[str] = ["combined"] + TARGET_NAMES
    fig, axes = plt.subplots(
        1, len(targets_to_plot), figsize=(4 * len(targets_to_plot), 5)
    )

    for ax_idx, tname in enumerate(targets_to_plot):
        ax: plt.Axes = axes[ax_idx]

        gin_vals: list[float] = mae_by_quartile["GIN"][tname]
        egnn_vals: list[float] = mae_by_quartile["EGNN"][tname]

        improvements: list[float] = []
        for g, e in zip(gin_vals, egnn_vals):
            if g > 0:
                improvements.append((g - e) / g * 100)
            else:
                improvements.append(0.0)

        bars = ax.bar(range(4), improvements, color="#C44E52", alpha=0.8)
        ax.set_xlabel("Bottleneck Quartile")
        ax.set_ylabel("Relative Improvement (%)")
        ax.set_title(f"{tname}")
        ax.set_xticks(range(4))
        ax.set_xticklabels(quartile_labels)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, improvements):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Relative MAE Improvement: EGNN over GIN", fontsize=13)
    fig.tight_layout()
    filepath: str = os.path.join(output_dir, "relative_improvement_by_quartile.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Relative improvement plot saved to {filepath}")
