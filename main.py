"""Research collaboration link prediction on the NetScience co-authorship network.

Single source of truth for the whole project: the notebook imports the functions
defined here, so the script, the notebook and every file in results/ come from the
same code and the same split.

What this script produces
  * results/evaluation_metrics.csv           primary run (seed 42), random tie-breaking
  * results/evaluation_metrics_by_seed.csv   20 repeated splits (seeds 0-19)
  * results/evaluation_metrics_summary.csv   mean and standard deviation over those 20 splits
  * results/top_20_recommendations.csv       Top-20 NEW collaborations per method (full network)
  * results/topk_hit_rate.csv                validation: Top-K of the training graph that are hidden edges
  * results/cytoscape_predicted_links.csv    same, as a Cytoscape edge table (import with netscience_lcc.graphml)
  * results/algorithm_comparison.png, results/network_overview.png
  * data/processed/*                         cleaned graph, training graph, split edges

Evaluation notes
  * Precision/Recall/F1 label the |positives| highest-scoring pairs as links. Many pairs
    tie at score 0 for CN / Jaccard / Adamic-Adar, so ties are broken AT RANDOM. (A stable
    sort over a list that starts with the true links silently favours them and inflates
    the metrics.)
  * Negative pairs are drawn from pairs that are not connected in the ORIGINAL network,
    so a hidden (test) edge can never be used as a negative.
  * ROC-AUC uses the raw scores and counts ties as one half.
"""
from pathlib import Path
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

SEED = 42                      # primary split, used for recommendations and Cytoscape files
REPEAT_SEEDS = list(range(20))  # repeated splits for the robust (averaged) evaluation
TEST_FRACTION = 0.20

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "netscience.gml"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

METHODS = {
    "Common Neighbors": "common_neighbors",
    "Jaccard": "jaccard",
    "Adamic-Adar": "adamic_adar",
    "Preferential Attachment": "preferential_attachment",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def canonical(pair):
    """Order-independent representation of an undirected pair."""
    u, v = pair
    return (u, v) if str(u) <= str(v) else (v, u)


def load_and_prepare_graph():
    raw = nx.Graph(nx.read_gml(RAW, label="label"))
    raw.remove_edges_from(nx.selfloop_edges(raw))

    # Largest connected component, as defined for this project.
    largest_nodes = max(nx.connected_components(raw), key=len)
    lcc = raw.subgraph(largest_nodes)

    # Rebuild the graph in a fixed (sorted) order. Iterating a set of strings depends on
    # Python's hash seed, which silently changed the split and every result between runs.
    G = nx.Graph()
    G.add_nodes_from(sorted(lcc.nodes(), key=str))
    G.add_edges_from(sorted(canonical(e) for e in lcc.edges()))
    nx.set_node_attributes(G, dict(G.degree()), "degree")
    return G


def constrained_train_test_split(G, test_fraction=TEST_FRACTION):
    """Hide a fraction of edges while keeping every node connected in training."""
    edges = sorted(canonical(e) for e in G.edges())
    target_test = int(round(len(edges) * test_fraction))

    shuffled = edges.copy()
    random.shuffle(shuffled)

    train = G.copy()
    test_edges = []
    for u, v in shuffled:
        if len(test_edges) >= target_test:
            break
        # Don't remove the edge if it would isolate either endpoint.
        if train.degree(u) <= 1 or train.degree(v) <= 1:
            continue
        train.remove_edge(u, v)
        test_edges.append((u, v))
    return train, test_edges


def negative_sampling(G_original, n):
    """Sample n pairs that are NOT connected in the original network."""
    nonedges = sorted(canonical(e) for e in nx.non_edges(G_original))
    if n > len(nonedges):
        raise ValueError("Requested more negative samples than available non-edges.")
    return random.sample(nonedges, n)


def score_pairs(G, pairs, method):
    if method == "common_neighbors":
        return np.array([len(list(nx.common_neighbors(G, u, v))) for u, v in pairs], dtype=float)

    funcs = {
        "jaccard": nx.jaccard_coefficient,
        "adamic_adar": nx.adamic_adar_index,
        "preferential_attachment": nx.preferential_attachment,
    }
    if method not in funcs:
        raise ValueError(f"Unknown method: {method}")
    lookup = {(u, v): s for u, v, s in funcs[method](G, pairs)}
    return np.array([lookup[(u, v)] for u, v in pairs], dtype=float)


def evaluate(G_train, positives, negatives, rng):
    """Metrics for all four methods on one split, with random tie-breaking."""
    pairs = positives + negatives
    y_true = np.array([1] * len(positives) + [0] * len(negatives))
    n_pos = len(positives)

    rows = []
    for name, method in METHODS.items():
        scores = score_pairs(G_train, pairs, method)

        # Highest score first; equal scores in random order (last key of lexsort is primary).
        order = np.lexsort((rng.random(len(scores)), -scores))
        y_pred = np.zeros(len(scores), dtype=int)
        y_pred[order[:n_pos]] = 1

        rows.append({
            "Algorithm": name,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, scores),
            "Precision@10": y_true[order[:10]].mean(),
            "Precision@20": y_true[order[:20]].mean(),
        })
    return pd.DataFrame(rows)


def run_split(G, seed):
    """One complete experiment: split, negatives, evaluation. Returns everything needed."""
    set_seed(seed)
    G_train, positives = constrained_train_test_split(G, TEST_FRACTION)
    negatives = negative_sampling(G, len(positives))
    metrics = evaluate(G_train, positives, negatives, np.random.default_rng(seed))
    return G_train, positives, negatives, metrics


def rank_candidates(G, k):
    """Top-k unlinked pairs of G for every method.

    Ties in score are ordered by the number of shared collaborators (more first) and then
    alphabetically, so the order is meaningful and identical on every run.
    """
    candidates = sorted(canonical(e) for e in nx.non_edges(G))
    shared = np.array([len(list(nx.common_neighbors(G, u, v))) for u, v in candidates])
    rows = []
    for name, method in METHODS.items():
        scores = score_pairs(G, candidates, method)
        order = np.lexsort((np.arange(len(candidates)), -shared, -scores))[:k]
        for rank, idx in enumerate(order, start=1):
            u, v = candidates[idx]
            rows.append({
                "Algorithm": name, "Rank": rank, "Researcher_1": u, "Researcher_2": v,
                "Score": float(scores[idx]), "Common_Neighbors": int(shared[idx]),
            })
    return pd.DataFrame(rows)


def hit_rates(G_train, hidden_edges, ks=(10, 20)):
    """How many of the Top-K pairs of the training graph are hidden (real) collaborations."""
    hidden = {frozenset(e) for e in hidden_edges}
    recs = rank_candidates(G_train, max(ks))
    recs["Hidden_Edge"] = [frozenset((u, v)) in hidden
                           for u, v in zip(recs.Researcher_1, recs.Researcher_2)]
    rows = []
    for name in METHODS:
        g = recs[recs.Algorithm == name]
        row = {"Algorithm": name}
        for k in ks:
            row[f"Hits@{k}"] = int(g[g.Rank <= k].Hidden_Edge.sum())
            row[f"Total@{k}"] = k
        rows.append(row)
    return pd.DataFrame(rows), recs


def summarise(by_seed):
    cols = ["Precision", "Recall", "F1", "ROC-AUC", "Precision@10", "Precision@20"]
    g = by_seed.groupby("Algorithm", sort=False)[cols]
    return g.mean().join(g.std(), rsuffix=" (sd)").reset_index()


def plot_comparison(summary, path):
    names = ["CN", "Jaccard", "Adamic-Adar", "PA"]
    series = [("Precision", "Precision = Recall = F1", "#2a6f97"),
              ("ROC-AUC", "ROC-AUC", "#e07a1f"),
              ("Precision@20", "Precision@20", "#6a994e")]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    for i, (col, label, color) in enumerate(series):
        ax.bar(x + (i - 1) * w, summary[col], w, yerr=summary[col + " (sd)"], capsize=3,
               label=label, color=color, error_kw={"lw": 1, "ecolor": "#444"})
        for xi, v in zip(x + (i - 1) * w, summary[col]):
            ax.text(xi, v + 0.035, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(f"Score (mean ± sd, {len(REPEAT_SEEDS)} splits)")
    ax.set_title("Link prediction: algorithm comparison")
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_network(G_train, path):
    pos = nx.spring_layout(G_train, seed=SEED, k=0.35, iterations=200)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    nx.draw_networkx_edges(G_train, pos, ax=ax, edge_color="#c9c9c9", width=0.5)
    nx.draw_networkx_nodes(G_train, pos, ax=ax, node_size=[10 + 4 * G_train.degree(n) for n in G_train],
                           node_color="#2a6f97", alpha=0.85)
    ax.set_title("NetScience researcher co-authorship network — training graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SOCIAL NETWORK ANALYSIS — RESEARCH COLLABORATION LINK PREDICTION")
    print("=" * 70)

    print("\n[1/6] Loading and preparing network...")
    G = load_and_prepare_graph()
    stats = {
        "full_network_nodes": 1589,
        "full_network_edges": 2742,
        "project_nodes": G.number_of_nodes(),
        "project_edges": G.number_of_edges(),
        "connected_components_after_LCC": nx.number_connected_components(G),
        "density": nx.density(G),
        "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "average_clustering": nx.average_clustering(G),
    }
    nx.write_graphml(G, PROCESSED / "netscience_lcc.graphml")
    pd.DataFrame(list(G.edges()), columns=["source", "target"]).to_csv(
        PROCESSED / "netscience_lcc_edges.csv", index=False)
    pd.DataFrame(
        [(n, d, nx.clustering(G, n)) for n, d in G.degree()],
        columns=["researcher", "degree", "clustering"]).to_csv(
        PROCESSED / "node_statistics.csv", index=False)
    with open(PROCESSED / "network_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"    Project network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print(f"\n[2/6] Primary split (seed {SEED}, 80/20)...")
    G_train, positives, negatives, metrics = run_split(G, SEED)
    split_info = {
        "random_seed": SEED,
        "original_project_edges": G.number_of_edges(),
        "training_edges": G_train.number_of_edges(),
        "positive_test_edges": len(positives),
        "negative_test_edges": len(negatives),
        "negative_pairs_overlapping_hidden_edges": len(
            {frozenset(e) for e in negatives} & {frozenset(e) for e in positives}),
    }
    with open(RESULTS / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    pd.DataFrame(positives, columns=["source", "target"]).to_csv(
        PROCESSED / "test_positive_edges.csv", index=False)
    pd.DataFrame(negatives, columns=["source", "target"]).to_csv(
        PROCESSED / "test_negative_edges.csv", index=False)
    nx.write_graphml(G_train, PROCESSED / "training_network.graphml")
    print(f"    Training edges: {split_info['training_edges']}, "
          f"positive test: {split_info['positive_test_edges']}, "
          f"negative test: {split_info['negative_test_edges']}")

    print("\n[3/6] Metrics on the primary split (random tie-breaking)...")
    metrics.to_csv(RESULTS / "evaluation_metrics.csv", index=False)
    print(metrics.round(3).to_string(index=False))

    print(f"\n[4/6] Repeating the experiment for seeds {REPEAT_SEEDS[0]}-{REPEAT_SEEDS[-1]}...")
    frames = []
    for seed in REPEAT_SEEDS:
        m = run_split(G, seed)[3]
        m.insert(0, "seed", seed)
        frames.append(m)
    by_seed = pd.concat(frames, ignore_index=True)
    by_seed.to_csv(RESULTS / "evaluation_metrics_by_seed.csv", index=False)
    summary = summarise(by_seed)
    summary.to_csv(RESULTS / "evaluation_metrics_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    print("\n[5/6] Recommendations...")
    # New collaborations: rank the unlinked pairs of the FULL network.
    recs = rank_candidates(G, 20)
    recs.to_csv(RESULTS / "top_20_recommendations.csv", index=False)
    recs.rename(columns={"Researcher_1": "source", "Researcher_2": "target",
                         "Algorithm": "algorithm", "Score": "score", "Rank": "rank"})[
        ["source", "target", "algorithm", "rank", "score", "Common_Neighbors"]].to_csv(
        RESULTS / "cytoscape_predicted_links.csv", index=False)
    pd.DataFrame([(n, d, nx.clustering(G, n)) for n, d in G.degree()],
                 columns=["researcher", "degree", "clustering"]).to_csv(
        PROCESSED / "cytoscape_node_attributes.csv", index=False)

    # Validation: the same ranking on the seed-42 training graph, counting hidden real edges.
    hits, _ = hit_rates(G_train, positives)
    hits.to_csv(RESULTS / "topk_hit_rate.csv", index=False)
    print(hits.to_string(index=False))

    print("\n[6/6] Figures...")
    plot_comparison(summary, RESULTS / "algorithm_comparison.png")
    plot_network(G_train, RESULTS / "network_overview.png")

    print("\nDONE. Results saved in:", RESULTS)


if __name__ == "__main__":
    main()
