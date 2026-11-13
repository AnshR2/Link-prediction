# Research Collaboration SNA — Link Prediction

## Problem
Predict potential research collaborations in a co-authorship network.

## Network Model
- Node = researcher
- Edge = two researchers have co-authored a paper
- Graph = undirected, unweighted
- Dataset = NetScience co-authorship network (M. E. J. Newman, 2006)
- Full dataset = 1589 nodes, 2742 edges
- Project network = largest connected component: 379 nodes, 914 edges

## Methods
Common Neighbors, Jaccard Coefficient, Adamic-Adar, Preferential Attachment.

## Evaluation
- 80/20 edge holdout (no researcher is left isolated in the training graph)
- Equal-size negative sampling from pairs never connected in the original network
- Precision, Recall, F1 (the |positives| top-scoring pairs are labelled as links), ROC-AUC, Precision@10, Precision@20
- Score ties are broken at random (a stable sort over a list that starts with the true links inflates the metrics)
- Primary split: seed 42. Robust estimate: mean and standard deviation over 20 splits (seeds 0-19)

## How to run
```
pip install -r requirements.txt
python main.py                 # regenerates every file in data/processed and results
jupyter notebook notebooks/research_collaboration_link_prediction.ipynb   # imports main.py
```
Results are deterministic (graph construction uses a fixed node order, so they do not
depend on Python's hash seed).

## Main files
- `main.py` — the single source of truth (the notebook imports it)
- `data/raw/netscience.gml` — original network
- `data/processed/netscience_lcc.graphml` — cleaned largest connected component
- `data/processed/training_network.graphml` — training graph (731 edges) for Cytoscape
- `data/processed/test_positive_edges.csv`, `test_negative_edges.csv` — the evaluation pairs
- `results/evaluation_metrics.csv` — metrics on the seed-42 split
- `results/evaluation_metrics_summary.csv` — mean and sd over 20 splits (the numbers used in the report)
- `results/evaluation_metrics_by_seed.csv` — every individual split
- `results/top_20_recommendations.csv` — Top-20 NEW collaborations per method (unlinked pairs of the full network; ties ordered by shared collaborators)
- `results/cytoscape_predicted_links.csv` — the same as a Cytoscape edge table (import together with `netscience_lcc.graphml`)
- `results/topk_hit_rate.csv` — validation: how many Top-10 / Top-20 pairs of the training graph are hidden real collaborations
- `results/algorithm_comparison.png`, `results/network_overview.png`
- `results/cytoscape_network.png` — screenshot from Cytoscape (re-export it from `netscience_lcc.graphml` after any re-run)
