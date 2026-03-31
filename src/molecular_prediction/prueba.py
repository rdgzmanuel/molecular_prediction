import json

import numpy as np

with open("results/curvature_analysis_results.json") as f:
    data = json.load(f)

scores = data["curvature_stats"]["bottleneck_scores"]
means = data["curvature_stats"]["mean_curvatures"]
mins = data["curvature_stats"]["min_curvatures"]

print(
    f"Bottleneck scores: min={min(scores):.4f}, max={max(scores):.4f}, "
    f"nonzero={sum(1 for s in scores if s > 0)}"
)
print(
    f"Mean curvature: min={min(means):.4f}, max={max(means):.4f}, "
    f"median={np.median(means):.4f}"
)
print(
    f"Min curvature: min={min(mins):.4f}, max={max(mins):.4f}, "
    f"median={np.median(mins):.4f}, p25={np.percentile(mins, 25):.4f}, "
    f"p75={np.percentile(mins, 75):.4f}"
)
