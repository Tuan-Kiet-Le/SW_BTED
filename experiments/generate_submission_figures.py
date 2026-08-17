"""Generate submission-safe figures from canonical SW-BTED artifacts."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "submission_figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                facecolor=color, edgecolor="#333333", linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)


def architecture():
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    levels = [("T1  ROOT", 5.8, "#d9eaf7"), ("T2  DOMAIN", 4.45, "#c9e6d0"),
              ("T3  INTENT", 3.1, "#fde2b8"), ("T4  TERMINOLOGY\n(normalized keyword)", 1.55, "#f5c6d6")]
    for label, y, color in levels: box(ax, 3.2, y, 3.6, .75, label, color)
    for y1, y2 in [(5.8, 5.2), (4.45, 3.85), (3.1, 2.3)]:
        ax.add_patch(FancyArrowPatch((5, y1), (5, y2), arrowstyle="-|>", mutation_scale=15, linewidth=1.4))
    ax.text(1.0, 6.65, "CapTree representation", fontsize=15, weight="bold")
    ax.text(8.35, 4.8, "schema-aware\nlayer weighting", fontsize=10, ha="center")
    ax.text(8.35, 2.15, "leaf-level\nnormalization", fontsize=10, ha="center")
    save(fig, "figure_1_captree_architecture")


def pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    labels = [("Document", "#d9eaf7"), ("Parse +\nnormalize", "#c9e6d0"),
              ("Build\nCapTree", "#fde2b8"), ("APTED\nalignment", "#f5c6d6"),
              ("Structural /\nhybrid score", "#e6d5f5"), ("Thresholded\ndecision", "#d9eaf7")]
    xs = [0.25, 2.2, 4.15, 6.1, 8.05, 10.0]
    for (label, color), x in zip(labels, xs): box(ax, x, 1.35, 1.55, 1.05, label, color)
    for x in xs[:-1]:
        ax.add_patch(FancyArrowPatch((x + 1.55, 1.88), (x + 1.95, 1.88), arrowstyle="-|>", mutation_scale=14, linewidth=1.3))
    ax.text(6, 3.35, "SW-BTED end-to-end comparison pipeline", ha="center", fontsize=15, weight="bold")
    ax.text(8.8, .45, "Hybrid: α·sim_struct + (1−α)·sim_global", ha="center", fontsize=10)
    save(fig, "figure_2_end_to_end_pipeline")


def results():
    data = json.loads((ROOT / "reports" / "audit" / "final_canonical_results_138.json").read_text(encoding="utf-8"))
    names = ["SW-BTED", "TF-IDF", "SBERT", "BGE", "MPNet", "Qwen3", "pq-Gram", "Std TED", "Flat Domain"]
    keys = ["SW-BTED", "TF-IDF", "SBERT_MiniLM", "BGE_Small_v1.5", "MPNet_Base_v2", "Qwen3-Embedding-4B", "pq-Gram", "Standard TED", "Genuine Flat Domain SBERT"]
    vals = [data["methods"][k]["pooled_f1"] for k in keys]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = ["#2c7fb8"] + ["#9ecae1"] * 5 + ["#fdae6b", "#d95f0e", "#d95f0e"]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Pooled F1"); ax.set_title("Canonical 138-pair results")
    ax.tick_params(axis="x", rotation=30)
    for bar, value in zip(bars, vals): ax.text(bar.get_x() + bar.get_width()/2, value + .02, f"{value:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=.25); fig.tight_layout()
    save(fig, "figure_3_canonical_results")


def interpretability_trace():
    data = json.loads((ROOT / "reports" / "interpretability" / "canonical_interpretability_trace_3.json").read_text(encoding="utf-8"))["cases"]
    domains = ["D1", "D2", "D3", "D4"]
    full_domains = ["D1_BUSINESS_CONTEXT", "D2_FUNCTIONAL", "D3_TECHNICAL_REALIZATION", "D4_EXECUTION_PLANNING"]
    x = list(range(len(domains))); width = .24
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, case, color in [(-width, data[0], "#2c7fb8"), (0, data[1], "#fdae6b"), (width, data[2], "#74c476")]:
        values = [next(d["normalized_similarity"] for d in case["trace"]["domains"] if d["domain"] == full) for full in full_domains]
        ax.bar([i + offset for i in x], values, width, label=f"Case {case['case']}", color=color)
    ax.set_xticks(x, domains); ax.set_ylim(0, 1.05); ax.set_ylabel("Per-domain structural similarity")
    ax.set_title("Canonical interpretability traces"); ax.legend(title="Pair")
    ax.grid(axis="y", alpha=.25); fig.tight_layout()
    save(fig, "figure_4_interpretability_traces")


def perturbation_and_runtime():
    perturb = json.loads((ROOT / "reports" / "audit" / "clean_structural_perturbation_metrics_20.json").read_text(encoding="utf-8"))
    runtime = json.loads((ROOT / "reports" / "audit" / "runtime_benchmark_canonical_138.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    methods = ["Full-doc\nSBERT", "SW-BTED\nstructural", "SW-BTED\nhybrid"]
    acc = [perturb["confusion"]["full_doc_sbert"]["accuracy"], perturb["confusion"]["sw_bted_structural"]["accuracy"], perturb["confusion"]["sw_bted_hybrid"]["accuracy"]]
    bars = axes[0].bar(methods, acc, color=["#d95f0e", "#2c7fb8", "#fdae6b"])
    axes[0].set_ylim(0, 1.12); axes[0].set_ylabel("Accuracy"); axes[0].set_title("20-pair structural perturbation")
    for bar, value in zip(bars, acc): axes[0].text(bar.get_x()+bar.get_width()/2, value+.03, f"{value:.0%}", ha="center")
    axes[0].grid(axis="y", alpha=.25)
    bins = runtime["size_bins"]
    labels = ["Small\nquartile", "Middle\nhalf", "Large\nquartile"]
    means = [x["mean_ms"] for x in bins]
    bars = axes[1].bar(labels, means, color=["#9ecae1", "#74c476", "#31a354"])
    axes[1].set_ylabel("Mean milliseconds per pair"); axes[1].set_title("Runtime by tree-size group")
    for bar, value in zip(bars, means): axes[1].text(bar.get_x()+bar.get_width()/2, value+.4, f"{value:.1f}", ha="center")
    axes[1].grid(axis="y", alpha=.25); fig.tight_layout()
    save(fig, "figure_5_perturbation_runtime")


if __name__ == "__main__":
    architecture(); pipeline(); results(); interpretability_trace(); perturbation_and_runtime()
