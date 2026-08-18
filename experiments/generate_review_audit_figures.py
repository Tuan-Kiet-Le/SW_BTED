"""Generate review-only figures for robustness and cross-domain results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "submission_figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def document_disjoint() -> None:
    methods = ["SW-BTED", "TF-IDF", "pq-Gram", "Section\nCosine", "Standard\nTED"]
    primary = np.array([0.9498, 0.9867, 0.9479, 0.6837, 0.4364])
    disjoint = np.array([0.9157, 0.9870, 0.9474, 0.6387, 0.4318])
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(x - width / 2, primary, width, label="Primary pair-level", color="#2563eb")
    ax.bar(x + width / 2, disjoint, width, label="Document-disjoint", color="#f59e0b")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Pooled F1")
    ax.set_title("Document-disjoint robustness audit")
    ax.set_xticks(x, methods)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    for values, offset in ((primary, -width / 2), (disjoint, width / 2)):
        for i, value in enumerate(values):
            ax.text(i + offset, value + 0.025, f"{value:.3f}", ha="center", fontsize=8)
    save(fig, "figure_6_document_disjoint_robustness")


def cross_domain() -> None:
    methods = ["Structural-only", "Hybrid", "SBERT"]
    out_of_box = np.array([0.5000, 0.9026, np.nan])
    adapted = np.array([0.6725, 0.9141, 0.9074])
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.bar(x - width / 2, out_of_box, width, label="Before domain adaptation", color="#94a3b8")
    ax.bar(x + width / 2, adapted, width, label="After adaptation", color="#16a34a")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1")
    ax.set_title("Cross-domain duplicate bug-report detection")
    ax.set_xticks(x, methods)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    for i, value in enumerate(adapted):
        ax.text(i + width / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=8)
    ax.text(2 - width / 2, 0.04, "n/a", ha="center", fontsize=8, color="#475569")
    ax.text(1.02, -0.16, "The second dataset is a separate transfer experiment; \nthresholds/taxonomy are adapted for the target genre.", transform=ax.transAxes, ha="center", fontsize=8, color="#475569")
    save(fig, "figure_7_cross_domain_evaluation")


if __name__ == "__main__":
    document_disjoint()
    cross_domain()
