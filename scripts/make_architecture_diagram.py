#!/usr/bin/env python3
"""Generates a clean PNG architecture diagram of the pipeline (matplotlib
only, no graphviz dependency) for the README / project report."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from training.config import FIGURES_DIR


def box(ax, xy, w, h, text, fc="#E8EEF7", ec="#2b4a7a", fontsize=9.5, weight="normal"):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4, edgecolor=ec, facecolor=fc, mutation_aspect=1,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, color="#12233f")


def arrow(ax, xy1, xy2, color="#333"):
    ax.annotate("", xy=xy2, xytext=xy1, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(11, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    ax.axis("off")
    ax.set_title("Frequency-Aware Explainable Dual-Branch Deepfake Video Detector", fontsize=13, weight="bold", pad=14)

    box(ax, (3.2, 20.6), 3.6, 0.8, "VIDEO INPUT", fc="#FFF3D6", ec="#8a6d1a", weight="bold")
    box(ax, (3.2, 19.2), 3.6, 0.8, "Frame Extraction\n(configurable N frames)")
    box(ax, (3.2, 17.8), 3.6, 0.8, "Face Detection & Cropping\n(MTCNN, ~10% margin; Haar fallback)")

    box(ax, (0.8, 16.0), 3.6, 0.8, "RGB face crop\n(224x224x3)", fc="#E3F2E1", ec="#2e6b2e")
    box(ax, (5.6, 16.0), 3.6, 0.8, "FFT / DCT magnitude map\n(224x224x1)", fc="#FCE4E4", ec="#8a2b2b")

    box(ax, (0.8, 14.4), 3.6, 1.0, "SPATIAL BRANCH\nMobileNetV3-Small\n(ImageNet-pretrained)", fc="#E3F2E1", ec="#2e6b2e", weight="bold")
    box(ax, (5.6, 14.4), 3.6, 1.0, "FREQUENCY BRANCH\nMesoNet-style CNN\n(trained from scratch)", fc="#FCE4E4", ec="#8a2b2b", weight="bold")

    box(ax, (0.8, 13.0), 3.6, 0.8, "Spatial feature (576-d)", fc="#E3F2E1", ec="#2e6b2e")
    box(ax, (5.6, 13.0), 3.6, 0.8, "Frequency feature (128-d)", fc="#FCE4E4", ec="#8a2b2b")

    box(ax, (2.4, 11.4), 5.2, 1.0, "ADAPTIVE GATED FUSION\n(per-channel learned gate; concat / cross-attention also available)",
        fc="#EDE4F7", ec="#5b3a8a", weight="bold")
    box(ax, (2.4, 10.0), 5.2, 0.8, "Fused feature per frame (256-d)", fc="#EDE4F7", ec="#5b3a8a")

    box(ax, (2.0, 8.4), 6.0, 1.0, "TEMPORAL AGGREGATOR\nBidirectional GRU + temporal-attention pooling", fc="#DDEEF9", ec="#1c5a80", weight="bold")
    box(ax, (2.8, 7.0), 4.4, 0.8, "Video-level feature")
    box(ax, (2.8, 5.6), 4.4, 0.8, "Classifier (FC -> sigmoid)")
    box(ax, (2.4, 4.2), 5.2, 0.8, "REAL / DEEPFAKE + confidence score", fc="#FFF3D6", ec="#8a6d1a", weight="bold")

    box(ax, (0.3, 1.6), 2.9, 1.8, "Spatial Explanation\nGrad-CAM heatmap\non RGB face", fc="#E3F2E1", ec="#2e6b2e", fontsize=8.7)
    box(ax, (3.55, 1.6), 2.9, 1.8, "Frequency Explanation\nFrequency Grad-CAM +\nlow/high band importance", fc="#FCE4E4", ec="#8a2b2b", fontsize=8.7)
    box(ax, (6.8, 1.6), 2.9, 1.8, "Fusion Explanation\nSpatial % vs Frequency %\n(from learned gate)", fc="#EDE4F7", ec="#5b3a8a", fontsize=8.7)

    # main vertical flow
    arrow(ax, (5, 20.6), (5, 20.0))
    arrow(ax, (5, 19.2), (5, 18.6))
    arrow(ax, (5, 17.8), (2.6, 16.8))
    arrow(ax, (5, 17.8), (7.4, 16.8))
    arrow(ax, (2.6, 16.0), (2.6, 15.4))
    arrow(ax, (7.4, 16.0), (7.4, 15.4))
    arrow(ax, (2.6, 14.4), (2.6, 13.8))
    arrow(ax, (7.4, 14.4), (7.4, 13.8))
    arrow(ax, (2.6, 13.0), (4.6, 12.4))
    arrow(ax, (7.4, 13.0), (5.4, 12.4))
    arrow(ax, (5, 11.4), (5, 10.8))
    arrow(ax, (5, 10.0), (5, 9.4))
    arrow(ax, (5, 8.4), (5, 7.8))
    arrow(ax, (5, 7.0), (5, 6.4))
    arrow(ax, (5, 5.6), (5, 5.0))
    arrow(ax, (3.5, 4.2), (1.75, 3.4))
    arrow(ax, (5, 4.2), (5, 3.4))
    arrow(ax, (6.5, 4.2), (8.25, 3.4))

    out_path = os.path.join(FIGURES_DIR, "architecture_diagram.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved architecture diagram -> {out_path}")


if __name__ == "__main__":
    main()
