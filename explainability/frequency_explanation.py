"""
Frequency-domain explanation: turns the frequency branch's Grad-CAM
heatmap (explainability/gradcam.py) into a human-readable summary of
WHICH frequency bands (low vs high) the model relied on, rather than just
a raw 2D heatmap.

Paper basis: the review (Section 2.2) explains that low-frequency content
corresponds to coarse image structure while high-frequency content
carries fine texture, edges and generator/decoder artifacts (e.g.
checkerboard patterns from transposed convolutions, attenuated
high-frequency energy). Section 6 calls for frequency explanations
alongside spatial ones. This module produces exactly that: a per-band
importance profile from the centre (low frequency) outward (high
frequency), computed from the model's OWN Grad-CAM attribution (not just
the raw spectrum), so it reflects what the trained model actually used.
"""
from typing import Dict

import numpy as np

from preprocessing.frequency_transform import radial_band_energy


def frequency_band_importance(freq_saliency: np.ndarray, num_bands: int = 8) -> Dict:
    """
    freq_saliency: (H, W) Grad-CAM heatmap over the frequency-branch input,
                   in [0,1] (from explainability/gradcam.frequency_gradcam).
    Returns per-band mean saliency plus a simple low/high summary that is
    easy to show in the UI ("the model relied more on low-frequency /
    high-frequency evidence").
    """
    band_energy = radial_band_energy(freq_saliency, num_bands=num_bands)
    band_energy = band_energy / (band_energy.sum() + 1e-8)  # normalize to a distribution over bands

    half = num_bands // 2
    low_freq_share = float(band_energy[:half].sum())
    high_freq_share = float(band_energy[half:].sum())

    band_labels = [f"band_{i}_{'low' if i < half else 'high'}" for i in range(num_bands)]
    return {
        "band_labels": band_labels,
        "band_importance": band_energy.tolist(),
        "low_freq_share": low_freq_share,
        "high_freq_share": high_freq_share,
        "dominant": "low-frequency" if low_freq_share > high_freq_share else "high-frequency",
    }


def describe_bands_for_ui(result: Dict) -> str:
    """Short natural-language sentence for the Streamlit explanation panel."""
    low_pct = result["low_freq_share"] * 100
    high_pct = result["high_freq_share"] * 100
    return (
        f"The frequency branch attributed {low_pct:.1f}% of its evidence to low-frequency "
        f"(coarse, smooth) content and {high_pct:.1f}% to high-frequency (fine texture / edge / "
        f"periodic artifact) content, dominated by {result['dominant']} evidence."
    )
