"""
Frequency-domain representation of a face crop.

Paper basis: Section 2.2 of the review explains that generative decoders
leave spectral signatures (e.g. checkerboard artifacts from transposed
convolutions, attenuated high-frequency energy) that are often invisible in
the RGB image but visible in the frequency domain, motivating architectures
such as F3-Net (DCT-based) and FreqNet/D4-Net (FFT-magnitude-based).

Proposed implementation decision: We implement BOTH the 2D FFT magnitude
spectrum and the 2D DCT, selectable via `DataConfig.freq_type`, so the
project can reproduce and compare the two frequency conventions used in
the reviewed literature. FFT is the default because it is what D4-Net's
frequency branch (the architecture we take closest inspiration from) uses.
"""
import cv2
import numpy as np


def compute_fft_magnitude(gray: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """2D FFT magnitude spectrum of a grayscale image.

    Steps (as specified in the project brief / standard practice):
      1. image is already grayscale (float32, [0, 255])
      2. apply 2D FFT
      3. shift zero-frequency component to the centre
      4. take magnitude
      5. optional log scaling (compresses the huge dynamic range so the
         result is usable as a CNN input, standard practice for spectral
         visualisation and frequency-branch CNN inputs)
      6. normalise to [0, 1]
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    if log_scale:
        magnitude = np.log1p(magnitude)  # log(1 + x) avoids log(0)
    magnitude = _normalize(magnitude)
    return magnitude


def compute_dct(gray: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """2D Discrete Cosine Transform (used by F3-Net-style frequency branches).

    OpenCV's dct() requires an even-sized, float32 input.
    """
    g = gray.astype(np.float32)
    h, w = g.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    g = g[:h2, :w2]
    coeffs = cv2.dct(g)
    if log_scale:
        coeffs = np.log1p(np.abs(coeffs))
    else:
        coeffs = np.abs(coeffs)
    coeffs = _normalize(coeffs)
    # pad back to original size if we trimmed a row/col for evenness
    if coeffs.shape != gray.shape:
        out = np.zeros(gray.shape, dtype=np.float32)
        out[: coeffs.shape[0], : coeffs.shape[1]] = coeffs
        coeffs = out
    return coeffs


def _normalize(x: np.ndarray) -> np.ndarray:
    x_min, x_max = float(x.min()), float(x.max())
    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min)).astype(np.float32)


def rgb_to_frequency_map(
    face_rgb: np.ndarray, freq_type: str = "fft", size: int = 224, log_scale: bool = True
) -> np.ndarray:
    """Full pipeline: RGB face crop -> single-channel frequency representation.

    Returns a float32 array in [0, 1] of shape (size, size), ready to be
    stacked into a 1-channel (or replicated 3-channel) tensor for the
    frequency branch CNN.
    """
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    if freq_type == "fft":
        return compute_fft_magnitude(gray, log_scale=log_scale)
    elif freq_type == "dct":
        return compute_dct(gray, log_scale=log_scale)
    else:
        raise ValueError(f"Unknown freq_type: {freq_type}. Use 'fft' or 'dct'.")


def radial_band_energy(magnitude: np.ndarray, num_bands: int = 8) -> np.ndarray:
    """Bins the (already-centred) FFT magnitude spectrum into `num_bands`
    concentric radial bands from the centre (low frequency) to the corners
    (high frequency), and returns the mean energy per band.

    This is used by explainability/frequency_explanation.py to give a
    human-readable "low-frequency vs high-frequency" importance summary,
    rather than just a raw 2D heatmap. Low-frequency bands (near the
    centre) correspond to coarse/smooth image content; high-frequency
    bands (near the edges/corners) correspond to fine texture, edges, and
    the periodic decoder/upsampling artifacts the paper discusses.
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_r = r.max()
    band_edges = np.linspace(0, max_r, num_bands + 1)
    energies = np.zeros(num_bands, dtype=np.float32)
    for i in range(num_bands):
        mask = (r >= band_edges[i]) & (r < band_edges[i + 1])
        if mask.sum() > 0:
            energies[i] = magnitude[mask].mean()
    return energies
