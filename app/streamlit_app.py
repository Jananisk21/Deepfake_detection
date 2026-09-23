"""
Streamlit demo application.

Upload a video -> the app runs the FULL pipeline (frame sampling, face
detection/cropping, RGB + FFT/DCT tensor construction, the trained
dual-branch model, temporal aggregation) and shows:
    - the video-level REAL/DEEPFAKE prediction + confidence
    - sample extracted face frames
    - the FFT/DCT magnitude spectrum
    - a Grad-CAM heatmap (spatial branch) for a chosen frame
    - a frequency-domain saliency/importance breakdown (low vs high freq)
    - the fusion module's learned spatial-vs-frequency contribution split
    - a frame-by-frame confidence graph

Run with:
    streamlit run app/streamlit_app.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from evaluation.evaluate import load_model
from explainability.frequency_explanation import describe_bands_for_ui, frequency_band_importance
from explainability.fusion_explanation import compute_branch_contribution, describe_contribution_for_ui
from explainability.gradcam import frequency_gradcam, overlay_heatmap, spatial_gradcam
from models.dual_branch_model import DualBranchDeepfakeDetector
from preprocessing.face_detector import build_face_detector
from preprocessing.frequency_transform import compute_fft_magnitude
from preprocessing.video_processor import process_video
from training.config import CFG_DATA, CHECKPOINT_DIR
from training.dataset import IMAGENET_MEAN, IMAGENET_STD
from utils.seed import get_device

st.set_page_config(page_title="Frequency-Aware Explainable Deepfake Detector", layout="wide")


@st.cache_resource
def get_detector(kind: str):
    return build_face_detector(kind)


@st.cache_resource
def get_model(checkpoint_path: str, force_untrained: bool = False):
    device = get_device()
    if force_untrained or checkpoint_path is None:
        model = DualBranchDeepfakeDetector(branch_mode="dual", fusion_type="gated", pretrained_spatial=True).to(device)
        model.eval()
        return model, {"branch_mode": "dual", "fusion_type": "gated"}, device
    model, train_args = load_model(checkpoint_path, device)
    return model, train_args, device


def list_checkpoints():
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    return sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")])


def to_model_tensors(rgb_frames: np.ndarray, freq_frames: np.ndarray, device):
    rgb_norm = (rgb_frames - IMAGENET_MEAN) / IMAGENET_STD
    rgb_t = torch.from_numpy(rgb_norm).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)  # (1,T,3,H,W)
    freq_t = torch.from_numpy(freq_frames).unsqueeze(1).float().unsqueeze(0).to(device)      # (1,T,1,H,W)
    return rgb_t, freq_t


def main():
    st.title("🎬 Frequency-Aware Explainable Deepfake Video Detector")
    st.caption(
        "Dual-branch (RGB + FFT/DCT) deep network with adaptive fusion and Grad-CAM-based "
        "branch-attributable explainability. Built as a final-year project based on the review "
        "paper *\"Frequency-Aware Explainable Deepfake Video Detection Using Dual-Branch Deep "
        "Neural Networks: A Review\"*."
    )

    with st.sidebar:
        st.header("Settings")
        checkpoints = list_checkpoints()
        options = ["(untrained demo model)"] + checkpoints
        chosen = st.selectbox("Model checkpoint", options, index=len(options) - 1 if checkpoints else 0)
        face_detector_kind = st.selectbox("Face detector", ["mtcnn", "haar"], index=0)
        num_frames = st.slider("Frames per video", 4, 16, CFG_DATA.frames_per_video)
        image_size = st.select_slider("Frame resolution", options=[112, 160, 224], value=CFG_DATA.image_size)
        freq_type = st.selectbox("Frequency representation", ["fft", "dct"], index=0)

        if not checkpoints:
            st.warning(
                "No trained checkpoint found in outputs/checkpoints/. "
                "The demo will run with an UNTRAINED model (random weights) so you can see the "
                "full pipeline and UI, but predictions will be meaningless until you run "
                "training/train.py. See the README for training instructions."
            )

    uploaded = st.file_uploader("Upload a video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    if uploaded is None:
        st.info("Upload a video to run the detector.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    checkpoint_path = None if chosen == "(untrained demo model)" else os.path.join(CHECKPOINT_DIR, chosen)
    model, train_args, device = get_model(checkpoint_path, force_untrained=checkpoint_path is None)
    detector = get_detector(face_detector_kind)

    with st.spinner("Extracting frames, detecting faces, computing frequency spectrum..."):
        pv = process_video(
            video_path=video_path,
            video_id="uploaded_video",
            label=-1,
            detector=detector,
            num_frames=num_frames,
            image_size=image_size,
            freq_type=freq_type,
            freq_size=image_size,
            face_margin=CFG_DATA.face_margin,
            log_scale_fft=CFG_DATA.log_scale_fft,
        )

    if pv is None:
        st.error("Could not read this video (unsupported codec or empty file).")
        return

    st.success(f"Extracted {pv.rgb_frames.shape[0]} frames | face detected in {pv.num_faces_detected}/{num_frames} frames"
               f"{' (fell back to centre-crop for the rest)' if pv.num_faces_detected < num_frames else ''}.")

    rgb_t, freq_t = to_model_tensors(pv.rgb_frames, pv.freq_frames, device)

    with st.spinner("Running dual-branch model..."):
        model.eval()
        with torch.no_grad():
            out = model(rgb_t, freq_t)
        video_prob = torch.sigmoid(out["logit"]).item()
    label_str = "DEEPFAKE" if video_prob >= 0.5 else "REAL"
    confidence = video_prob if video_prob >= 0.5 else 1 - video_prob

    st.header(f"Prediction: {'🟥' if label_str=='DEEPFAKE' else '🟩'} {label_str}")
    st.metric("Video-level confidence", f"{confidence*100:.1f}%")
    if checkpoint_path is None:
        st.caption("⚠️ Running with an UNTRAINED model -- this prediction is not meaningful. Train the model first (see README).")

    tabs = st.tabs(["Sample Frames", "Frequency Spectrum", "Grad-CAM (Spatial)", "Frequency Explanation", "Fusion Explanation", "Frame-wise Confidence"])

    with tabs[0]:
        st.subheader("Extracted & cropped face frames")
        cols = st.columns(min(4, pv.rgb_frames.shape[0]))
        for i in range(pv.rgb_frames.shape[0]):
            cols[i % len(cols)].image(pv.rgb_frames[i], caption=f"frame {i}", use_container_width=True)

    with tabs[1]:
        st.subheader("RGB face → FFT magnitude spectrum")
        frame_idx = st.slider("Frame index", 0, pv.rgb_frames.shape[0] - 1, 0, key="freq_frame_idx")
        c1, c2 = st.columns(2)
        c1.image(pv.rgb_frames[frame_idx], caption="Original RGB face", use_container_width=True)
        fig, ax = plt.subplots()
        ax.imshow(pv.freq_frames[frame_idx], cmap="inferno")
        ax.set_title(f"{freq_type.upper()} magnitude spectrum (log-scaled, normalized)")
        ax.axis("off")
        c2.pyplot(fig)
        st.caption(
            "Low frequencies (image centre after FFT-shift) = coarse/smooth content. "
            "High frequencies (towards the edges/corners) = fine texture, edges, and the kind of "
            "periodic upsampling/decoder artifacts the review paper associates with generative synthesis."
        )

    with tabs[2]:
        st.subheader("Spatial-branch Grad-CAM")
        frame_idx = st.slider("Frame index", 0, pv.rgb_frames.shape[0] - 1, 0, key="gradcam_frame_idx")
        rgb_1 = rgb_t[:, frame_idx : frame_idx + 1]
        freq_1 = freq_t[:, frame_idx : frame_idx + 1]
        heatmap, frame_prob = spatial_gradcam(model, rgb_1, freq_1)
        overlay = overlay_heatmap((pv.rgb_frames[frame_idx] * 255).astype(np.uint8), heatmap)
        c1, c2 = st.columns(2)
        c1.image(pv.rgb_frames[frame_idx], caption="Original face", use_container_width=True)
        c2.image(overlay, caption=f"Grad-CAM overlay (single-frame P(fake)={frame_prob:.3f})", use_container_width=True)
        st.caption("Warmer (red/yellow) regions contributed more to the fake/real decision, per the spatial branch's own gradients.")

    with tabs[3]:
        st.subheader("Frequency-branch explanation")
        frame_idx = st.slider("Frame index", 0, pv.rgb_frames.shape[0] - 1, 0, key="freqexp_frame_idx")
        rgb_1 = rgb_t[:, frame_idx : frame_idx + 1]
        freq_1 = freq_t[:, frame_idx : frame_idx + 1]
        freq_heatmap, _ = frequency_gradcam(model, rgb_1, freq_1)
        band_result = frequency_band_importance(freq_heatmap, num_bands=8)
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots()
        ax.imshow(freq_heatmap, cmap="viridis")
        ax.set_title("Frequency-branch Grad-CAM (which spectral regions mattered)")
        ax.axis("off")
        c1.pyplot(fig)
        fig2, ax2 = plt.subplots()
        ax2.bar(range(len(band_result["band_importance"])), band_result["band_importance"])
        ax2.set_xlabel("Radial frequency band (0=low freq/centre -> high=high freq/edge)")
        ax2.set_ylabel("Normalized importance")
        ax2.set_title("Low-vs-high frequency importance")
        c2.pyplot(fig2)
        st.info(describe_bands_for_ui(band_result))

    with tabs[4]:
        st.subheader("Fusion explanation (branch-level attribution)")
        contrib = compute_branch_contribution(out)
        if contrib["available"]:
            c1, c2 = st.columns(2)
            c1.metric("Spatial branch contribution", f"{contrib['video_spatial_pct']}%")
            c2.metric("Frequency branch contribution", f"{contrib['video_freq_pct']}%")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(
                [contrib["video_spatial_pct"], contrib["video_freq_pct"]],
                labels=["Spatial", "Frequency"],
                autopct="%1.1f%%",
                colors=["#4C72B0", "#DD8452"],
            )
            st.pyplot(fig)
            st.caption(describe_contribution_for_ui(contrib))
        else:
            st.warning(describe_contribution_for_ui(contrib))

    with tabs[5]:
        st.subheader("Frame-wise confidence & temporal attention")
        with torch.no_grad():
            frame_probs = []
            for t in range(pv.rgb_frames.shape[0]):
                single_out = model(rgb_t[:, t : t + 1], freq_t[:, t : t + 1])
                frame_probs.append(torch.sigmoid(single_out["logit"]).item())
        frame_weights = out["frame_weights"].detach().cpu().numpy()[0]
        fig, ax = plt.subplots()
        ax.plot(frame_probs, marker="o", label="P(fake) if this frame were judged alone")
        ax2 = ax.twinx()
        ax2.bar(range(len(frame_weights)), frame_weights, alpha=0.3, color="gray", label="Temporal attention weight")
        ax.axhline(0.5, linestyle="--", color="red", linewidth=0.8)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("P(fake)")
        ax2.set_ylabel("Temporal attention")
        ax.set_title("Per-frame confidence vs. how much the video-level GRU attended to each frame")
        st.pyplot(fig)
        st.caption(
            "The video-level prediction is a temporal-attention-weighted aggregation over frames "
            "(see models/temporal_model.py), addressing the paper's identified gap of limited "
            "temporal modeling in single-frame evaluation protocols."
        )

    os.unlink(video_path)


if __name__ == "__main__":
    main()
