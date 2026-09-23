# Frequency-Aware Explainable Deepfake Video Detection (Dual-Branch)

A final-year academic project implementing a **frequency-aware, explainable, dual-branch deepfake video detector**, built as a direct response to the research gaps identified in the review paper:

> *"Frequency-Aware Explainable Deepfake Video Detection Using Dual-Branch Deep Neural Networks: A Review"*

This README explains what comes from the paper, what is this project's own engineering design, and what has actually been tested versus what is provided as instructions for future runs on real data.

---

## 1. Research contribution

> **An explainable, frequency-aware dual-branch deepfake video detection framework that jointly learns spatial (RGB) and frequency-domain (FFT/DCT) evidence through an adaptive gated fusion mechanism, aggregates evidence across frames with a lightweight temporal attention module for genuine video-level prediction, and provides branch-attributable explanations — spatial Grad-CAM, frequency-domain Grad-CAM with low/high-band importance, and learned spatial-vs-frequency contribution percentages — directly addressing the paper's identified gap in branch-level explainability for dual-branch architectures.**

This is a systems/engineering contribution appropriate for a final-year project: it does **not** claim a new state-of-the-art architecture or a research breakthrough. The review paper itself proposes no concrete architecture (it is a literature review); every architectural choice below is this project's own design, explicitly justified against what the paper reports about existing work.

---

## 2. Paper-derived concepts vs. proposed implementation vs. actual results

| | Source |
|---|---|
| Spatial + frequency information are complementary; dual-branch fusion outperforms single-domain detection | **Paper** (Sections 2.3, 7) |
| Xception / MesoNet / F3-Net / FreqNet / D4-Net as reference architectures | **Paper** (Section 2, Section 3) |
| MTCNN + ~10% margin as the standard face-detection convention | **Paper** (Section 3) |
| Naive concatenation underperforms learned/adaptive fusion; frequency-routing sensitivity (routing FFT to the wrong branch costs >15 F1 points in D4-Net's own ablations) | **Paper** (Section 2.3, Section 5) |
| Grad-CAM is the dominant XAI method for deepfake detectors, but branch-attributable explanations for dual-branch models are rare | **Paper** (Section 4, Section 5) |
| Gaps: generalization to unseen generators, frequency-routing sensitivity, compression robustness, lack of branch-level XAI, limited temporal modeling, compute-vs-deployability | **Paper** (Section 5) |
| Future directions: branch-attributable Grad-CAM extension, adaptive frequency routing, lightweight temporal modules (GRU/temporal attention), cross-dataset evaluation | **Paper** (Section 6) |
| MobileNetV3-Small as the spatial backbone (in place of full Xception) | **This project's implementation decision** — feasibility on a CPU-only student machine (see §9) |
| MesoNet-style 4-block CNN built from scratch as the frequency branch | **This project's implementation** — following the architectural pattern the paper describes, not a copy of a specific released codebase |
| Gated adaptive fusion (per-channel learned gate) as the default fusion, with concat and a 2-token cross-attention module also implemented for ablation | **This project's implementation** — a lightweight realization of the "adaptive fusion" concept the paper describes in D4-Net |
| GRU + temporal-attention pooling for video-level aggregation | **This project's implementation of the paper's Section 6 suggestion** |
| Grad-CAM applied separately to both branches + fusion-gate readout as spatial%/frequency% | **This project's implementation of the paper's Section 6 suggestion** ("extending Grad-CAM-style attribution separately to the semantic and frequency branches ... and to the fusion weights themselves") |
| Union-find leakage-safe video-level split | **This project's implementation** — an engineering safeguard, not discussed in the paper |
| All accuracy/precision/recall/F1/AUC numbers currently in `outputs/` | **Actual results, but computed ONLY on a small synthetic placeholder dataset** (see §10) — NOT representative of real-world performance. Real FaceForensics++ / Celeb-DF results are marked **Not evaluated** until you run the pipeline on real data (see §6–§8). |

---

## 3. Proposed architecture

```
VIDEO INPUT
    |
Frame Extraction (configurable N frames, evenly sampled across the clip)
    |
Face Detection / Cropping (MTCNN, ~10% margin; OpenCV Haar-cascade fallback)
    |
    +-------------------------------+
    |                               |
RGB FRAME (224x224x3)      FFT/DCT MAGNITUDE MAP (224x224x1)
    |                               |
SPATIAL BRANCH                 FREQUENCY BRANCH
(MobileNetV3-Small,             (MesoNet-style 4-block
 ImageNet-pretrained,            CNN, trained from scratch)
 early layers frozen)
    |                               |
Spatial feature (576-d)       Frequency feature (128-d)
    +-------------------------------+
                |
   ADAPTIVE GATED FUSION (per-channel learned gate; concat / cross-attention also available)
                |
      Fused feature (256-d) per frame
                |
   TEMPORAL AGGREGATOR (bidirectional GRU + temporal attention pooling)
                |
        Video-level feature
                |
      Classifier (FC -> sigmoid)
                |
        REAL / DEEPFAKE + confidence
                |
   EXPLAINABLE AI: spatial Grad-CAM, frequency Grad-CAM + band importance,
                    fusion-gate spatial%/frequency% readout,
                    per-frame temporal-attention confidence graph
```

Total model size: **~1.6M parameters** (dual-branch, gated fusion) — deliberately small so it trains on a normal laptop CPU in minutes, not hours.

A rendered version of this diagram is at `outputs/figures/architecture_diagram.png` (regenerate anytime with `python3 scripts/make_architecture_diagram.py`).

---

## 4. Project structure

```
deepfake_detection/
├── app/streamlit_app.py            # web demo (upload video -> full explainable pipeline)
├── data/{raw,processed,splits}/    # datasets (raw videos / cached .npz / split CSVs) - gitignored
├── models/
│   ├── spatial_branch.py           # RGB branch (MobileNetV3-Small)
│   ├── frequency_branch.py         # FFT/DCT branch (MesoNet-style CNN)
│   ├── fusion.py                   # gated / concat / cross-attention fusion
│   ├── temporal_model.py           # GRU + temporal attention
│   └── dual_branch_model.py        # assembled model (+ single-branch baselines)
├── preprocessing/
│   ├── face_detector.py            # MTCNN + Haar fallback
│   ├── frequency_transform.py      # FFT / DCT + radial band energy
│   ├── video_processor.py          # per-video pipeline + caching
│   ├── build_manifest.py           # scans FF++ / Celeb-DF folders -> manifest CSV
│   └── dataset_split.py            # leakage-safe (union-find) train/val/test split
├── scripts/
│   ├── preprocess_dataset.py       # batch-runs video_processor.py over a split CSV
│   └── generate_synthetic_dataset.py  # tiny synthetic FF++-layout dataset for pipeline testing
├── training/
│   ├── config.py                   # all hyperparameters, in one place
│   ├── dataset.py                  # PyTorch Dataset/DataLoader over cached .npz files
│   ├── train.py                    # full training loop
│   └── validate.py                 # shared validation loop
├── evaluation/
│   ├── metrics.py                  # accuracy/precision/recall/F1/AUC/confusion/ROC
│   ├── evaluate.py                 # video-level + frame-level test evaluation, plots
│   ├── robustness.py               # JPEG/blur/resize/noise/brightness/re-encode degradation tests
│   └── compare_baselines.py        # Section-12 baseline comparison table
├── explainability/
│   ├── gradcam.py                  # Grad-CAM, applied to both branches
│   ├── frequency_explanation.py    # low/high frequency-band importance
│   └── fusion_explanation.py       # spatial%/frequency% from the learned gate
├── utils/{seed.py, logger.py}
├── outputs/{checkpoints,figures,logs}/   # produced by training/evaluation runs
├── requirements.txt
└── README.md   (this file)
```

---

## 5. Installation

```bash
git clone <this project folder>   # or unzip it
cd deepfake_detection
python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Verify the install:
```bash
python3 -c "import torch, torchvision, cv2, facenet_pytorch, streamlit; print('OK')"
```

**GPU is optional.** `utils/seed.get_device()` auto-detects CUDA and falls back to CPU everywhere — training, evaluation, and the app all run on CPU by default (add `--cpu` to force it even if a GPU is present).

**Note on pretrained weights:** `SpatialBranch` uses `torchvision`'s ImageNet-pretrained MobileNetV3-Small by default, which torchvision downloads on first use over the internet. If your machine has restricted network access, pass `--no_pretrained` to `training/train.py` to skip the download and train the spatial backbone from scratch instead (this is what was used for the synthetic smoke test in this repository, since the sandbox this project was authored in has restricted outbound access — on your own machine with normal internet access, leave pretrained weights ON for meaningfully better spatial-branch features).

---

## 6. Dataset setup (real data)

### Primary: FaceForensics++
1. Request access at https://github.com/ondyari/FaceForensics (an EULA form; access is not instant).
2. Download at minimum the `c23` (HQ) compression level for `original_sequences/youtube` and the four manipulation methods (`DeepFakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`). The full c23 set is ~10 GB; c40 is smaller if disk space is tight — pass `--compression c40` compatible paths to `build_manifest.py` by pointing `--root` at a folder with that layout.
3. Keep the official folder layout:
   ```
   <root>/original_sequences/youtube/c23/videos/*.mp4
   <root>/manipulated_sequences/{DeepFakes,Face2Face,FaceSwap,NeuralTextures}/c23/videos/*.mp4
   ```

### Secondary (for generalization testing): Celeb-DF (V2)
Request access at https://github.com/yuezunli/celeb-deepfakeforensics. Layout:
```
<root>/Celeb-real/*.mp4
<root>/Celeb-synthesis/*.mp4
<root>/YouTube-real/*.mp4
```

### If you cannot download the full datasets
Use a small subset: FF++ lets you download a handful of videos per method for a first pass, or use the bundled synthetic dataset generator to validate the pipeline before committing to a full download (see §10). Everything downstream (manifest → split → preprocessing → training → evaluation → app) is identical regardless of dataset size.

### Building the manifest + leakage-safe split
```bash
python3 preprocessing/build_manifest.py --root /path/to/FaceForensics++ --dataset ffpp --out data/splits/ffpp_manifest.csv
python3 preprocessing/dataset_split.py --manifest data/splits/ffpp_manifest.csv --out_dir data/splits/ffpp \
    --train_frac 0.70 --val_frac 0.15 --test_frac 0.15
```
This performs a **video-level, identity-leakage-safe split**: source identities that co-occur in any manipulated video (e.g. "identity A's face swapped onto identity B's video") are grouped with union-find and always kept entirely on one side of the split, so no face/person ever appears in both train and test. The script prints an explicit leakage check ("no source identity appears in more than one split").

---

## 7. Preprocessing

```bash
python3 scripts/preprocess_dataset.py --split train --manifest data/splits/ffpp/train.csv
python3 scripts/preprocess_dataset.py --split val   --manifest data/splits/ffpp/val.csv
python3 scripts/preprocess_dataset.py --split test  --manifest data/splits/ffpp/test.csv
```
Configurable flags: `--num_frames` (default 8), `--image_size` (default 224), `--freq_type {fft,dct}`, `--face_detector {mtcnn,haar}`, `--limit N` (process only the first N videos — useful for a quick check before committing to the full split). Each video is cached once as a compressed `.npz` under `data/processed/<split>/`, so re-running training never re-decodes video or re-runs face detection.

---

## 8. Training

```bash
# proposed model
python3 training/train.py --train_dir data/processed/ffpp/train --val_dir data/processed/ffpp/val \
    --epochs 15 --batch_size 8 --branch_mode dual --fusion_type gated --run_name dual_gated

# baselines (Section 12 comparison)
python3 training/train.py --train_dir data/processed/ffpp/train --val_dir data/processed/ffpp/val \
    --branch_mode spatial_only  --run_name baseline_rgb
python3 training/train.py --train_dir data/processed/ffpp/train --val_dir data/processed/ffpp/val \
    --branch_mode frequency_only --run_name baseline_freq
python3 training/train.py --train_dir data/processed/ffpp/train --val_dir data/processed/ffpp/val \
    --branch_mode dual --fusion_type concat --run_name dual_concat
```
Configurable: `--epochs`, `--batch_size`, `--lr`, `--optimizer {adam,adamw,sgd}`, `--scheduler {cosine,step,none}`, `--early_stopping_patience`, `--spatial_backbone {mobilenet_v3_small,mobilenet_v3_large,efficientnet_b0}`, `--no_class_weights`, `--seed`. Every run saves `outputs/checkpoints/<run_name>_best.pt` (lowest val loss), `_last.pt` (resumable), and `outputs/logs/<run_name>_history.json` (per-epoch loss/accuracy).

---

## 9. Evaluation, baseline comparison, robustness, generalization

```bash
# video-level + frame-level metrics, confusion matrix, ROC curve, training curves
python3 evaluation/evaluate.py --checkpoint outputs/checkpoints/dual_gated_best.pt \
    --test_dir data/processed/ffpp/test --run_name dual_gated

# Section 12 comparison table (Model | Accuracy | Precision | Recall | F1 | AUC)
python3 evaluation/compare_baselines.py --test_dir data/processed/ffpp/test \
    --checkpoints \
        "RGB-only (baseline)=outputs/checkpoints/baseline_rgb_best.pt" \
        "Frequency-only (baseline)=outputs/checkpoints/baseline_freq_best.pt" \
        "Dual-branch + concat fusion (ablation)=outputs/checkpoints/dual_concat_best.pt" \
        "Dual-branch + gated fusion (proposed)=outputs/checkpoints/dual_gated_best.pt"

# Section 15 robustness table (JPEG / resize / blur / brightness / noise / re-encode)
python3 evaluation/robustness.py --checkpoint outputs/checkpoints/dual_gated_best.pt \
    --test_dir data/processed/ffpp/test --run_name dual_gated
```

**Cross-dataset generalization (Section 14, optional):** train on FF++, then evaluate the same checkpoint on a Celeb-DF (V2) test split built the same way (`build_manifest.py --dataset celebdf` → `dataset_split.py` → `preprocess_dataset.py` → `evaluate.py --test_dir data/processed/celebdf/test`). This experiment is **optional and marked "Not evaluated" in this delivery** because it requires downloading a second full dataset; the pipeline supports it directly once you have the data.

---

## 10. What has actually been run in this delivery (read this before quoting any number)

Downloading FaceForensics++ (~10+ GB, license-gated) or Celeb-DF (V2) was not possible in the sandboxed environment this project was built in. **Every script above was still fully implemented and exercised end-to-end**, using a small synthetic placeholder dataset generated by `scripts/generate_synthetic_dataset.py` (cartoon "face" clips: real clips are plain-shaded, fake clips have an injected faint checkerboard pattern + color-cast, loosely mimicking a decoder artifact, purely so the pipeline has *some* learnable signal to chew on). This is **not real face data** and the metrics computed on it are **not evidence of real-world detection performance** — they only prove the code runs correctly.

Concretely, in `outputs/logs/` and `outputs/figures/` you will find (already generated, from real runs of the code on the synthetic set — not fabricated numbers, just not meaningful ones): per-epoch training curves for all four model variants (`baseline_rgb`, `baseline_freq`, `dual_concat`, `dual_gated`), a video-level and frame-level confusion matrix + ROC curve for `dual_gated`, a robustness table across all six degradations, and a Section-12 baseline comparison table. Re-run any of these commands yourself on the synthetic set (`data/processed/{train,val,test}` are already cached) to reproduce them, or point `--train_dir`/`--test_dir` at your own preprocessed FF++/Celeb-DF data once downloaded — **every command in §6–§9 is unchanged either way.**

*Note: to keep this delivery's download size reasonable, only `baseline_freq_best.pt` and `dual_gated_best.pt` (the checkpoint the Streamlit app defaults to) are bundled in `outputs/checkpoints/`; the `_last.pt` resumable checkpoints and the `baseline_rgb`/`dual_concat` checkpoints were pruned after their metrics/plots were already saved into `outputs/logs/` and `outputs/figures/`. Every one regenerates in well under a minute by re-running the corresponding `training/train.py` command from §8 on `data/processed/{train,val,test}` (the cached synthetic data is included).*

**To reproduce the synthetic smoke test from scratch:**
```bash
python3 scripts/generate_synthetic_dataset.py --num_identities 16 --num_fake_pairs_per_method 6
python3 preprocessing/build_manifest.py --root data/raw/synthetic_ffpp --dataset ffpp --out data/splits/synth_manifest.csv
python3 preprocessing/dataset_split.py --manifest data/splits/synth_manifest.csv --out_dir data/splits/synth
python3 scripts/preprocess_dataset.py --split train --manifest data/splits/synth/train.csv --face_detector haar
python3 scripts/preprocess_dataset.py --split val   --manifest data/splits/synth/val.csv   --face_detector haar
python3 scripts/preprocess_dataset.py --split test  --manifest data/splits/synth/test.csv  --face_detector haar
python3 training/train.py --train_dir data/processed/train --val_dir data/processed/val --branch_mode dual --fusion_type gated --run_name dual_gated --no_pretrained
```
(`--face_detector haar` is used for the synthetic set because the cartoon faces are not real faces MTCNN was trained to find; use the default `mtcnn` for real video data.)

**Real FaceForensics++ / Celeb-DF accuracy, precision, recall, F1, AUC, cross-dataset generalization, and true robustness numbers are marked "Not evaluated" in this delivery** and require you to run §6–§9 on the real datasets on your own machine (a CPU-only laptop can realistically handle a few hundred videos per class for a demonstrable result within a few hours; a GPU makes the full dataset practical).

---

## 11. Running the web app

```bash
streamlit run app/streamlit_app.py
```
Upload a `.mp4`/`.avi`/`.mov` file. If no trained checkpoint exists yet in `outputs/checkpoints/`, the app runs with an untrained model and says so explicitly (useful to demo the UI before training finishes). Once you've trained `dual_gated`, select it from the sidebar. The app shows: video-level prediction + confidence, sample extracted face frames, the RGB→FFT spectrum, spatial-branch Grad-CAM, frequency-branch Grad-CAM + low/high-band importance, the fusion gate's spatial%/frequency% split, and a per-frame confidence graph with the temporal-attention weights overlaid.

---

## 12. Module-by-module explanation (for your viva)

- **`preprocessing/face_detector.py`** — MTCNN (primary, per the paper's Section 3 convention) with an OpenCV Haar-cascade fallback if MTCNN can't be constructed (e.g. restricted environments). Crops with a configurable margin (default 10%, per the paper).
- **`preprocessing/frequency_transform.py`** — 2D FFT magnitude spectrum (shift + log-scale + normalize) and 2D DCT, selectable. `radial_band_energy()` bins the spectrum into concentric low→high frequency rings, used later for explainability.
- **`preprocessing/dataset_split.py`** — the leakage-safe split: builds a graph of source identities linked by shared appearance in manipulated videos, finds connected components (union-find), and assigns whole components to train/val/test so no identity crosses a split boundary.
- **`models/spatial_branch.py` / `frequency_branch.py`** — two architecturally distinct CNNs (never share weights or mix inputs), exactly matching the brief's "do not feed the FFT image into the same RGB branch" requirement and the paper's frequency-routing-sensitivity finding.
- **`models/fusion.py`** — `GatedFusion` (proposed): a small MLP looks at both branches and predicts a per-channel gate in [0,1] deciding how much to trust each branch; averaging the gate gives an interpretable spatial%/frequency% split. `ConcatFusion` (baseline) and `CrossAttentionFusion` (optional, D4-Net-inspired) are also implemented for the ablation comparison.
- **`models/temporal_model.py`** — single-layer bidirectional GRU + learned temporal-attention pooling over frames, directly implementing the paper's Section 6 suggestion for lightweight video-level temporal modeling.
- **`explainability/gradcam.py`** — one generic Grad-CAM implementation, hooked onto each branch's last conv layer independently, producing separate spatial and frequency heatmaps — addressing the paper's Section 5/6 branch-level explainability gap.
- **`explainability/fusion_explanation.py`** — reads the model's own learned gate values (never a hand-picked number) and reports them as a temporal-attention-weighted spatial%/frequency% split.

---

## 13. Hardware notes

- Runs entirely on CPU by default; a CUDA GPU is auto-detected and used automatically if present (`utils/seed.get_device()`).
- Model is ~1.6M parameters (dual-branch) — trains in minutes per epoch on a few hundred short clips on a modern laptop CPU.
- `--image_size`, `--num_frames`, and `--spatial_backbone` are the main knobs to shrink further if needed on a weaker machine.

## 14. VS Code setup

1. Open the `deepfake_detection/` folder in VS Code.
2. Select the `venv` interpreter (Python: Select Interpreter).
3. Install the Python extension if you don't have it, for inline linting/debugging.
4. Run any script from the integrated terminal exactly as shown above (all scripts insert the project root onto `sys.path` themselves, so they work whether run from the project root or via VS Code's "Run Python File").

## 15. Limitations (state these explicitly in your report/viva)

- No real-dataset numbers are included (§10) — this is disclosed, not hidden, per the project's academic-integrity requirement.
- The frequency branch is a from-scratch MesoNet-style CNN, not a literal reproduction of any single paper's released code.
- Cross-attention fusion and cross-dataset generalization are implemented but not the default path, given CPU-only training constraints for a student project.
- The "frame-level" metrics in `evaluate.py` reuse the trained video-level model on length-1 sequences rather than a separately-trained frame classifier (documented in that file's docstring).

## References
See the uploaded review paper's own reference list (MesoNet: Afchar et al. 2018; Xception: Chollet 2017; F3-Net: Qian et al. 2020; D4-Net: Karmakar et al. 2025; FaceForensics++: Rössler et al. 2019; Celeb-DF: Li et al. 2020; and others cited throughout this codebase's docstrings where a specific design choice traces back to a specific reviewed work).
