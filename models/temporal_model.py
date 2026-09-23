"""
Temporal aggregation over per-frame fused features.

Paper basis: Section 5/6 of the review identifies "video-level temporal
modeling" as an open gap -- "the prevailing single-frame evaluation
protocol ... does not capture temporal artifacts (flicker, inter-frame
inconsistency)" -- and Section 6 (Future Directions) explicitly proposes
"incorporating lightweight temporal modules (e.g., a shallow GRU or
temporal attention layer over per-frame fused descriptors)" as a concrete
next step. This module implements exactly that suggestion.

Proposed implementation decision: a single-layer bidirectional GRU over
the sequence of per-frame fused feature vectors, followed by a temporal
attention pooling step (a learned weighting over frames, so the video
prediction is not just the last GRU hidden state). This keeps the module
lightweight (one GRU layer, small hidden size) as required by the brief
("keep it lightweight ... do not make the sequence unnecessarily long").
"""
import torch
import torch.nn as nn


class TemporalAggregator(nn.Module):
    """Input: (B, T, D) per-frame fused features. Output: (B, D_out) video-level feature
    plus (B, T) per-frame attention weights (used for the frame-wise confidence graph
    in the Streamlit app, and as a lightweight explainability signal)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, bidirectional: bool = True):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attn_score = nn.Linear(gru_out_dim, 1)
        self.output_dim = gru_out_dim

    def forward(self, x: torch.Tensor):
        """x: (B, T, D)"""
        gru_out, _ = self.gru(x)                      # (B, T, gru_out_dim)
        scores = self.attn_score(gru_out).squeeze(-1)   # (B, T)
        weights = torch.softmax(scores, dim=-1)          # (B, T) sums to 1 over frames
        video_feat = torch.sum(gru_out * weights.unsqueeze(-1), dim=1)  # (B, gru_out_dim)
        return video_feat, weights
