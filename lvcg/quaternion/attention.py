"""V8, LQA-LVCG: one learnable query per superclass reading the quaternion sequence.

The five PTB-XL superclasses are not diagnosed from the same part of a heart beat --
hypertrophy is a QRS voltage criterion, ST/T changes live in repolarisation. V8 gives
each label its own query over the dynamics sequence and lets it choose where to look:

    h_t     = DynamicEncoder(features(VCG))            the feature map, not yet pooled
    a_{k,t} = softmax_t( q_k . LN(h_t) / sqrt(E) )     one query per label
    e_Q^k   = sum_t a_{k,t} h_t
    logit_k = w_k . [ e_base ; scale * e_Q^k ] + b_k   one small head per label

Three pooling modes make the claim falsifiable. ``label`` is the plan's five queries;
``shared`` is a single query used by every label, which keeps the attention but removes
the label conditioning; ``mean`` is uniform masked averaging, which is exactly what V1
and V3 do. Comparing the three separates "attention helps" from "label-specific
attention helps", and ``mean`` reproduces V3's variant to float rounding (1e-7: the same
arithmetic, summed in a different order).

Because the attention weights are per label and per time step, they can be read: the
script reports how much of each label's attention mass falls inside the QRS and T
windows of V4, which is the interpretability V8 is really for.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import QuaternionDynamicFeatures
from .mrq import VARIANTS
from .qdf import DynamicEncoder

MODES = ("label", "shared", "mean")


class LabelAttentionProbe(nn.Module):
    """V8: label-conditioned attention over the dynamics sequence, beside the frozen embedding.

    Args:
        variant: channels to read, from ``lvcg.quaternion.mrq.VARIANTS``. ``mq`` is the
            plan's quaternion set; ``mo`` is the strongest set V3 found and the more
            informative choice, since V8 asks *where* the labels look, not *what* they read.
        mode: ``label`` (one query per class), ``shared`` (one query for all) or ``mean``
            (uniform averaging, i.e. V3).
        scale: 0 reproduces V0 exactly.
    """

    def __init__(
        self,
        variant="mo",
        mode="label",
        base_dim=640,
        num_classes=5,
        fs=100,
        embedding_dim=128,
        hidden=64,
        kernel=7,
        dropout=0.1,
        min_magnitude_fraction=0.02,
        sign_continuity=True,
        scale=1.0,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {list(VARIANTS)}")
        self.variant, self.mode = variant, mode
        self.num_classes, self.embedding_dim = num_classes, embedding_dim

        self.dynamics = QuaternionDynamicFeatures(
            VARIANTS[variant], 1.0 / float(fs), min_magnitude_fraction, sign_continuity
        )
        self.encoder = DynamicEncoder(self.dynamics.channels, embedding_dim, hidden, kernel, dropout)
        if mode != "mean":
            queries = 1 if mode == "shared" else num_classes
            self.queries = nn.Parameter(torch.randn(queries, embedding_dim) * embedding_dim ** -0.5)
            # The encoder's feature map has a standard deviation of about 0.03, so raw dot
            # products would leave the softmax flat whatever the query. Scoring against a
            # normalised copy makes the temperature meaningful; the pooling itself still
            # uses the untouched feature map, which is what keeps ``mean`` equal to V3.
            self.score_norm = nn.LayerNorm(embedding_dim)
        # One small head per label, each reading e_base and its own attended embedding;
        # the plan asks explicitly for this rather than five large MLPs.
        self.head = nn.Linear(base_dim + embedding_dim, num_classes * 1) if mode == "mean" else None
        self.label_heads = None if mode == "mean" else nn.Parameter(
            torch.zeros(num_classes, base_dim + embedding_dim))
        self.label_bias = None if mode == "mean" else nn.Parameter(torch.zeros(num_classes))
        if self.label_heads is not None:
            nn.init.normal_(self.label_heads, std=(base_dim + embedding_dim) ** -0.5)
        self.register_buffer("scale", torch.tensor(float(scale)))

    def _pooled_mask(self, mask, length):
        """Carry the reliability mask through the encoder's three stride-2 convolutions."""
        m = mask.unsqueeze(1).to(torch.float32)
        for _ in range(DynamicEncoder.STRIDES):
            m = F.max_pool1d(m, kernel_size=2, stride=2, ceil_mode=True)
        return m.squeeze(1)[:, :length] > 0

    def attention(self, vcg):
        """(weights [B, K, T'], feature map h [B, T', E], reliability mask [B, T'])."""
        features, reliable = self.dynamics(vcg)
        h = self.encoder.encode(features).transpose(1, 2)  # [B, T', E]
        mask = self._pooled_mask(reliable, h.shape[1])
        if self.mode == "mean":
            weights = mask.to(h.dtype).unsqueeze(1)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0)
            return weights, h, mask
        scores = torch.einsum("ke,bte->bkt", self.queries, self.score_norm(h)) * self.embedding_dim ** -0.5
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        # A record whose mask is empty would produce NaN; keep it finite and uniform.
        return torch.nan_to_num(weights, nan=0.0), h, mask

    def embeddings(self, vcg):
        """e_Q per label [B, K, E]; in ``mean`` mode the single pooled embedding [B, 1, E]."""
        weights, h, _ = self.attention(vcg)
        pooled = torch.einsum("bkt,bte->bke", weights, h)
        return self.encoder.norm(self.encoder.dropout(pooled))

    def forward(self, base_embedding, vcg):
        pooled = self.embeddings(vcg) * self.scale
        if self.mode == "mean":
            return self.head(torch.cat((base_embedding, pooled[:, 0]), dim=-1))
        if pooled.shape[1] == 1:  # a shared query feeds every label head
            pooled = pooled.expand(-1, self.num_classes, -1)
        joint = torch.cat((base_embedding.unsqueeze(1).expand(-1, self.num_classes, -1), pooled), dim=-1)
        return (joint * self.label_heads).sum(-1) + self.label_bias

    def parameter_counts(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        queries = self.queries.numel() if self.mode != "mean" else 0
        return {
            "trainable_total": sum(p.numel() for p in trainable),
            "branch": sum(p.numel() for p in self.encoder.parameters()),
            "queries": queries,
        }
