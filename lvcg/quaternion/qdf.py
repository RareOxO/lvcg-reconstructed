"""V1, QDF-LVCG: a quaternion dynamic branch beside the frozen pretrained embedding.

    e_base = LVCG(ECG)                        640-d, the validated V0 embedding
    e_Q    = DynamicEncoder(features(VCG))    128-d, trained here
    logits = Linear([e_base ; scale * e_Q])   5 PTB-XL super-classes

The backbone is not touched: ``scripts/train_qdf.py`` runs it once, frozen, and caches
e_base and the latent VCG, so this module only ever sees those two tensors. ``scale``
is what makes the V0-equivalence check of plan section 7.1 exact -- at ``scale = 0`` the
quaternion branch cannot reach the logits, and the model is V0's linear probe on the
same embedding.

The same encoder serves the parameter-matched real-valued control of plan section 6.1:
only the input channels change, everything downstream is identical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import QuaternionDynamicFeatures


class DynamicEncoder(nn.Module):
    """Temporal encoder, identical for the quaternion branch and its control.

    Input batch normalisation puts theta (radians per step) and omega (radians per
    second) on the same footing. It is scale-invariant, so at a fixed dt the theta-only
    and omega-only ablations see the same input up to its epsilon.
    """

    STRIDES = 3  # three stride-2 convolutions

    def __init__(self, in_channels, embedding_dim=128, hidden=64, kernel=7, dropout=0.1):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        layers, width = [nn.BatchNorm1d(in_channels)], in_channels
        for out in (hidden, embedding_dim, embedding_dim):
            layers += [
                nn.Conv1d(width, out, kernel, stride=2, padding=kernel // 2),
                nn.BatchNorm1d(out),
                nn.GELU(),
            ]
            width = out
        self.net = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        # LVCG layer-normalises each embedding part before concatenation; e_Q follows suit.
        self.norm = nn.LayerNorm(embedding_dim)

    def encode(self, x):
        """The convolutional feature map [B, E, T'], before pooling."""
        return self.net(x)

    def pool(self, h, pool_mask=None):
        """Average a feature map over time; with ``pool_mask`` [B, T], over its steps only.

        The mask is carried through the strides with max pooling, which reproduces each
        convolution's output length exactly, so a pooled step counts when any input step
        it summarises was inside the mask. A record whose mask is empty pools to zero.
        V4 pools one feature map several times, once per cardiac phase.
        """
        if pool_mask is None:
            pooled = h.mean(dim=-1)
        else:
            m = pool_mask.unsqueeze(1).to(h.dtype)
            for _ in range(self.STRIDES):
                m = F.max_pool1d(m, kernel_size=2, stride=2, ceil_mode=True)
            pooled = (h * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)
        return self.norm(self.dropout(pooled))

    def forward(self, x, pool_mask=None):
        return self.pool(self.encode(x), pool_mask)


class QDFProbe(nn.Module):
    """The trainable part of V1: quaternion branch + linear head over [e_base ; e_Q].

    Args:
        base_dim: width of the frozen LVCG embedding (640).
        num_classes: 5 PTB-XL super-classes.
        features: channels of plan section 3.2, or the real-valued control set.
        fs: sampling rate of the cached VCG, so dt = 1 / fs and omega is rad/s.
        scale: multiplies e_Q before the head; 0 reproduces V0 exactly.
    """

    def __init__(
        self,
        base_dim=640,
        num_classes=5,
        features=("q", "theta", "omega", "magnitude"),
        fs=100,
        embedding_dim=128,
        hidden=64,
        kernel=7,
        dropout=0.1,
        min_magnitude_fraction=0.02,
        sign_continuity=True,
        masked_pooling=True,
        scale=1.0,
    ):
        super().__init__()
        self.dynamics = QuaternionDynamicFeatures(
            features, 1.0 / float(fs), min_magnitude_fraction, sign_continuity
        )
        self.encoder = DynamicEncoder(self.dynamics.channels, embedding_dim, hidden, kernel, dropout)
        self.head = nn.Linear(base_dim + embedding_dim, num_classes)
        self.masked_pooling = bool(masked_pooling)
        self.embedding_dim = embedding_dim
        self.register_buffer("scale", torch.tensor(float(scale)))

    def quaternion_embedding(self, vcg):
        features, mask = self.dynamics(vcg)
        return self.encoder(features, mask if self.masked_pooling else None)

    def forward(self, base_embedding, vcg):
        """base_embedding [B, 640] (frozen), vcg [B, 3, T] (frozen) -> logits [B, 5]."""
        e_q = self.quaternion_embedding(vcg) * self.scale
        return self.head(torch.cat((base_embedding, e_q), dim=-1))

    def parameter_counts(self):
        head_base = self.head.in_features - self.embedding_dim
        branch = sum(p.numel() for p in self.encoder.parameters())
        branch += self.embedding_dim * self.head.out_features  # the head's share of e_Q
        return {
            "trainable_total": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "quaternion_branch": branch,
            "v0_head": head_base * self.head.out_features + self.head.out_features,
        }
