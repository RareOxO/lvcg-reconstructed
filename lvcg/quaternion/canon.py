"""V5, QC-LVCG: canonicalising the latent VCG's global orientation before the frozen model.

A recording's cardiac vector sits in whatever frame the electrodes happened to define.
V5 predicts that frame -- a single rotation per record -- and undoes it, so the frozen
pretrained model always sees the trajectory in one canonical pose:

    q_pose = PoseNet(V_obs),   V_c = R(q_pose)^T V_obs,   logits = frozen LVCG(V_c)

The plan's hard requirement is identity initialisation: the pose head predicts an
axis-angle vector whose last layer starts at zero, so ``q_pose = [1, 0, 0, 0]`` exactly
and V5 at step 0 is V0, not a model that has already perturbed the input distribution
the backbone was trained on.

Two things make this stage different from V1-V4. First, canonicalisation acts on the
*input* of the frozen model rather than beside its output, so the pretrained beat
encoder, the GRU and the norms are replayed on the canonicalised trajectory -- frozen,
but differentiable, because the pose head learns through them. Second, the rotation
commutes with beat segmentation (R peaks come from the ECG, and the resampling is linear
in the VCG), which is verified in the tests, so the cached beat patches can be rotated
directly instead of re-running segmentation.

**What has to be ruled out.** Any gain here must be separated from plain rotation
augmentation: a model that merely saw rotated copies during training would also become
steadier. ``CanonProbe(canonicalize=False)`` with ``--model augment`` is that control,
trained on exactly the same rotated batches without ever estimating a pose.
"""

import torch
import torch.nn as nn

from .qdf import DynamicEncoder
from .qdt import rollout_hidden
from .utils import normalize_quaternion, quaternion_angle, quaternion_to_rotation_matrix

TINY = 1e-12


def axis_angle_to_quaternion(vector):
    """[..., 3] -> unit quaternion [..., 4]. A zero vector gives exactly the identity."""
    angle = torch.sqrt(vector.square().sum(-1, keepdim=True) + TINY)
    axis = vector / angle
    half = 0.5 * angle
    return torch.cat((torch.cos(half), torch.sin(half) * axis), dim=-1)


def random_rotations(count, degrees=None, generator=None, device=None):
    """``count`` rotation matrices [count, 3, 3] with uniformly random axes.

    ``degrees`` fixes the angle -- what the robustness sweep of the plan needs -- while
    ``None`` draws it uniformly from [0, 180), which is what augmentation uses.
    """
    axis = torch.randn(count, 3, generator=generator)
    axis = axis / torch.sqrt(axis.square().sum(-1, keepdim=True) + TINY)
    if degrees is None:
        angle = torch.rand(count, 1, generator=generator) * torch.pi
    else:
        angle = torch.full((count, 1), float(degrees) * torch.pi / 180.0)
    matrices = quaternion_to_rotation_matrix(axis_angle_to_quaternion(axis * angle))
    return matrices.to(device) if device is not None else matrices


def rotate_beats(beats, rotation):
    """beats [B, N, 3, P] turned by rotation [B, 3, 3] (or [3, 3]).

    The result is made contiguous because the pretrained beat encoder reshapes with
    ``view``, which a rotated (and therefore strided) tensor does not satisfy.
    """
    if rotation.dim() == 2:
        return torch.einsum("ij,bnjp->bnip", rotation, beats).contiguous()
    return torch.einsum("bij,bnjp->bnip", rotation, beats).contiguous()


class PoseNet(nn.Module):
    """One rotation per record, predicted from its own beats and identity at the start.

    The final layer is zero-initialised, so the axis-angle output is exactly zero and the
    pose is exactly the identity quaternion until training moves it. ``max_degrees``
    bounds the correction with a tanh, which keeps an untrained pose head from throwing
    the input far outside the distribution the backbone knows.
    """

    def __init__(self, embedding_dim=64, hidden=32, kernel=7, dropout=0.0, max_degrees=None):
        super().__init__()
        self.encoder = DynamicEncoder(3, embedding_dim, hidden, kernel, dropout)
        self.head = nn.Linear(embedding_dim, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.max_degrees = max_degrees

    def forward(self, beats, beat_mask=None):
        B, N, _, P = beats.shape
        trajectory = beats.permute(0, 2, 1, 3).reshape(B, 3, N * P)
        mask = None
        if beat_mask is not None:
            mask = beat_mask.to(torch.bool).unsqueeze(-1).expand(B, N, P).reshape(B, N * P)
        vector = self.head(self.encoder(trajectory, mask))
        if self.max_degrees:
            limit = float(self.max_degrees) * torch.pi / 180.0
            norm = torch.sqrt(vector.square().sum(-1, keepdim=True) + TINY)
            vector = vector * (limit * torch.tanh(norm / limit) / norm)
        return normalize_quaternion(axis_angle_to_quaternion(vector))


class CanonProbe(nn.Module):
    """V5: pose estimation in front of the frozen pretrained path, plus the linear head.

    The frozen pieces -- beat encoder, StateGRU, norm_struct, norm_dynamic -- are passed
    in and kept at ``requires_grad=False``; gradients still travel through them to the
    pose head, which is the whole point. The GRU keeps its dropout off and follows the
    probe's training mode, because cuDNN refuses to back-propagate through an RNN in
    evaluation mode.
    """

    def __init__(
        self,
        beat_encoder,
        state_generator,
        norm_struct,
        norm_dynamic,
        num_classes=5,
        token_dim=256,
        rhythm_dim=128,
        canonicalize=True,
        pose_dim=64,
        pose_hidden=32,
        max_degrees=None,
        pose_module=None,
    ):
        super().__init__()
        self.beat_encoder = beat_encoder
        self.state_generator = state_generator
        self.norm_struct = norm_struct
        self.norm_dynamic = norm_dynamic
        for module in (beat_encoder, state_generator, norm_struct, norm_dynamic):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for module in self.state_generator.modules():
            if isinstance(module, nn.RNNBase):
                module.dropout = 0.0
            elif isinstance(module, nn.Dropout):
                module.p = 0.0

        self.canonicalize = bool(canonicalize)
        # V6 supplies its own rotation source (a single shared geometry correction, or a
        # per-record one); V5 uses the per-record pose head.
        self.pose = pose_module or (
            PoseNet(pose_dim, pose_hidden, max_degrees=max_degrees) if canonicalize else None)
        self.head = nn.Linear(token_dim + state_generator.hidden_dim + rhythm_dim, num_classes)

    def train(self, mode=True):
        super().train(mode)
        self.beat_encoder.eval()
        self.norm_struct.eval()
        self.norm_dynamic.eval()
        self.state_generator.train(mode)  # cuDNN needs this for the backward pass
        return self

    def embedding(self, beats, rr_intervals, beat_mask, steps, rhythm):
        """The frozen model's embedding of whatever beats it is given, plus the pose."""
        pose = None
        if self.canonicalize:
            pose = self.pose(beats, beat_mask)
            beats = rotate_beats(beats, quaternion_to_rotation_matrix(pose).transpose(-1, -2))
        tokens = self.beat_encoder(beats)
        anchor = tokens[:, 1] if tokens.shape[1] > 1 else tokens[:, 0]
        emb_struct = self.norm_struct(anchor)
        emb_dynamic = self.norm_dynamic(rollout_hidden(self.state_generator, anchor, steps))
        return torch.cat((emb_struct, emb_dynamic, rhythm), dim=-1), pose

    def forward(self, beats, rr_intervals, beat_mask, steps, rhythm):
        embedding, _ = self.embedding(beats, rr_intervals, beat_mask, steps, rhythm)
        return self.head(embedding)

    def pose_angles(self, beats, beat_mask=None):
        """Correction angle per record, in degrees -- zero for a model with no pose head."""
        if not self.canonicalize:
            return torch.zeros(beats.shape[0], device=beats.device)
        angles = quaternion_angle(self.pose(beats, beat_mask)) * 180.0 / torch.pi
        return angles.expand(beats.shape[0]) if angles.numel() == 1 else angles

    def parameter_counts(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        return {
            "trainable_total": sum(p.numel() for p in trainable),
            "pose": sum(p.numel() for p in self.pose.parameters()) if self.pose else 0,
            "head": sum(p.numel() for p in self.head.parameters()),
        }
