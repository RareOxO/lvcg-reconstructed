"""V3, MRQ-LVCG (revised after V1): magnitude / absolute orientation / relative rotation.

The revised plan turns V3 from "quaternion instead of Cartesian" into a decomposition
diagnostic. Each latent cardiac vector is split into

    P_t = r_t u_t,   r_t = ||P_t||,   u_t = P_t / ||P_t||,   q_t = Rot(u_t -> u_{t+1})

where r_t says how strong the vector is, u_t where it points in the fixed latent VCG
frame, and q_t how the direction turns between samples. V1 modelled only q_t (plus a
magnitude channel) and lost to the Cartesian reference by 1.33 pp; V3 asks how much of
that gap is the absent absolute orientation.

Architecture is V1's, unchanged on purpose (plan section 6): one feature extractor, one
``DynamicEncoder``, one linear head over ``[e_base ; scale * e_G]``, the same frozen
feature cache. A variant is therefore only a choice of input channels, which keeps the
comparison attributable and the parameter counts within a percent of each other.

    variant  channels                          question
    m        r                                 magnitude alone
    o        u                                 absolute orientation alone
    q        q, theta, omega                    relative rotation alone
    mq       r, q, theta, omega                 V1's QDF, bit-for-bit
    oq       u, q, theta, omega                 does orientation close V1's gap?
    mo       r, u                               the Cartesian vector, factorised
    moq      r, u, q, theta, omega              the full decomposition
    cdf      P_t, P_{t+1}, dP_t                 Cartesian reference (V1's control)
    v0       -                                  the frozen embedding alone

``cdf`` is what earlier reports called the "real-valued control". The new name is the
plan's: quaternion components are stored as real numbers too, so "quaternion vs real"
was never the contrast being measured -- Cartesian versus decomposed is.
"""

from .qdf import QDFProbe

# The plan's feature sets. Channels keep one canonical order everywhere -- rotation,
# then magnitude, then orientation -- which is V1's order, so ``mq`` is V1's QDF
# channel for channel and its locked numbers carry over.
VARIANTS = {
    "m": ("magnitude",),
    "o": ("direction",),
    "q": ("q", "theta", "omega"),
    "mq": ("q", "theta", "omega", "magnitude"),
    "oq": ("q", "theta", "omega", "direction"),
    "mo": ("magnitude", "direction"),
    "moq": ("q", "theta", "omega", "magnitude", "direction"),
    "cdf": ("position", "next_position", "delta"),
    "v0": ("q", "theta", "omega", "magnitude"),  # channels unused: the branch is scaled to 0
}
# V1 named these two differently; its rows stay valid and comparable.
LEGACY_NAMES = {"qdf": "mq", "control": "cdf"}

STAGE_A = ("v0", "o", "oq")           # the orientation diagnostic, run first
STAGE_B = ("m", "mo", "moq")          # the full decomposition, only after Gate A
LOCKED_FROM_V1 = {"v0": 85.28, "mq": 88.00, "cdf": 89.33, "q": 87.49}


def build_probe(variant: str, base_dim=640, num_classes=5, **kwargs) -> QDFProbe:
    """A V1 probe reading the variant's channels; ``v0`` scales the branch to zero.

    ``magnitude`` reaches the encoder through its input BatchNorm, which standardises it
    against the training set -- the stabilising transform the plan asks for, applied
    identically in every set that contains it.
    """
    name = LEGACY_NAMES.get(variant, variant)
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {list(VARIANTS)}")
    kwargs.setdefault("scale", 0.0 if name == "v0" else 1.0)
    return QDFProbe(base_dim=base_dim, num_classes=num_classes, features=VARIANTS[name], **kwargs)
