from typing import Any, Callable, Optional, TypeVar
from typing_extensions import TypeAlias

from cryojax.dataset import ParticleStackInfo
from jaxtyping import Array, Float


PerParticleT = TypeVar("PerParticleT")
ConstantT = TypeVar("ConstantT")

LossFn: TypeAlias = Callable[
    [
        Float[Array, "n_atoms 3"],
        ParticleStackInfo,
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Optional[Any],
        Optional[bool],
        ConstantT,
        PerParticleT,
    ],
    Float,
]
