from typing import Tuple
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
from optax import Schedule

from ..utils.rmsd_alignment import rigid_align_positions
from .base_guidance import AbstractGuidanceModel


class PointCloudGuidanceModel(AbstractGuidanceModel):
    target_point_clouds: Float[Array, "n_models n_points 3"]
    guidance_schedule: Schedule

    def __init__(
        self,
        reference_point_clouds: Float[Array, "n_models n_points 3"],
        guidance_schedule: Schedule,
    ):
        self.target_point_clouds = reference_point_clouds
        self.guidance_schedule = guidance_schedule

    @override
    def compute_loss_and_gradient(
        self, positions: Float[Array, "n_models n_points 3"]
    ) -> Tuple[Float[Array, " n_models"], Float[Array, "n_models n_points 3"]]:
        return _compute_loss_and_gradient(positions, self.target_point_clouds)


@eqx.filter_vmap
def _compute_loss_and_gradient(
    positions: Float[Array, "n_points 3"],
    target_point_clouds: Float[Array, "n_points 3"],
) -> Tuple[Float, Float[Array, "n_points 3"]]:
    aligned_reference, _, _ = rigid_align_positions(target_point_clouds, positions)

    diff = positions - aligned_reference
    return jnp.sum(diff**2), 2.0 * diff
