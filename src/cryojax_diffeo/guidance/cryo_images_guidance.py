from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jax_dataloader import DataLoader
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_simplex
from jaxtyping import Array, Float, Int
from optax import Schedule

from ..cryo_em import (
    compute_likelihood_matrix,
    compute_neg_log_likelihood_from_weights,
    LikelihoodFn,
)
from ..utils import rigid_align_positions
from . import AbstractGuidanceModel


class ImageLikelihoodGuidanceModel(AbstractGuidanceModel):
    likelihood_fn: LikelihoodFn
    relion_dataloader: DataLoader
    reference_positions: Float[Array, "n_atoms 3"]
    n_batches: int
    guidance_schedule: Schedule

    def __init__(
        self,
        likelihood_fn: LikelihoodFn,
        relion_dataloader: DataLoader,
        reference_positions: Float[Array, "n_atoms 3"],
        n_batches: int,
        guidance_schedule: Schedule,
    ):
        self.likelihood_fn = likelihood_fn
        self.relion_dataloader = relion_dataloader
        self.reference_positions = reference_positions
        self.n_batches = n_batches
        self.guidance_schedule = guidance_schedule

    def compute_loss_and_gradient(
        self,
        positions: Float[Array, "n_walkers n_atoms 3"],
    ) -> tuple[float, Float[Array, "n_walkers n_atoms 3"]]:
        aligned_positions, rot_mtx1, disp1 = _align_walkers_to_reference(
            positions, self.reference_positions
        )
        weights = jnp.ones((positions.shape[0],)) / positions.shape[0]
        grad = jnp.zeros_like(positions)
        loss = 0.0

        """
        not needed for now as testing with a single model
        will need to be removed when using multiple models
        I think we should optimize weights for a larget batch size
        something like 1000 images at least
        Go through LikelihoodFn and compute_likelihood_matrix
        you will see that I used this filter_bmap thing to process batches
        maybe it is not working as expected, and things are running in a
        single batch.

        I did this because I was computing weights per-batch, but we might
        be ok by simply computing the weights using the filter_bmap, and
        then simply adding out the gradients (so we would do a filter_bmap)
        inside the _compute_loss_and_gradient function.
        """

        # weights = _optimize_weights(
        #     aligned_positions,
        #     weights,
        #     self.likelihood_fn,
        #     batch["images"],
        #     per_particle_args=batch.get("per_particle_args", {}),
        #     n_steps=50,
        #     batch_size_walkers=1,
        #     batch_size_images=10,
        # )
        for i in range(self.n_batches):
            batch = next(iter(self.relion_dataloader))

            updates = _compute_loss_and_gradient(
                aligned_positions,
                weights,
                batch["particle_stack"],
                batch["per_particle_args"],
                self.likelihood_fn,
            )

            loss += updates[0]
            grad += updates[1]

        grad = jnp.einsum("bij, bjk -> bik", grad - disp1, rot_mtx1)

        return loss, grad


@eqx.filter_jit
def _align_walkers_to_reference(walkers, reference_positions):
    return eqx.filter_vmap(rigid_align_positions, in_axes=(0, None))(
        walkers, reference_positions
    )


@eqx.filter_jit
@eqx.filter_value_and_grad
def _compute_loss_and_gradient(
    walkers, weights, relion_stack, per_particle_args, likelihood_fn
):
    return likelihood_fn(
        walkers,
        weights,
        relion_stack,
        per_particle_args,
        batch_size_walkers=1,
        batch_size_images=50,
    )


"""
# example of how the function would look with filter_bmap
@eqx.filter_jit
@eqx.filter_value_and_grad
def _compute_loss_and_gradient(
    walkers, weights, relion_stack, per_particle_args, likelihood_fn
):
    return filter_bmap(
        lambda x: likelihood_fn(
            walkers, weights, x[0], x[1], batch_size_walkers=1, batch_size_images=1
        ),
        (relion_stack, per_particle_args),
        batch_size=10,
    )


"""


@eqx.filter_jit
def _optimize_weights(
    walkers: Float[Array, " n_walkers"],
    weights: Float[Array, " n_walkers"],
    likelihood_fn: LikelihoodFn,
    relion_stack: Float[Array, " n_images image_size image_size"],
    per_particle_args: dict,
    n_steps: Int = 500,
    batch_size_walkers: Optional[int] = None,
    batch_size_images: Optional[int] = None,
) -> Float[Array, " n_walkers"]:
    likelihood_matrix = compute_likelihood_matrix(
        walkers,
        relion_stack,
        likelihood_fn.amplitudes,
        likelihood_fn.variances,
        likelihood_fn.image_to_walker_log_likelihood_fn,
        likelihood_fn.dilated_mask,
        constant_args=likelihood_fn.loss_fn_constant_args,
        per_particle_args=per_particle_args,
        batch_size_walkers=batch_size_walkers,
        batch_size_images=batch_size_images,
    )
    pg = ProjectedGradient(
        fun=compute_neg_log_likelihood_from_weights,
        projection=projection_simplex,
        maxiter=n_steps,
    )
    return pg.run(weights, likelihood_matrix=likelihood_matrix).params
