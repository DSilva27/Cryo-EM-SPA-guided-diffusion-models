import equinox as eqx
import jax.numpy as jnp
from jax_dataloader import DataLoader
from jaxtyping import Array, Float
from optax import Schedule

from ..cryo_em import LikelihoodOptimalWeightsFn
from ..utils import rigid_align_positions
from . import AbstractGuidanceModel


class ImageLikelihoodGuidanceModel(AbstractGuidanceModel):
    likelihood_fn: LikelihoodOptimalWeightsFn
    relion_dataloader: DataLoader
    reference_positions: Float[Array, "n_atoms 3"]
    n_batches: int
    guidance_schedule: Schedule

    def __init__(
        self,
        likelihood_fn: LikelihoodOptimalWeightsFn,
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

        for i in range(self.n_batches):
            batch = next(iter(self.relion_dataloader))

            updates = _compute_loss_and_gradient(
                aligned_positions, weights, batch, self.likelihood_fn
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
def _compute_loss_and_gradient(walkers, weights, relion_batch, likelihood_fn):
    return likelihood_fn(
        walkers, weights, relion_batch, batch_size_walkers=1, batch_size_images=10
    )[0]
