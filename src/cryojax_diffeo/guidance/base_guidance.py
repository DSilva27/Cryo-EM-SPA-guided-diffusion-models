import abc
from typing import Tuple

import equinox as eqx
from jaxtyping import Array, Float


class AbstractGuidanceModel(eqx.Module, strict=True):
    @abc.abstractmethod
    def compute_loss_and_gradient(
        self, coordinates: Float[Array, "n_models n_points 3"]
    ) -> Tuple[Float[Array, " n_models"], Float[Array, "n_models n_points 3"]]:
        pass
