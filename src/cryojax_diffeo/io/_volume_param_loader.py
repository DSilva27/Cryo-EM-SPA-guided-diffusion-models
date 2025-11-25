import logging
from pathlib import Path
from typing import List, Tuple

import cryojax.simulator as cxs
from cryojax.io import read_array_from_mrc

from ..io._atomic_model_reader import read_atomic_models


def load_as_volume_parametrization(
    pdb_or_mrc_filenames: List[str],
    *,
    selection_string: str = "all",
    loads_b_factors: bool = False,
) -> Tuple[cxs.GaussianMixtureVolume]:
    """
    Load atomic models from files and convert them to Gaussian mixture volumes.

    TODO: More general atomic model formats!
    **Arguments:**
        atomic_models_filenames: List of filenames containing atomic models.
            The atomic models are expected to be in pdb format.
        selection_string: Selection string for the atomic models in mdtraj format.
        loads_b_factors: If True, loads b factors from the atomic models.
    **Returns:**
        A tuple of Gaussian mixture volumes.
    """

    valid_extensions = (".pdb", ".mrc")
    all_extensions = [Path(filename).suffix for filename in pdb_or_mrc_filenames]
    # check that all extensions are equal
    assert all(
        [ext == all_extensions[0] for ext in all_extensions]
    ), "All files must have the same extension."
    assert all(
        [ext in valid_extensions for ext in all_extensions]
    ), f"Invalid file extension. Supported extensions are: {valid_extensions}"
    volumes = []

    logging.info("Reading atomic models")

    if all_extensions[0] == ".pdb":
        atomic_models_scattering_params = read_atomic_models(
            pdb_or_mrc_filenames,
            selection_string=selection_string,
            loads_b_factors=loads_b_factors,
        )
        for atomic_model in atomic_models_scattering_params.values():
            volume = cxs.GaussianMixtureVolume(
                positions=atomic_model["atom_positions"],
                amplitudes=atomic_model["amplitudes"],
                variances=atomic_model["variances"],
            )
            volumes.append(volume)

    else:
        for filename in pdb_or_mrc_filenames:
            voxel_grid = read_array_from_mrc(filename)
            volumes.append(cxs.FourierVoxelGridVolume.from_real_voxel_grid(voxel_grid))

    volumes = tuple(volumes)
    logging.info("Potentials generated.")
    return volumes
