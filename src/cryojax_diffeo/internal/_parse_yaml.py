from pathlib import Path

import jax.numpy as jnp
import mdtraj
import optax
import yaml
from cryojax.dataset import RelionParticleParameterFile, RelionParticleStackDataset

from cryojax_diffeo.cryo_em import LikelihoodFn
from cryojax_diffeo.guidance import (
    AbstractGuidanceModel,
    ImageLikelihoodGuidanceModel,
    PointCloudGuidanceModel,
)
from cryojax_diffeo.io import read_atomic_models

from . import GuidanceConfig


def parse_guidance_yaml(
    path: str | Path,
) -> AbstractGuidanceModel:
    with Path(path).open("r") as file:
        data = yaml.safe_load(file)

    guidance_config = GuidanceConfig(**data)

    return _make_guidance_model(guidance_config)


def _make_guidance_model(
    guidance_config: GuidanceConfig,
) -> AbstractGuidanceModel:
    if guidance_config.guidance_mode == "point-cloud":
        return _make_point_cloud_guidance(guidance_config.guidance_params)
    elif guidance_config.guidance_mode == "cryo-images":
        return _make_cryo_images_guidance(guidance_config.guidance_params)
    else:
        raise ValueError(f"Unknown guidance model type: {guidance_config.guidance_mode}")


def _make_cryo_images_guidance(guidance_params: dict) -> ImageLikelihoodGuidanceModel:
    from cryojax_diffeo.dataset import create_dataloader

    data_sign_factor = (
        -1.0 if guidance_params["data_params"]["data_sign"] == "dark-on-light" else 1.0
    )
    relion_dataset = RelionParticleStackDataset(
        RelionParticleParameterFile(guidance_params["data_params"]["path_to_starfile"]),
        guidance_params["data_params"]["path_to_relion_project"],
    )
    dataloader = create_dataloader(
        relion_dataset,
        batch_size=guidance_params["batch_size"],
        shuffle=True,
        jax_prng_key=guidance_params["rng_seed"],
    )
    amplitudes, variances = _parse_topology(guidance_params["topology_file"])
    reference_positions = _load_reference_positions(guidance_params["reference_pdb"])

    likelihood_fn = LikelihoodFn(
        amplitudes,
        variances,
        image_to_walker_log_likelihood_fn="iso_gaussian_var_marg",
        loss_fn_constant_args=data_sign_factor,
        dilated_mask=None,
        estimates_pose=False,
    )

    # scale_schedule = optax.schedules.cosine_decay_schedule(
    #     init_value=2.0, decay_steps=50, alpha=0.5
    # )

    scale_schedule = optax.constant_schedule(guidance_params["guidance_scale"])

    return ImageLikelihoodGuidanceModel(
        likelihood_fn,
        dataloader,
        reference_positions,
        n_batches=guidance_params["n_batches"],
        guidance_schedule=scale_schedule,
    )


def _load_reference_positions(path_to_pdb: str | Path):
    positions = mdtraj.load(str(path_to_pdb)).center_coordinates().xyz[0] * 10.0
    return jnp.array(positions)


def _parse_topology(path_to_pdb: str | Path):
    atomic_model = read_atomic_models([path_to_pdb])[0]
    return atomic_model["amplitudes"], atomic_model["variances"]


def _make_point_cloud_guidance(guidance_params: dict) -> PointCloudGuidanceModel:
    reference_point_clouds = []
    for file in guidance_params["target_pdbs"]:
        pdb = mdtraj.load(str(file))
        pdb = pdb.atom_slice(pdb.top.select("not element H"))
        reference_point_clouds.append(pdb.xyz[0] * 10.0)

    reference_point_clouds = jnp.array(reference_point_clouds)

    guidance_model = PointCloudGuidanceModel(
        reference_point_clouds=reference_point_clouds,
        guidance_schedule=optax.constant_schedule(guidance_params["guidance_scale"]),
    )

    return guidance_model
