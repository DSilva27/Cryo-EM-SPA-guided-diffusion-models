from pathlib import Path
from typing import List
from typing_extensions import Literal

from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    field_validator,
    FilePath,
    model_validator,
    PositiveFloat,
    PositiveInt,
)


class GuidanceParamsPointCloud(BaseModel, extra="forbid"):
    target_pdbs: FilePath | List[FilePath] = Field(
        description="Path to the target PDB file for guidance."
    )
    guidance_scale: PositiveFloat = Field(
        default=0.5,
        description="Scale factor for the guidance loss.",
    )

    @field_validator("target_pdbs")
    @classmethod
    def validate_target_pdbs(cls, v):
        if isinstance(v, str):
            v = [v]
        for file in v:
            if Path(file).suffix not in [".pdb", ".cif"]:
                raise ValueError("target_pdbs must be PDB or CIF files.")
        return v


class GuidanceParamsImageLikelihoodDataParams(BaseModel, extra="forbid"):
    path_to_starfile: FilePath = Field(description="Path to the Relion starfile.")
    path_to_relion_project: DirectoryPath = Field(
        description="Path to the Relion project directory."
    )
    data_sign: Literal["dark-on-light", "light-on-dark"] = Field(
        default="dark-on-light",
        description="Sign of the data term in the likelihood computation. "
        + "Use 'dark-on-light' for positive log-likelihood and 'light-on-dark' "
        + " for negative log-likelihood.",
    )

    @field_validator("path_to_starfile")
    @classmethod
    def validate_path_to_starfile(cls, v):
        if Path(v).suffix not in [".star"]:
            raise ValueError("path_to_starfile must be a STAR file.")
        return v


class GuidanceParamsImageLikelihood(BaseModel, extra="forbid"):
    reference_pdb: FilePath = Field(description="Path to the reference PDB file.")
    topology_file: FilePath = Field(description="Path to the topology file PDB.")
    data_params: dict = Field(description="Data parameters for the Relion dataset.")
    batch_size: PositiveInt = Field(
        description="Batch size for the dataloader.",
    )
    rng_seed: int = Field(
        default=42,
        description="Random seed for shuffling the dataloader.",
    )
    n_batches: PositiveInt = Field(
        description="Number of batches to use for guidance computation.",
    )
    guidance_scale: PositiveFloat = Field(
        default=0.5,
        description="Scale factor for the guidance loss.",
    )

    @field_validator("reference_pdb")
    @classmethod
    def validate_reference_pdb(cls, v):
        if Path(v).suffix not in [".pdb", ".cif"]:
            raise ValueError("reference_pdb must be PDB or CIF files.")
        return v

    @field_validator("topology_file")
    @classmethod
    def validate_topology_file(cls, v):
        if Path(v).suffix not in [".pdb"]:
            raise ValueError("topology_file must be PDB files.")
        return v


class GuidanceConfig(BaseModel, extra="forbid"):
    guidance_mode: Literal["point-cloud", "cryo-images"] = Field(
        description="Type of guidance to use. Currently only 'point-cloud' is supported."
    )
    guidance_params: dict = Field(description="Parameters for the guidance model.")

    @model_validator(mode="after")
    def validate_guidance_params(self):
        if self.guidance_mode == "point-cloud":
            self.guidance_params = dict(
                GuidanceParamsPointCloud(**self.guidance_params).model_dump()
            )
        elif self.guidance_mode == "cryo-images":
            self.guidance_params = dict(
                GuidanceParamsImageLikelihood(**self.guidance_params).model_dump()
            )
        else:
            raise ValueError(f"Unknown guidance mode: {self.guidance_mode}")
        return self
