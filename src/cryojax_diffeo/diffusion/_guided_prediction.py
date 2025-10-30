import os
from math import sqrt
from typing import Any, Optional

import boltz.model.layers.initialize as init
import mdtraj
import torch

# Boltz imports
from boltz.data import const
from boltz.model.loss.diffusion import (
    weighted_rigid_align,
)
from boltz.model.models.boltz1 import Boltz1
from boltz.model.modules.confidence import ConfidenceModule
from boltz.model.modules.diffusion import AtomDiffusion
from boltz.model.modules.encoders import RelativePositionEncoder
from boltz.model.modules.trunk import (
    DistogramModule,
    InputEmbedder,
    MSAModule,
    PairformerModule,
)
from boltz.model.modules.utils import (
    compute_random_augmentation,
    default,
)
from torch import nn
from torchmetrics import MeanMetric


def compute_guidance_loss_and_grad(
    coords,
    ref_coords,
    ref_mask,
):
    aligned_ref_coords = weighted_rigid_align(
        ref_coords.float(),
        coords.float(),
        ref_mask,
        ref_mask,
    )

    r = coords - aligned_ref_coords
    r_norm = torch.linalg.norm(r, dim=-1)

    r_hat = r / r_norm.unsqueeze(-1)
    grad = (r_hat * ref_mask.unsqueeze(-1)).unsqueeze(1)

    return r_norm, grad[0]


class GuidedAtomDiffusion(AtomDiffusion):
    def __init__(self, *args, guidance_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.guidance_scale = guidance_scale

    def sample(
        self,
        atom_mask,
        steering_args,
        num_sampling_steps=None,
        multiplicity=1,
        max_parallel_samples=None,
        train_accumulate_token_repr=False,
        **network_condition_kwargs,
    ):
        assert (
            steering_args is not None
        ), "steering_args must be provided for guided sampling."

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)

        # write mask to disk
        # torch.save(atom_mask, "atom_mask.pt")
        reference_coords = torch.zeros((atom_mask.shape[1], 3), device=atom_mask.device)

        n_actual_atoms = steering_args["reference_coords"].shape[0]
        reference_coords[:n_actual_atoms, :] = steering_args["reference_coords"].to(
            self.device
        )

        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        reference_coords = reference_coords[None, ...]
        # print(f"Reference coords shape: {reference_coords.shape}")

        guidance_schedule = steering_args["guidance_schedule"]

        shape = (*atom_mask.shape, 3)
        token_repr_shape = (
            multiplicity,
            network_condition_kwargs["feats"]["token_index"].shape[1],
            2 * self.token_s,
        )

        # get the schedule, which is returned as (sigma, gamma) tuple,
        # and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        # atom position is noise at the beginning
        # print("SHAPE:", shape)
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        atom_coords_denoised = None
        model_cache = {} if self.use_inference_model_cache else None

        token_repr = None
        token_a = None

        rmsd_loss = []
        writer = mdtraj.formats.XTCTrajectoryFile(
            os.path.join(steering_args["out_dir"], "trajectory.xtc"), "w"
        )
        # gradually denoise
        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            random_R, random_tr = compute_random_augmentation(
                multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
            )
            atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
            atom_coords = torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
            if atom_coords_denoised is not None:
                atom_coords_denoised -= atom_coords_denoised.mean(dim=-2, keepdims=True)
                atom_coords_denoised = (
                    torch.einsum("bmd,bds->bms", atom_coords_denoised, random_R)
                    + random_tr
                )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = sqrt(noise_var) * torch.randn(shape, device=self.device)
            atom_coords_noisy = atom_coords + eps

            with torch.no_grad():
                atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
                token_a = torch.zeros(token_repr_shape).to(atom_coords_noisy)

                sample_ids = torch.arange(multiplicity).to(atom_coords_noisy.device)
                sample_ids_chunks = sample_ids.chunk(
                    multiplicity % max_parallel_samples + 1
                )
                for sample_ids_chunk in sample_ids_chunks:
                    atom_coords_denoised_chunk, token_a_chunk = (
                        self.preconditioned_network_forward(
                            atom_coords_noisy[sample_ids_chunk],
                            t_hat,
                            training=False,
                            network_condition_kwargs=dict(
                                multiplicity=sample_ids_chunk.numel(),
                                model_cache=model_cache,
                                **network_condition_kwargs,
                            ),
                        )
                    )
                    atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk
                    token_a[sample_ids_chunk] = token_a_chunk

            if self.accumulate_token_repr:
                if token_repr is None:
                    token_repr = torch.zeros_like(token_a)

                with torch.set_grad_enabled(train_accumulate_token_repr):
                    sigma = torch.full(
                        (atom_coords_denoised.shape[0],),
                        t_hat,
                        device=atom_coords_denoised.device,
                    )
                    token_repr = self.out_token_feat_update(
                        times=self.c_noise(sigma), acc_a=token_repr, next_a=token_a
                    )

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            # print(f"Denoised over sigma shape: {denoised_over_sigma.shape}")
            ############ GUIDANCE STEP ############
            with torch.no_grad():
                loss_guidance, guidance_grad = compute_guidance_loss_and_grad(
                    atom_coords_noisy,
                    reference_coords,
                    atom_mask,
                )
                # print(f"Guidance grad shape: {guidance_grad.shape}")

                unguided_norm = torch.linalg.vector_norm(
                    denoised_over_sigma, dim=(1, 2), keepdim=True
                )
                guided_norm = torch.linalg.vector_norm(
                    guidance_grad, dim=(1, 2), keepdim=True
                )

                guidance_grad *= unguided_norm / guided_norm
                denoised_over_sigma += guidance_schedule * guidance_grad
            #######################################

            atom_coords_next = (
                atom_coords_noisy
                + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            atom_coords = atom_coords_next
            rmsd_loss.append(loss_guidance.mean().item())
            writer.write(atom_coords_next[atom_mask == 1, :].cpu().numpy() / 10.0)

        writer.close()
        rmsd_loss = torch.tensor(rmsd_loss)
        torch.save(rmsd_loss, os.path.join(steering_args["out_dir"], "rmsd_loss.pt"))

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)


class Boltz1Guided(Boltz1):
    def __init__(  # noqa: PLR0915, C901, PLR0912
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        training_args: dict[str, Any],
        validation_args: dict[str, Any],
        embedder_args: dict[str, Any],
        msa_args: dict[str, Any],
        pairformer_args: dict[str, Any],
        score_model_args: dict[str, Any],
        diffusion_process_args: dict[str, Any],
        diffusion_loss_args: dict[str, Any],
        confidence_model_args: dict[str, Any],
        atom_feature_dim: int = 128,
        confidence_prediction: bool = False,
        confidence_imitate_trunk: bool = False,
        alpha_pae: float = 0.0,
        structure_prediction_training: bool = True,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        compile_pairformer: bool = False,
        compile_structure: bool = False,
        compile_confidence: bool = False,
        nucleotide_rmsd_weight: float = 5.0,
        ligand_rmsd_weight: float = 10.0,
        no_msa: bool = False,
        no_atom_encoder: bool = False,
        ema: bool = False,
        ema_decay: float = 0.999,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        predict_args: Optional[dict[str, Any]] = None,
        steering_args: Optional[dict[str, Any]] = None,
        use_kernels: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.lddt = nn.ModuleDict()
        self.disto_lddt = nn.ModuleDict()
        self.complex_lddt = nn.ModuleDict()
        if confidence_prediction:
            self.top1_lddt = nn.ModuleDict()
            self.iplddt_top1_lddt = nn.ModuleDict()
            self.ipde_top1_lddt = nn.ModuleDict()
            self.pde_top1_lddt = nn.ModuleDict()
            self.ptm_top1_lddt = nn.ModuleDict()
            self.iptm_top1_lddt = nn.ModuleDict()
            self.ligand_iptm_top1_lddt = nn.ModuleDict()
            self.protein_iptm_top1_lddt = nn.ModuleDict()
            self.avg_lddt = nn.ModuleDict()
            self.plddt_mae = nn.ModuleDict()
            self.pde_mae = nn.ModuleDict()
            self.pae_mae = nn.ModuleDict()
        for m in const.out_types + ["pocket_ligand_protein"]:
            self.lddt[m] = MeanMetric()
            self.disto_lddt[m] = MeanMetric()
            self.complex_lddt[m] = MeanMetric()
            if confidence_prediction:
                self.top1_lddt[m] = MeanMetric()
                self.iplddt_top1_lddt[m] = MeanMetric()
                self.ipde_top1_lddt[m] = MeanMetric()
                self.pde_top1_lddt[m] = MeanMetric()
                self.ptm_top1_lddt[m] = MeanMetric()
                self.iptm_top1_lddt[m] = MeanMetric()
                self.ligand_iptm_top1_lddt[m] = MeanMetric()
                self.protein_iptm_top1_lddt[m] = MeanMetric()
                self.avg_lddt[m] = MeanMetric()
                self.pde_mae[m] = MeanMetric()
                self.pae_mae[m] = MeanMetric()
        for m in const.out_single_types:
            if confidence_prediction:
                self.plddt_mae[m] = MeanMetric()
        self.rmsd = MeanMetric()
        self.best_rmsd = MeanMetric()

        self.train_confidence_loss_logger = MeanMetric()
        self.train_confidence_loss_dict_logger = nn.ModuleDict()
        for m in [
            "plddt_loss",
            "resolved_loss",
            "pde_loss",
            "pae_loss",
        ]:
            self.train_confidence_loss_dict_logger[m] = MeanMetric()

        self.ema = None
        self.use_ema = ema
        self.ema_decay = ema_decay

        self.training_args = training_args
        self.validation_args = validation_args
        self.diffusion_loss_args = diffusion_loss_args
        self.predict_args = predict_args
        self.steering_args = steering_args

        self.use_kernels = use_kernels

        self.nucleotide_rmsd_weight = nucleotide_rmsd_weight
        self.ligand_rmsd_weight = ligand_rmsd_weight

        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.is_pairformer_compiled = False

        # Input projections
        s_input_dim = token_s + 2 * const.num_tokens + 1 + len(const.pocket_contact_info)
        self.s_init = nn.Linear(s_input_dim, token_s, bias=False)
        self.z_init_1 = nn.Linear(s_input_dim, token_z, bias=False)
        self.z_init_2 = nn.Linear(s_input_dim, token_z, bias=False)

        # Input embeddings
        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_feature_dim": atom_feature_dim,
            "no_atom_encoder": no_atom_encoder,
            **embedder_args,
        }
        self.input_embedder = InputEmbedder(**full_embedder_args)
        self.rel_pos = RelativePositionEncoder(token_z)
        self.token_bonds = nn.Linear(1, token_z, bias=False)

        # Normalization layers
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        # Recycling projections
        self.s_recycle = nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = nn.Linear(token_z, token_z, bias=False)
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        # Pairwise stack
        self.no_msa = no_msa
        if not no_msa:
            self.msa_module = MSAModule(
                token_z=token_z,
                s_input_dim=s_input_dim,
                **msa_args,
            )
        self.pairformer_module = PairformerModule(token_s, token_z, **pairformer_args)
        if compile_pairformer:
            # Big models hit the default cache limit (8)
            self.is_pairformer_compiled = True
            torch._dynamo.config.cache_size_limit = 512
            torch._dynamo.config.accumulated_cache_size_limit = 512
            self.pairformer_module = torch.compile(
                self.pairformer_module,
                dynamic=False,
                fullgraph=False,
            )

        # Output modules
        use_accumulate_token_repr = (
            confidence_prediction
            and "use_s_diffusion" in confidence_model_args
            and confidence_model_args["use_s_diffusion"]
        )

        ########################## IMPORTANT CHANGE ##########################
        ## This is the only change with respect to Boltz1
        self.structure_module = GuidedAtomDiffusion(
            score_model_args={
                "token_z": token_z,
                "token_s": token_s,
                "atom_z": atom_z,
                "atom_s": atom_s,
                "atoms_per_window_queries": atoms_per_window_queries,
                "atoms_per_window_keys": atoms_per_window_keys,
                "atom_feature_dim": atom_feature_dim,
                **score_model_args,
            },
            compile_score=compile_structure,
            accumulate_token_repr=use_accumulate_token_repr,
            **diffusion_process_args,
        )
        #######################################################################

        self.distogram_module = DistogramModule(token_z, num_bins)
        self.confidence_prediction = confidence_prediction
        self.alpha_pae = alpha_pae

        self.structure_prediction_training = structure_prediction_training
        self.confidence_imitate_trunk = confidence_imitate_trunk
        if self.confidence_prediction:
            if self.confidence_imitate_trunk:
                self.confidence_module = ConfidenceModule(
                    token_s,
                    token_z,
                    compute_pae=alpha_pae > 0,
                    imitate_trunk=True,
                    pairformer_args=pairformer_args,
                    full_embedder_args=full_embedder_args,
                    msa_args=msa_args,
                    **confidence_model_args,
                )
            else:
                self.confidence_module = ConfidenceModule(
                    token_s,
                    token_z,
                    compute_pae=alpha_pae > 0,
                    **confidence_model_args,
                )
            if compile_confidence:
                self.confidence_module = torch.compile(
                    self.confidence_module, dynamic=False, fullgraph=False
                )

        # Remove grad from weights they are not trained for ddp
        if not structure_prediction_training:
            for name, param in self.named_parameters():
                if name.split(".")[0] != "confidence_module":
                    param.requires_grad = False
