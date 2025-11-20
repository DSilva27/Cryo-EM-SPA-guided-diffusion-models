import jax.numpy as jnp


def rigid_align_positions(target_pos, ref_pos):
    com_ref = jnp.mean(ref_pos, axis=0, keepdims=True)
    com_target = jnp.mean(target_pos, axis=0, keepdims=True)

    cross_cov_matrix = jnp.dot((ref_pos - com_ref).T, target_pos - com_target)
    U, _, Vh = jnp.linalg.svd(cross_cov_matrix)
    det = jnp.linalg.det(U) * jnp.linalg.det(Vh)
    rot_matrix = U @ jnp.diag(jnp.array([1.0, 1.0, det])) @ Vh

    displacement = com_ref - com_target @ rot_matrix.T

    aligned_pos = target_pos @ rot_matrix.T + displacement
    return aligned_pos, rot_matrix, displacement
