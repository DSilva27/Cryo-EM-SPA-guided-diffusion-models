# Cryo-EM-SPA-guided-diffusion-models

## Installation

Installing the required dependencies needs to follow a very specific order, otherwise JAX and Torch will not be compatible. First create and activate your environment. Then install Boltz

```bash
pip install -U boltz[cuda]
```

Then install JAX:

```bash
pip install "jax[cuda12]"
```

Now check if both libraries see your GPU

```bash
python -c "import torch; import jax; print(torch.cuda.is_available()); print(jax.devices('gpu'))"
```

Finally, install this library

```bash
pip install .
```

Or, if you are a developer, install in development mode

```bash
pip install -e ".[dev]"
```

It is still possible that there is something wrong with the installation, as sometimes the numpy version required by JAX clashes with the numpy version required by Boltz. This should be fine, if you get an error message, then simply reinstall with the required numpy version

```bash
pip install numpy==x.xx
```


## Guided prediction with a target PDB


You can run an inference round guided with a PDB using our module with:

```bash
guided_diff predict input_path --use_msa_server
```


## Instructions for the linter

If you are a developer, then after installing the library, install the `pre-commit` hooks:

```bash
pre-commit install
```

This should block you from making commits that don't follow the linter requirements set up for the repo.
