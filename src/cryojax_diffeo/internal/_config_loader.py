from pathlib import Path
from typing import overload
from typing_extensions import Literal

import yaml

from ._config_validators import DatasetSimulatorConfig

def load_config(
    path_to_config: str | Path,
) -> DatasetSimulatorConfig:
    """
    Load a configuration file and parse it into the appropriate configuration object.

    Parameters
    ----------
    path_to_config : str | Path
        Path to the configuration file (YAML format).

    Returns
    -------
    DatasetSimulatorConfig
        Parsed configuration object.

    Raises
    ------
    ValueError
        If the provided config_mode is not recognized.
    """
    with open(path_to_config, "r") as f:
        config_dict = yaml.safe_load(f)

    return DatasetSimulatorConfig(**config_dict)