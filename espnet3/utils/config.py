import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_line(path):
    """
    Load lines from a text file and return as a list of strings.

    This function is used as a custom resolver in OmegaConf,
    allowing YAML files to reference external text line files
    dynamically via `${load_line:some/file.txt}`.

    This resolver is intended to load vocab file in configuration.

    Args:
        path (str or Path): Path to the file.

    Returns:
        List[str]: A list of stripped lines from the file.
    """
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


OMEGACONF_ESPNET3_RESOLVER = {
    "load_line": load_line,
}
for name, resolver in OMEGACONF_ESPNET3_RESOLVER.items():
    OmegaConf.register_new_resolver(name, resolver)
    logging.info(f"Registered ESPnet-3 OmegaConf Resolver: {name}")


def load_config_with_defaults(path: str) -> OmegaConf:
    """
    Load an OmegaConf YAML config file with support for recursive `_self_` merging
    based on Hydra-style `defaults` lists.

    This function recursively loads and merges dependent YAML files specified
    in the `defaults` key of the given config. It mimics Hydra’s composition mechanism
    without using Hydra's runtime, which makes it suitable for standalone YAML handling
    (e.g., for distributed training or script-based training setups).

    Supported formats inside `defaults`:
    - `"subconfig"` → loads `subconfig.yaml`
    - `{"key": "value"}` → loads `key/value.yaml`
    - `"_self_"` → appends the current config in-place

    Example:
        # config.yaml
        defaults:
          - model: conformer
          - optimizer: adam
          - _self_

        # This will recursively load:
        #   model/conformer.yaml
        #   optimizer/adam.yaml
        # and merge them with config.yaml itself at the end.

    Args:
        path (str): Path to the main YAML config file.

    Returns:
        OmegaConf.DictConfig: Fully resolved and merged configuration object.
    """
    base_path = Path(path).parent
    main_cfg = OmegaConf.load(path)
    cfg_self = main_cfg.copy()

    if "defaults" not in main_cfg:
        return cfg_self

    merged_cfgs = []
    self_merged = False

    for entry in main_cfg.defaults:
        if isinstance(entry, str):
            if entry == "_self_":
                merged_cfgs.append(cfg_self)
                self_merged = True
            else:
                cfg_path = _build_config_path(base_path, entry)
                merged_cfgs.append(load_config_with_defaults(str(cfg_path)))

        elif isinstance(entry, DictConfig):
            for key, val in entry.items():
                if val is None:
                    continue
                composed = f"{key}/{val}" if "/" not in val else val
                cfg_path = _build_config_path(base_path, composed)
                merged_cfgs.append({key: load_config_with_defaults(str(cfg_path))})

        elif entry == "_self_":
            merged_cfgs.append(cfg_self)
            self_merged = True

    if not self_merged:
        merged_cfgs.append(cfg_self)

    final_cfg = OmegaConf.merge(*merged_cfgs)

    if "defaults" in final_cfg:
        del final_cfg["defaults"]

    return final_cfg


def _build_config_path(base_path: Path, entry: str) -> Path:
    if not entry.endswith(".yaml"):
        entry += ".yaml"
    return base_path / entry
