# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import torch
import yaml
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Type, TypeVar

from pat.utils.torch import RE_PREFIX, get_index_linspace, instantiate_module

logger = logging.getLogger()

T = TypeVar("T")
CUSTOM_RE_PREFIX = f"{RE_PREFIX}.*"


def set_struct_recursively(cfg, strict: bool = True):
    # Set struct mode for the current level
    OmegaConf.set_struct(cfg, strict)

    # Traverse through nested dictionaries and lists
    if isinstance(cfg, DictConfig):
        for key, value in cfg.items():
            if isinstance(value, (DictConfig, ListConfig)):
                set_struct_recursively(value, strict)
    elif isinstance(cfg, ListConfig):
        for item in cfg:
            if isinstance(item, (DictConfig, ListConfig)):
                set_struct_recursively(item, strict)


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dataclass_from_dict(cls: Type[T], data: dict, strict: bool = True) -> T:
    """
    Converts a dictionary to a dataclass instance, recursively for nested structures.
    """
    base = OmegaConf.structured(cls())
    OmegaConf.set_struct(base, strict)
    override = OmegaConf.create(data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))


def dataclass_to_dict(dataclass_instance: T) -> dict:
    """
    Converts a dataclass instance to a dictionary, recursively for nested structures.
    """
    if isinstance(dataclass_instance, dict):
        return dataclass_instance

    return OmegaConf.to_container(
        OmegaConf.structured(dataclass_instance), resolve=True
    )


def load_config_file(config_file, dataclass_cls: Type[T]) -> T:
    config = OmegaConf.to_container(OmegaConf.load(config_file), resolve=True)
    return dataclass_from_dict(dataclass_cls, config)


def dump_config(config, path, log_config=True):
    yaml_dump = OmegaConf.to_yaml(OmegaConf.structured(config))
    with open(path, "w") as f:
        if log_config:
            logger.info("Using the following config for this run:")
            logger.info(yaml_dump)
        f.write(yaml_dump)


def load_prune_config(
    config_path: str,
    prune_reg_lambda: float,
    num_blocks: int,
    device: torch.device,
    gamma_index_slope: float = 0.0,
    lambda_block_index_slope: float = 1.0,
):
    with open(config_path, "r") as f:
        prune_config = yaml.load(f, Loader=yaml.CLoader)

    for k in list(prune_config.keys()):
        if isinstance(k, tuple):
            prune_config[(instantiate_module(k[0]), k[1])] = prune_config[k]
            del prune_config[k]

    old_to_new_key = {}
    for k, v in prune_config.items():
        if v["group_type"].endswith("SVDGrouper"):
            torch.backends.cuda.preferred_linalg_library("cusolver")

        if gamma_index_slope > 0:
            prune_config[k].setdefault("gamma_index_slope", gamma_index_slope)

        if v.get("per_block_lambda", False):
            assert k.startswith(CUSTOM_RE_PREFIX), (
                f"{k} must start with {CUSTOM_RE_PREFIX}"
            )
            multipliers = get_index_linspace(
                lambda_block_index_slope, num_blocks, device
            ).tolist()
            for i, multiplier in enumerate(multipliers):
                new_key = f"{CUSTOM_RE_PREFIX}\.{i}\.{k[len(CUSTOM_RE_PREFIX) :]}"
                old_to_new_key.setdefault(k, list()).append((new_key, multiplier))

    for old_k, new_tups in old_to_new_key.items():
        for new_k, multiplier in new_tups:
            prune_config[new_k] = prune_config[old_k].copy()
            prune_config[new_k]["reg_lambda"] = prune_reg_lambda * multiplier
        del prune_config[old_k]
    return prune_config
