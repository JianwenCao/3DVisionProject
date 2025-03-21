import yaml
import numpy as np
import rich
import copy
import torch
from matplotlib import cm


_log_styles = {
    "GSBackend": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
    "PGBA": "bold blue",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="GSBackend"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)


def load_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from per scene/dataset cfg.
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # merge per dataset cfg. and main cfg.
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively. dict1 get masked by dict2, and we retuen dict1.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def colorize_np(x, cmap_name='jet', range=None):
    if range is not None:
        vmin, vmax = range
    else:
        vmin, vmax = np.percentile(x, (1, 99))

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]
    return x_new


def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj
