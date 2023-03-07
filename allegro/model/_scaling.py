import logging
from typing import List, Optional, Union

import torch
from torch.utils.data import ConcatDataset

from nequip.nn import GraphModuleMixin
from nequip.data import AtomicDataDict
from allegro.nn import PerSpeciesFixedScaleShift


def PerSpeciesRescale(
    model: GraphModuleMixin,
    config,
    dataset: ConcatDataset,
    initialize: bool,
):
    """Add global rescaling for energy(-based quantities).

    If ``initialize`` is false, doesn't compute statistics.
    """
    module_prefix = "per_species_rescale"

    scales = config.get(
        module_prefix + "_scales",
        f"dataset_per_atom_{AtomicDataDict.PER_ATOM_ENERGY_KEY}_std",
    )
    shifts = config.get(
        module_prefix + "_shifts",
        f"dataset_per_atom_{AtomicDataDict.PER_ATOM_ENERGY_KEY}_mean",
    )
    arguments_in_dataset_units = None

    if initialize:
        scales = torch.tensor([1., 10., 1., 1., 1., 1., 1.])
        shifts = torch.tensor([6., 115., 115., 0., 0., 0., 0.])
        arguments_in_dataset_units = True
    else:
        # Put dummy values
        # the real ones will be loaded from the state dict later
        # note that the state dict includes buffers,
        # so this is fine regardless of whether its trainable.
        scales = torch.tensor([1., 10., 1., 1., 1., 1., 1.]) # 1.0 if scales is not None else None #
        shifts = torch.tensor([6., 115., 115., 0., 0., 0., 0.]) # 0.0 if shifts is not None else None #
        # values correctly scaled according to where the come from
        # will be brought from the state dict later,
        # so what you set this to doesnt matter:
        arguments_in_dataset_units = False

    # insert in per species shift
    params = dict(
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        scales=scales,
        shifts=shifts,
    )

    params["arguments_in_dataset_units"] = arguments_in_dataset_units
    model.insert_from_parameters(
        after="per_atom_energy",
        name=module_prefix,
        shared_params=config,
        builder=PerSpeciesFixedScaleShift,
        params=params,
    )

    # == Build the model ==
    return model