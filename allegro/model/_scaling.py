import logging
from typing import List, Optional, Union

import torch
from torch.utils.data import ConcatDataset

from nequip.nn import GraphModuleMixin
from nequip.data import AtomicDataDict
from nequip.model._scaling import _compute_stats
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
    arguments_in_dataset_units = None

    scale_values =  config.get(f"{module_prefix}_scale_values", None) or [1. for _ in config.get("chemical_symbol_to_type")]
    shift_values =  config.get(f"{module_prefix}_shift_values", None) or [0. for _ in config.get("chemical_symbol_to_type")]

    scales = torch.tensor(scale_values)
    shifts = torch.tensor(shift_values)
    
    if initialize:
        arguments_in_dataset_units = True
    else:
        # Put dummy values
        # the real ones will be loaded from the state dict later
        # note that the state dict includes buffers,
        # so this is fine regardless of whether its trainable.
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


def PerSpeciesGlobalRescale(
    model: GraphModuleMixin,
    config,
    dataset: ConcatDataset,
    initialize: bool,
):
    """Add global rescaling for energy(-based quantities).

    If ``initialize`` is false, doesn't compute statistics.
    """
    module_prefix = "per_species_rescale"

    # = Determine energy rescale type =
    scales = config.get(
        module_prefix + "_scales",
        f"dataset_{AtomicDataDict.FORCE_KEY}_rms"
        # if `train_on_keys` isn't provided, assume conservatively
        # that we aren't "training" on anything (i.e. take the
        # most general defaults)
        if AtomicDataDict.FORCE_KEY in config.get("train_on_keys", [])
        else f"dataset_per_atom_{AtomicDataDict.TOTAL_ENERGY_KEY}_std",
    )
    shifts = config.get(
        module_prefix + "_shifts",
        f"dataset_per_atom_{AtomicDataDict.TOTAL_ENERGY_KEY}_mean",
    )

    # Check for common double shift mistake with defaults
    if "RescaleEnergyEtc" in config.get("model_builders", []):
        # if the defaults are enabled, then we will get bad double shift
        # THIS CHECK IS ONLY GOOD ENOUGH FOR EMITTING WARNINGS
        has_global_shift = config.get("global_rescale_shift", None) is not None
        if has_global_shift:
            if shifts is not None:
                # using default of per_atom shift
                raise RuntimeError(
                    "A global_rescale_shift was provided, but the default per-atom energy shift was not disabled."
                )
        del has_global_shift

    # = Determine what statistics need to be compute =\
    arguments_in_dataset_units = None
    if initialize:
        str_names = []
        for value in [scales, shifts]:
            if isinstance(value, str):
                str_names += [value]
            elif (
                value is None
                or isinstance(value, float)
                or isinstance(value, list)
                or isinstance(value, torch.Tensor)
            ):
                # valid values
                pass
            else:
                raise ValueError(f"Invalid value `{value}` of type {type(value)}")

        if len(str_names) == 2:
            # Both computed from dataset
            arguments_in_dataset_units = True
        elif len(str_names) == 1:
            if None in [scales, shifts]:
                # if the one that isnt str is null, it's just disabled
                # that has no units
                # so it's ok to have just one and to be in dataset units
                arguments_in_dataset_units = True
            else:
                assert config[
                    module_prefix + "_arguments_in_dataset_units"
                ], "Requested to set either the shifts or scales of the per_species_rescale using dataset values, but chose to provide the other in non-dataset units. Please give the explictly specified shifts/scales in dataset units and set per_species_rescale_arguments_in_dataset_units"

        # = Compute shifts and scales =
        computed_stats = _compute_stats(
            str_names=str_names,
            dataset=dataset,
            stride=config.dataset_statistics_stride,
            kwargs=config.get(module_prefix + "_kwargs", {}),
        )

        if isinstance(shifts, str):
            s = shifts

            # Init per-dataset shifts tensor
            shifts_per_dataset = [
                torch.zeros(
                    (dataset.datasets[0].type_mapper.num_types,)
                ) * torch.nan for _ in dataset.datasets
            ]

            # Extract shift values from computed_stats
            shifts_updates = [cs[str_names.index(s)].squeeze(-1) for cs in computed_stats]
            
            # Assign scale values to scales_updates
            for s_up, s_dataset in zip(shifts_updates, shifts_per_dataset):
                s_up[s_up == 0.] = torch.nan # zeros are just atom types that do not appear in the _dataset
                s_dataset[:len(s_up)] = s_up # computed_stats is an array of shape (_dataset.fixed_fields['atom_types'].max() + 1)
            
            # Perform mean atom type-wise, ignoring NaN values
            shifts = torch.nanmean(torch.stack(shifts_per_dataset, dim=0), dim=0)

            # Assign default scale value to atom types that do not have statistics
            shifts = torch.nan_to_num(shifts, nan=0.)
            logging.info(f"Replace string {s} to {shifts}")
        elif isinstance(shifts, (list, float)):
            shifts = torch.as_tensor(shifts)

        if isinstance(scales, str):
            s = scales

            # Init per-dataset scales tensor
            scales_per_dataset = [
                torch.zeros(
                    (dataset.datasets[0].type_mapper.num_types,)
                ) * torch.nan for _ in dataset.datasets
            ]

            # Extract scale values from computed_stats
            scales_updates = [cs[str_names.index(s)].squeeze(-1) for cs in computed_stats]
            
            # Assign scale values to scales_updates
            for s_up, s_dataset in zip(scales_updates, scales_per_dataset):
                s_up[s_up == 0.] = torch.nan # zeros are just atom types that do not appear in the _dataset
                s_dataset[:len(s_up)] = s_up # computed_stats is an array of shape (_dataset.fixed_fields['atom_types'].max() + 1)
            
            # Perform mean atom type-wise, ignoring NaN values
            stacked_scales_per_dataset = torch.stack(scales_per_dataset, dim=0)
            scales_mean = torch.nanmean(stacked_scales_per_dataset, dim=0)
            scales = torch.nanmean(torch.pow(stacked_scales_per_dataset - scales_mean[None, ...], 2), dim=0)

            # Assign default scale value to atom types that do not have statistics
            scales[scales == 0.] = torch.nan
            scales = torch.nan_to_num(scales, nan=1.)
            logging.info(f"Replace string {s} to {scales}")
        elif isinstance(scales, (list, float)):
            scales = torch.as_tensor(scales)

    else:
        scale_values =  config.get(f"{module_prefix}_scale_values", None) or [1. for _ in config.get("type_names", config.get("chemical_symbol_to_type"))]
        shift_values =  config.get(f"{module_prefix}_shift_values", None) or [0. for _ in config.get("type_names", config.get("chemical_symbol_to_type"))]    
        scales = torch.tensor(scale_values)
        shifts = torch.tensor(shift_values)

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