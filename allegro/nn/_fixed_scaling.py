import logging
from typing import Optional, List, Union

import torch
import torch.nn.functional

from nequip.data import AtomicDataDict
from nequip.data.transforms import TypeMapper
from nequip.nn._graph_mixin import GraphModuleMixin


class PerSpeciesFixedScaleShift(GraphModuleMixin, torch.nn.Module):

    field: str
    out_field: str
    scales_trainble: bool
    shifts_trainable: bool
    has_scales: bool
    has_shifts: bool

    def __init__(
        self,
        field: str,
        num_types: int,
        type_names: List[str],
        shifts: Optional[Union[float, torch.Tensor, List[float], List[torch.Tensor]]],
        # float: Single dataset, all atoms shifted the same
        # torch.Tensor: Single dataset, each atom shifted by a different value
        # List[torch.Tensor] shape (n_dataset,): Multiple datasets, all atoms in each dataset shifted by a different value
        # List[torch.Tensor] shape (n_dataset, num_types): Multiple datasets, all atoms in each dataset shifted by a different value
        scales: Optional[Union[float, torch.Tensor, List[float], List[torch.Tensor]]], # Same as for shifts
        arguments_in_dataset_units: bool,
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = True,
        irreps_in={},
    ):
        super().__init__()
        self.num_types = num_types
        self.type_names = type_names
        self.field = field
        self.out_field = f"shifted_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={self.field: "0e"},  # input to shift must be a single scalar
            irreps_out={self.out_field: irreps_in[self.field]},
        )

        self.has_shifts = shifts is not None
        if shifts is not None:
            if isinstance(shifts, List):
                shifts = torch.stack(shifts, dim=0)
                if len(shifts.shape) == 1:
                    shifts = shifts.reshape(-1, 1)
                if shifts.shape[-1] == 1:
                    shifts = torch.ones(len(shifts), num_types) * shifts
                assert shifts.shape[-1] == num_types, f"Invalid shape of shifts {shifts}"
            else:
                if isinstance(shifts, float):
                    shifts = torch.as_tensor(shifts, dtype=torch.get_default_dtype())
            self.shifts_trainable = shifts_trainable

            if shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)
        else:
            self.register_buffer("shifts", torch.tensor([]))

        self.has_scales = scales is not None
        if scales is not None:
            if isinstance(scales, List):
                scales = torch.stack(scales, dim=0)
                if len(scales.shape) == 1:
                    scales = scales.reshape(-1, 1)
                if scales.shape[-1] == 1:
                    scales = torch.ones(len(scales), num_types) * scales
                assert scales.shape[-1] == num_types, f"Invalid shape of scales {scales}"
            else:
                if isinstance(scales, float):
                    scales = torch.as_tensor(scales, dtype=torch.get_default_dtype())

            self.scales_trainable = scales_trainable
            if scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)
        else:
            self.register_buffer("scales", torch.tensor([]))

        self.arguments_in_dataset_units = arguments_in_dataset_units

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        if self.production or not (self.has_scales or self.has_shifts):
            return data

        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        in_field = data[self.field]
        assert len(in_field) == len(
            species_idx
        ), "in_field doesnt seem to have correct per-atom shape"
        if self.has_scales:
            in_field = self.scales[species_idx].view(-1, 1) * in_field
        if self.has_shifts:
            in_field = self.shifts[species_idx].view(-1, 1) + in_field
        data[self.out_field] = in_field
        return data

    def update_for_rescale(self, rescale_module):
        if hasattr(rescale_module, "related_scale_keys"):
            if self.out_field not in rescale_module.related_scale_keys:
                return
        if self.arguments_in_dataset_units and rescale_module.has_scale:
            logging.debug(
                f"PerSpeciesScaleShift's arguments were in dataset units; rescaling:\n  "
                f"Original scales: {','.join([TypeMapper.format(scales, self.type_names) for scales in self.scales]) if self.has_scales else 'n/a'} "
                f"shifts: {','.join([TypeMapper.format(shifts, self.type_names) for shifts in self.shifts]) if self.has_shifts else 'n/a'}"
            )
            with torch.no_grad():
                if self.has_scales:
                    self.scales.div_(rescale_module.scale_by.reshape(1, 1))
                if self.has_shifts:
                    self.shifts.div_(rescale_module.scale_by.reshape(1, 1))
            logging.debug(
                f"  New scales: {','.join([TypeMapper.format(scales, self.type_names) for scales in self.scales]) if self.has_scales else 'n/a'} "
                f"shifts: {','.join([TypeMapper.format(shifts, self.type_names) for shifts in self.shifts]) if self.has_shifts else 'n/a'}"
            )
