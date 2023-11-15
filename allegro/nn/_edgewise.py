from typing import Optional
import math

import torch
from torch_runstats.scatter import scatter

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from .. import _keys


class EdgewiseReduce(GraphModuleMixin, torch.nn.Module):
    """Like ``nequip.nn.AtomwiseReduce``, but accumulating per-edge data into per-atom data."""

    _factor: Optional[float]

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        normalize_edge_reduce: bool = True,
        avg_num_neighbors: Optional[float] = None,
        reduce="sum",
        irreps_in={},
    ):
        """Sum edges into nodes."""
        super().__init__()
        assert reduce in ("sum", "mean", "min", "max")
        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_in[self.field]}
            if self.field in irreps_in
            else {},
        )
        self._factor = None
        if normalize_edge_reduce and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get destination nodes 🚂
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        out = scatter(
            data[self.field],
            edge_dst,
            dim=0,
            dim_size=len(data[AtomicDataDict.POSITIONS_KEY]),
            reduce=self.reduce,
        )

        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            out = out * factor

        data[self.out_field] = out

        return data


class EdgewiseEnergySum(GraphModuleMixin, torch.nn.Module):
    """Sum edgewise energies.

    Includes optional per-species-pair edgewise energy scales.
    """

    out_field: str
    _factor: Optional[float]

    def __init__(
        self,
        num_types: int,
        field: str = _keys.EDGE_FEATURES,
        out_field: str = AtomicDataDict.PER_ATOM_ENERGY_KEY,
        avg_num_neighbors: Optional[float] = None,
        normalize_edge_energy_sum: bool = True,
        per_edge_species_scale: float = None,
        irreps_in={},
    ):
        """Sum edges into nodes."""
        super().__init__()
        self.field = field
        self.out_field = out_field

        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={field: irreps_in[field]},
            irreps_out={out_field: irreps_in[field]},
        )

        self._factor = None
        if normalize_edge_energy_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

        self.per_edge_species_scale = per_edge_species_scale
        if self.per_edge_species_scale is not None:
            self.per_edge_scales = torch.nn.Parameter(self.per_edge_species_scale * torch.ones(num_types, num_types))
        else:
            self.register_buffer("per_edge_scales", torch.Tensor())

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        edge_eng = data[self.field]
        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        center_species = species[edge_center]
        neighbor_species = species[edge_neighbor]

        if self.per_edge_species_scale:
            edge_eng = edge_eng * self.per_edge_scales[
                center_species, neighbor_species
            ].reshape(-1, *[1 for _ in range(len(edge_eng.shape)-1)])

        atom_eng = scatter(edge_eng, edge_center, dim=0, dim_size=len(species)) # / torch.bincount(edge_center, minlength=len(species)).unsqueeze(1)
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            atom_eng = atom_eng * factor

        data[self.out_field] = atom_eng

        return data


class EdgewiseForcesSum(GraphModuleMixin, torch.nn.Module):
    """Sum edgewise forces.

    """

    _factor: Optional[float]

    def __init__(
        self,
        irreps_in={},
    ):
        """Sum edges into nodes."""
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={_keys.EDGE_FORCES: "1o"},
            irreps_out={AtomicDataDict.FORCE_KEY: "1o"},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        edge_f = data[_keys.EDGE_FORCES]
        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)

        atom_f = scatter(edge_f, edge_center, dim=0, dim_size=len(species))

        data[AtomicDataDict.FORCE_KEY] = atom_f
        # data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = atom_f.sum(dim=-1, keepdim=True)

        return data