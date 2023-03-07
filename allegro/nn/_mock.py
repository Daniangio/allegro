import torch
from torch_runstats.scatter import scatter

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from e3nn.o3 import Irreps


class OutputMock(GraphModuleMixin, torch.nn.Module):
    """"""

    def __init__(
        self,
        out_field: str,
        out_field_irreps: Irreps,
        irreps_in={},
    ):
        """ Mock out field values """
        super().__init__()
        self.out_field: str = out_field
        self.out_dim: int = out_field_irreps.dim
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: out_field_irreps}
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        pos = data[AtomicDataDict.POSITIONS_KEY]
        batch_n_atoms, _ = pos.shape
        data[self.out_field] = torch.zeros((batch_n_atoms, self.out_dim), dtype=torch.float32, device=pos.device)
        return data
