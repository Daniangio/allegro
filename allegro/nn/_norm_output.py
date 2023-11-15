import torch
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


@compile_mode("script")
class NormOutput(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        func: GraphModuleMixin,
        of: str,
        out_field: str,
    ):
        super().__init__()
        self.of = of
        self.out_field = out_field
        self.func = func

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={of: Irreps("1o")},
            irreps_out=func.irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = self.func(data)
        out = data[self.of]
        data[self.out_field] = torch.norm(out, dim=-1, keepdim=False)
        return data