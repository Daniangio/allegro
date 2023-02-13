from typing import List, Tuple, Union, Optional

import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


@compile_mode("script")
class CurlOutput(GraphModuleMixin, torch.nn.Module):
    r"""Wrap a model and include as an output its partial gradients.

    Args:
        func: the model to wrap
        of: the name of the output field of ``func`` to take the gradient with respect to. The field must be a single scalar (i.e. have irreps ``0e``)
        wrt: the input field(s) of ``func`` to take the gradient of ``of`` with regards to.
    """

    def __init__(
        self,
        func: GraphModuleMixin,
        of: str,
        wrt: str,
        out_field: str,
    ):
        super().__init__()
        self.of = of
        self.wrt = wrt
        self.out_field = out_field
        self.func = func

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={of: Irreps("1o")},
            irreps_out=func.irreps_out,
        )

        # The gradient of a single scalar w.r.t. something of a given shape and irrep just has that shape and irrep
        # Ex.: gradient of energy (0e) w.r.t. position vector (L=1) is also an L = 1 vector
        self.irreps_out.update(
            {
                self.out_field: self.irreps_in[self.wrt]
            }
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.production:
            data[self.out_field] = torch.zeros((1,), dtype=torch.float32).to(data["pos"].device)
            return self.func(data)
        return self.forward_impl(data)

    @torch.jit.unused
    def forward_impl(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # set req grad
        old_requires_grad = data[self.wrt].requires_grad
        data[self.wrt].requires_grad_(True)
        wrt_tensors = data[self.wrt]
        # run func
        data = self.func(data)
        out = data[self.of]

        external_grads_list: List[Tuple[int, torch.Tensor]] = []
        # for i in random.sample(torch.tensor(list(range(len(out)))), torch.tensor(0.1 * len(out), dtype=torch.int)):
        for i in torch.randint(0, len(out), (data[AtomicDataDict.BATCH_KEY].max().item() + 1,)):
            external_grad: torch.Tensor = torch.zeros_like(out)
            external_grad[i, 0] = 1.
            external_grads_list.append((i.item(), external_grad))
            external_grad: torch.Tensor = torch.zeros_like(out)
            external_grad[i, 1] = 1.
            external_grads_list.append((i.item(), external_grad))
            external_grad: torch.Tensor = torch.zeros_like(out)
            external_grad[i, 2] = 1.
            external_grads_list.append((i.item(), external_grad))

        grads_list: List[List[torch.Tensor]] = []
        triplet_list: List[torch.Tensor] = []
        for i, external_grad in external_grads_list:
            grad_outputs: Optional[List[Optional[torch.Tensor]]] = [external_grad]
            grads = torch.autograd.grad(
                        [out],
                        wrt_tensors,
                        create_graph=True,
                        retain_graph=True,
                        grad_outputs=grad_outputs,
                    )[0]
            assert grads is not None
            grads: torch.Tensor = grads[i]
            triplet_list.append(grads)
            if len(triplet_list) >= 3:
                grads_list.append(list(triplet_list))
                triplet_list.clear()
        
        gradients = torch.stack([torch.stack(g) for g in grads_list])
        cx = gradients[:, 2, 1] - gradients[:, 1, 2]
        cy = gradients[:, 0, 2] - gradients[:, 2, 0]
        cz = gradients[:, 1, 0] - gradients[:, 0, 1]
        curl = torch.stack([cx, cy, cz]).T
        data[self.out_field] = curl

        # unset requires_grad_
        data[self.wrt].requires_grad_(old_requires_grad)

        return data