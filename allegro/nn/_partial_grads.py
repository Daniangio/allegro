import random
from typing import List, Union, Optional

import torch

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
        wrt: Union[str, List[str]],
        out_field: str,
        deployed: bool = False
    ):
        super().__init__()
        self.deployed = deployed
        self.of = of

        # TO DO: maybe better to force using list?
        if isinstance(wrt, str):
            wrt = [wrt]
        self.wrt = wrt
        self.func = func
        self.out_field = out_field

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_in,
            irreps_out=func.irreps_out,
        )

        # The gradient of a single scalar w.r.t. something of a given shape and irrep just has that shape and irrep
        # Ex.: gradient of energy (0e) w.r.t. position vector (L=1) is also an L = 1 vector
        self.irreps_out.update(
            {f: self.irreps_in[wrt] for f, wrt in zip(self.out_field, self.wrt)}
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.deployed:
            return data

        # set req grad
        wrt_tensors = []
        old_requires_grad: List[bool] = []
        for k in self.wrt:
            old_requires_grad.append(data[k].requires_grad)
            data[k].requires_grad_(True)
            wrt_tensors.append(data[k])
        # run func
        data = self.func(data)
        out = data[self.of]

        external_grads_list = []
        for i in random.sample(range(len(out)), int(0.1 * len(out))):
            external_grad = torch.zeros_like(out)
            external_grad[i, 0] = 1.
            external_grads_list.append((i, external_grad))
            external_grad = torch.zeros_like(out)
            external_grad[i, 1] = 1.
            external_grads_list.append((i, external_grad))
            external_grad = torch.zeros_like(out)
            external_grad[i, 2] = 1.
            external_grads_list.append((i, external_grad))

        grads_list: List[List[torch.Tensor]] = []
        triplet_list: List[torch.Tensor] = []
        for i, external_grad in external_grads_list:
            grad_outputs: Optional[List[Optional[torch.Tensor]]] = [external_grad]
            grads = torch.autograd.grad(
                        [out],
                        wrt_tensors,
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
        for req_grad, k in zip(old_requires_grad, self.wrt):
            data[k].requires_grad_(req_grad)

        return data