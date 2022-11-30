from nequip.nn import GraphModuleMixin
from nequip.data import AtomicDataDict
from allegro.nn._partial_grads import CurlOutput
from allegro._keys import CURL


def ForcesCurlOutput(config, model: GraphModuleMixin) -> CurlOutput:
    r"""Add forces to a model that predicts energy.

    Args:
        model: the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``GradientOutput`` wrapping ``model``.
    """
    
    if AtomicDataDict.FORCE_KEY not in model.irreps_out:
        raise ValueError("This model misses force outputs.")

    return CurlOutput(
        func=model,
        of=AtomicDataDict.FORCE_KEY,
        wrt=AtomicDataDict.POSITIONS_KEY,
        out_field=CURL,
        deployed=config.get("deployed", False)
    )



    
