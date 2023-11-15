from nequip.nn import GraphModuleMixin
from nequip.data import AtomicDataDict
from allegro.nn._norm_output import NormOutput


def NormOutputModule(config, model: GraphModuleMixin) -> NormOutput:

    return NormOutput(
        func=model,
        of=AtomicDataDict.FORCE_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
    )



    
