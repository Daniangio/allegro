from typing import Optional
import logging

from e3nn import o3

from torch.utils.data import ConcatDataset

from nequip.data import AtomicDataDict
from nequip.nn import SequentialGraphNetwork
from nequip.nn.radial_basis import BesselBasis

from nequip.nn.embedding import (
    OneHotAtomEncoding,
    SphericalHarmonicEdgeAttrs,
    RadialBasisSquaredEdgeEncoding,
)

from allegro.nn import (
    NormalizedBasis,
    EdgewiseEnergySum,
    Allegro_MACE_V2_Module,
    ExponentialScalarMLPFunction,
)
from allegro._keys import EDGE_ENERGY

from nequip.model import builder_utils


def Allegro_MACE_V2_sq_NMR(config, initialize: bool, dataset: Optional[ConcatDataset] = None):
    logging.debug("Building Allegro model...")

    # Handle avg num neighbors auto
    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    # Handle simple irreps
    if "l_max" in config:
        l_max = int(config["l_max"])
        parity_setting = config["parity"]
        assert parity_setting in ("o3_full", "o3_restricted", "so3")
        irreps_edge_sh = repr(
            o3.Irreps.spherical_harmonics(
                l_max, p=(1 if parity_setting == "so3" else -1)
            )
        )
        nonscalars_include_parity = parity_setting == "o3_full"
        # check consistant
        assert config.get("irreps_edge_sh", irreps_edge_sh) == irreps_edge_sh
        assert (
            config.get("nonscalars_include_parity", nonscalars_include_parity)
            == nonscalars_include_parity
        )
        config["irreps_edge_sh"] = irreps_edge_sh
        config["nonscalars_include_parity"] = nonscalars_include_parity

    layers = {
        # -- Encode --
        # Get various edge invariants
        "one_hot": (
            OneHotAtomEncoding,
            dict(
                node_input_features=config.get("node_input_features", [])
            )
        ),
        "radial_basis": (
            RadialBasisSquaredEdgeEncoding,
            dict(
                basis=(
                    NormalizedBasis
                    if config.get("normalize_basis", True)
                    else BesselBasis
                ),
                out_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            ),
        ),
        # Get edge nonscalars
        "spharm": SphericalHarmonicEdgeAttrs,
        # The core allegro model:
        "allegro": (
            Allegro_MACE_V2_Module,
            dict(
                field=AtomicDataDict.EDGE_ATTRS_KEY,  # initial input is the edge SH
                edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
                node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
                env_embed=ExponentialScalarMLPFunction,
            ),
        ),
        # Sum edgewise energies -> per-atom energies:
        "per_atom_energy": (
            EdgewiseEnergySum,
            dict(
                field=EDGE_ENERGY,
                out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY
            ),
        ),
    }

    model = SequentialGraphNetwork.from_parameters(shared_params=config, layers=layers)

    return model