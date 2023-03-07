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
    RadialBasisEdgeEncoding,
)
from nequip.nn._atomwise import AtomwiseLinear, AtomwiseReduce

from allegro.nn import (
    NormalizedBasis,
    EdgewiseForcesSum,
    AllegroGDML_Module,
    OutputMock,
)
from allegro._keys import EDGE_FEATURES, EDGE_FORCES

from nequip.model import builder_utils


def AllegroGDML(config, initialize: bool, dataset: Optional[ConcatDataset] = None):
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
        "one_hot": OneHotAtomEncoding,
        "radial_basis": (
            RadialBasisEdgeEncoding,
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
            AllegroGDML_Module,
            dict(
                field=AtomicDataDict.EDGE_ATTRS_KEY,  # initial input is the edge SH
                edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
                node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
            ),
        ),
        "edge_f": (
            AtomwiseLinear,
            dict(field=EDGE_FEATURES, out_field=EDGE_FORCES),
        ),
        # Sum edgewise forces -> per-atom forces:
        "node_f": EdgewiseForcesSum,
        "per_atom_energy": (
            OutputMock,
            dict(
                out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field_irreps=o3.Irreps("1x0e")
            ),
        ),
        "total_energy_sum": (
            AtomwiseReduce,
            dict(
                reduce="sum",
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
            ),
        ),
    }

    model = SequentialGraphNetwork.from_parameters(shared_params=config, layers=layers)

    return model