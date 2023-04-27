from typing import Optional, List
import math
import functools

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torch_runstats.scatter import scatter

from e3nn import o3
from e3nn.o3 import TensorProduct
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from ._fc import ScalarMLPFunction, ExponentialScalarMLPFunction
from .. import _keys
from ._strided import MakeWeightedChannels, Linear
from .cutoffs import cosine_cutoff, polynomial_cutoff

from mace.modules.blocks import RealAgnosticInteractionBlock, EquivariantProductBasisBlock
from mace.modules.irreps_tools import reshape_irreps, inverse_reshape_irreps


# def pick_mpl_function(func):
#     if isinstance(func, Callable):
#         return func
#     assert isinstance(func, str)
#     if func.lower() == "ScalarMLPFunction".lower():
#         return ScalarMLPFunction
#     if func.lower() == "ExponentialScalarMLPFunction".lower():
#         return ExponentialScalarMLPFunction
#     raise Exception(f"MLP Funciton {func} not implemented.")


@compile_mode("script")
class Allegro_MACE_Module(GraphModuleMixin, torch.nn.Module):
    # saved params
    num_layers: int
    field: str
    out_field: str
    num_types: int
    env_embed_mul: int
    weight_numel: int
    latent_resnet: bool

    # internal values
    _env_builder_w_index: List[int]
    _env_builder_n_irreps: int
    _input_pad: int

    def __init__(
        self,
        # required params
        num_layers: int,
        num_types: int,
        r_max: float,
        avg_num_neighbors: Optional[float] = None,
        # cutoffs
        r_start_cos_ratio: float = 0.8,
        PolynomialCutoff_p: float = 6,
        per_layer_cutoffs: Optional[List[float]] = None,
        cutoff_type: str = "polynomial",
        # general hyperparameters:
        field: str = AtomicDataDict.EDGE_ATTRS_KEY,
        edge_invariant_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        node_invariant_field: str = AtomicDataDict.NODE_ATTRS_KEY,
        env_embed_multiplicity: int = 32,
        linear_after_env_embed: bool = True,
        nonscalars_include_parity: bool = True,
        product_correlation: int = 3,
        # MLP parameters:
        env_embed=ScalarMLPFunction,
        env_embed_kwargs={},
        two_body_latent=ScalarMLPFunction,
        two_body_latent_kwargs={},
        latent=ScalarMLPFunction,
        latent_kwargs={},
        latent_resnet: bool = True,
        latent_resnet_update_ratios: Optional[List[float]] = None,
        latent_resnet_update_ratios_learnable: bool = False,
        out_field: Optional[str] = _keys.EDGE_ENERGY,
        # Performance parameters:
        pad_to_alignment: int = 1,
        sparse_mode: Optional[str] = None,
        # Other:
        irreps_in=None,
    ):
        super().__init__()
        SCALAR = o3.Irrep("0e")  # define for convinience

        # save parameters
        assert (
            num_layers >= 1
        )  # zero layers is "two body", but we don't need to support that fallback case
        self.num_layers = num_layers
        self.nonscalars_include_parity = nonscalars_include_parity
        self.field = field
        self.out_field = out_field
        self.edge_invariant_field = edge_invariant_field
        self.node_invariant_field = node_invariant_field
        self.latent_resnet = latent_resnet
        self.env_embed_mul = env_embed_multiplicity
        self.r_start_cos_ratio = r_start_cos_ratio
        self.polynomial_cutoff_p = float(PolynomialCutoff_p)
        self.cutoff_type = cutoff_type
        assert cutoff_type in ("cosine", "polynomial")
        self.avg_num_neighbors = avg_num_neighbors
        self.linear_after_env_embed = linear_after_env_embed
        self.num_types = num_types

        # TODO build dynamic class loading
        # env_embed = pick_mpl_function(env_embed)
        # two_body_latent = pick_mpl_function(two_body_latent)
        # latent = pick_mpl_function(latent)

        # set up irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                self.field,
                self.edge_invariant_field,
                self.node_invariant_field,
            ],
        )

        # for normalization of features
        # one per layer
        self.register_buffer(
            "env_sum_normalizations",
            # dividing by sqrt(N)
            torch.as_tensor([avg_num_neighbors] * num_layers).reciprocal(),
        )

        latent = functools.partial(latent, **latent_kwargs)
        env_embed = functools.partial(env_embed, **env_embed_kwargs)

        self.latents = torch.nn.ModuleList([])
        self.env_embed_mlps = torch.nn.ModuleList([])

        self.products = torch.nn.ModuleList([])
        self.reshape_back_modules = torch.nn.ModuleList([])
        
        self.feature_linears = torch.nn.ModuleList([])
        self.env_linears = torch.nn.ModuleList([])

        # Embed to the spharm * it as mul
        input_irreps = self.irreps_in[self.field]
        # this is not inherant, but no reason to fix right now:
        assert all(mul == 1 for mul, ir in input_irreps)

        # Environment builder:
        self._initial_env_weighter = MakeWeightedChannels(
            irreps_in=input_irreps,
            multiplicity_out=env_embed_multiplicity,
            pad_to_alignment=pad_to_alignment,
        )

        env_embed_irreps = o3.Irreps([(env_embed_multiplicity, ir) for _, ir in input_irreps])
        assert (
            env_embed_irreps[0].ir == SCALAR
        ), "env_embed_irreps must start with scalars"
        self._input_pad = (
            int(math.ceil(env_embed_irreps.dim / pad_to_alignment)) * pad_to_alignment
        ) - env_embed_irreps.dim
        self.register_buffer("_zero", torch.zeros(1, 1))

        self._env_weighter = MakeWeightedChannels(
            irreps_in=input_irreps,
            multiplicity_out=env_embed_multiplicity,
            pad_to_alignment=pad_to_alignment,
        )

        # - begin irreps -
        for layer_idx in range(self.num_layers):
            # Make env embed mlp
            generate_n_weights = (
                self._env_weighter.weight_numel
            )  # the weight for the edge embedding
            if layer_idx == 0:
                # also need weights to embed the edge itself
                # this is because the 2 body latent is mixed in with the first layer
                # in terms of code
                generate_n_weights += self._initial_env_weighter.weight_numel
            
            # Make the env embed linear
            if self.linear_after_env_embed:
                self.env_linears.append(
                    Linear(
                        [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps],
                        [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps],
                        shared_weights=True,
                        internal_weights=True,
                    )
                )
            else:
                self.env_linears.append(torch.nn.Identity())
            
            # Make interaction
            interaction_irreps: o3.Irreps = (self.irreps_in[self.field] * env_embed_multiplicity).sort()[0].simplify()

            # Make product
            self.products.append(
                EquivariantProductBasisBlock(
                    node_feats_irreps=interaction_irreps,
                    target_irreps=env_embed_irreps,
                    correlation=product_correlation,
                    num_elements=self.irreps_in[self.node_invariant_field].dim,
                    use_sc=False,
                )
            )

            # Reshape back product so that you can perform tp
            self.reshape_back_modules.append(reshape_irreps(env_embed_irreps))

            # the linear acts after the extractor
            self.feature_linears.append(
                Linear(
                    env_embed_irreps,
                    env_embed_irreps,
                    shared_weights=True,
                    internal_weights=True,
                    pad_to_alignment=pad_to_alignment,
                )
            )

            self._n_degree_l_outs: List[int] = [2*ir.l + 1 for _, ir in env_embed_irreps]

            if layer_idx == 0:
                # at the first layer, we have no invariants from previous products
                self.latents.append(
                    two_body_latent(
                        mlp_input_dimension=(
                            (
                                # Node invariants for center and neighbor (chemistry)
                                2 * self.irreps_in[self.node_invariant_field].num_irreps
                                # Plus edge invariants for the edge (radius).
                                + self.irreps_in[self.edge_invariant_field].num_irreps
                            )
                        ),
                        mlp_output_dimension=None,
                        **two_body_latent_kwargs,
                    )
                )
                self._latent_dim = self.latents[-1].out_features
            else:
                self.latents.append(
                    latent(
                        mlp_input_dimension=(
                            (
                                # the embedded latent invariants from the previous layer(s)
                                self.latents[-1].out_features
                                # and the invariants extracted from the last layer's product:
                                + env_embed_multiplicity * len(self._n_degree_l_outs)
                            )
                        ),
                        mlp_output_dimension=None,
                    )
                )
            
            # the env embed MLP takes the last latent's output as input
            # and outputs enough weights for the env embedder
            self.env_embed_mlps.append(
                env_embed(
                    mlp_input_dimension=self.latents[-1].out_features,
                    mlp_output_dimension=generate_n_weights,
                )
            )

        # For the final layer, we specialize:
        # we don't need to propagate nonscalars, so there is no product
        # thus we only need the latent:
        self._features_final_dim = env_embed_irreps.dim
        self.final_linear = Linear(
                    env_embed_irreps,
                    env_embed_irreps,
                    shared_weights=False,
                    internal_weights=False,
                    pad_to_alignment=pad_to_alignment,
                )
        
        self.final_latent_to_weights = latent(
            mlp_input_dimension=self.latents[-1].out_features,
            mlp_output_dimension=self.final_linear.weight_numel,
        )

        final_out_irreps = o3.Irreps([(env_embed_multiplicity, (0, 1)) for _ in range(env_embed_irreps.lmax + 1)])

        instr = []
        for i_out, (_, ir_out) in enumerate(final_out_irreps):
            tmp_i_out: int = 0
            for i_1, (_, ir_1) in enumerate(env_embed_irreps):
                for i_2, (_, ir_2) in enumerate(env_embed_irreps):
                    if ir_out in ir_1 * ir_2:
                        instr.append((i_1, i_2, tmp_i_out, "uuu", False))
                        tmp_i_out += 1
        
        self.final_tp = TensorProduct(
                irreps_in1=env_embed_irreps,
                irreps_in2=env_embed_irreps,
                irreps_out=final_out_irreps,
                instructions=instr,
                shared_weights=True,
            )

        self.readout = latent(
            mlp_input_dimension=self.latents[-1].out_features
            + final_out_irreps.dim,
            mlp_output_dimension=1,
        )
        # - end build modules -

        # - layer resnet update weights -
        if latent_resnet_update_ratios is None:
            # We initialize to zeros, which under the sigmoid() become 0.5
            # so 1/2 * layer_1 + 1/4 * layer_2 + ...
            # note that the sigmoid of these are the factor _between_ layers
            # so the first entry is the ratio for the latent resnet of the first and second layers, etc.
            # e.g. if there are 3 layers, there are 2 ratios: l1:l2, l2:l3
            latent_resnet_update_params = torch.zeros(self.num_layers)
        else:
            latent_resnet_update_ratios = torch.as_tensor(
                latent_resnet_update_ratios, dtype=torch.get_default_dtype()
            )
            assert latent_resnet_update_ratios.min() > 0.0
            assert latent_resnet_update_ratios.min() < 1.0
            latent_resnet_update_params = torch.special.logit(
                latent_resnet_update_ratios
            )
            # The sigmoid is mostly saturated at ±6, keep it in a reasonable range
            latent_resnet_update_params.clamp_(-6.0, 6.0)
        assert latent_resnet_update_params.shape == (
            num_layers,
        ), f"There must be {num_layers} layer resnet update ratios (layer0:layer1, layer1:layer2)"
        if latent_resnet_update_ratios_learnable:
            self._latent_resnet_update_params = torch.nn.Parameter(
                latent_resnet_update_params
            )
        else:
            self.register_buffer(
                "_latent_resnet_update_params", latent_resnet_update_params
            )

        # - Per-layer cutoffs -
        if per_layer_cutoffs is None:
            per_layer_cutoffs = torch.full((num_layers + 1,), r_max)
        self.register_buffer("per_layer_cutoffs", torch.as_tensor(per_layer_cutoffs))
        assert torch.all(self.per_layer_cutoffs <= r_max)
        assert self.per_layer_cutoffs.shape == (
            num_layers + 1,
        ), "Must be one per-layer cutoff for layer 0 and every layer for a total of {num_layers} cutoffs (the first applies to the two body latent, which is 'layer 0')"
        assert (
            self.per_layer_cutoffs[1:] <= self.per_layer_cutoffs[:-1]
        ).all(), "Per-layer cutoffs must be equal or decreasing"
        assert (
            self.per_layer_cutoffs.min() > 0
        ), "Per-layer cutoffs must be >0. To remove higher layers entirely, lower `num_layers`."
        self.register_buffer("_zero", torch.as_tensor(0.0))

        self.register_parameter(
            "final_features_modulator",
            torch.nn.Parameter(torch.as_tensor(1.)),
        )

        self.irreps_out.update(
            {
                self.out_field: o3.Irreps(
                    [(1, (0, 1))]
                ),
            }
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """Evaluate.

        :param data: AtomicDataDict.Type
        :return: AtomicDataDict.Type
        """

        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        edge_attr = data[self.field]
        # pad edge_attr
        if self._input_pad > 0:
            edge_attr = torch.cat(
                (
                    edge_attr,
                    self._zero.expand(len(edge_attr), self._input_pad),
                ),
                dim=-1,
            )

        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
        num_edges: int = len(edge_attr)
        edge_invariants = data[self.edge_invariant_field]
        node_invariants = data[self.node_invariant_field]
        # pre-declare variables as Tensors for TorchScript
        scalars = self._zero
        coefficient_old = scalars
        coefficient_new = scalars
        # Initialize state
        latents = torch.zeros(
            (num_edges, self._latent_dim),
            dtype=edge_attr.dtype,
            device=edge_attr.device,
        )
        active_edges = torch.arange(
            num_edges,
            device=edge_attr.device,
        )

        # The nonscalar features. Initially, the edge data.
        features = torch.zeros(
            (num_edges, self._features_final_dim),
            dtype=edge_attr.dtype,
            device=edge_attr.device,
        )

        # For the first layer, we use the input invariants:
        # The center and neighbor invariants and edge invariants
        latent_inputs_to_cat = [
            node_invariants[edge_center],
            node_invariants[edge_neighbor],
            edge_invariants,
        ]

        layer_index: int = 0
        # compute the sigmoids vectorized instead of each loop
        layer_update_coefficients = self._latent_resnet_update_params.sigmoid()

        # Vectorized precompute per layer cutoffs
        if self.cutoff_type == "cosine":
            cutoff_coeffs_all = cosine_cutoff(
                edge_length,
                self.per_layer_cutoffs,
                r_start_cos_ratio=self.r_start_cos_ratio,
            )
        elif self.cutoff_type == "polynomial":
            cutoff_coeffs_all = polynomial_cutoff(
                edge_length, self.per_layer_cutoffs, p=self.polynomial_cutoff_p
            )
        else:
            # This branch is unreachable (cutoff type is checked in __init__)
            # But TorchScript doesn't know that, so we need to make it explicitly
            # impossible to make it past so it doesn't throw
            # "cutoff_coeffs_all is not defined in the false branch"
            assert False, "Invalid cutoff type"

        # !!!! REMEMBER !!!! update final layer if update the code in main loop!!!
        # This goes through layer0, layer1, ..., layer_max-1
        for latent, env_embed_mlp, env_linear, \
            feature_linear, prod, reshape_back in zip(
            self.latents, self.env_embed_mlps, self.env_linears,
            self.feature_linears, self.products, self.reshape_back_modules,
        ):
            # Determine which edges are still in play
            cutoff_coeffs = cutoff_coeffs_all[layer_index]
            prev_mask = cutoff_coeffs[active_edges] > -1e-3
            active_edges = (cutoff_coeffs > -1e-3).nonzero().squeeze(-1)

            # Compute latents
            new_latents = latent(torch.cat(latent_inputs_to_cat, dim=-1)[prev_mask])
            # Apply cutoff, which propagates through to everything else
            new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents

            if self.latent_resnet and layer_index > 0:
                this_layer_update_coeff = layer_update_coefficients[layer_index - 1]
                # At init, we assume new and old to be approximately uncorrelated
                # Thus their variances add
                # we always want the latent space to be normalized to variance = 1.0,
                # because it is critical for learnability. Still, we want to preserve
                # the _relative_ magnitudes of the current latent and the residual update
                # to be controled by `this_layer_update_coeff`
                # Solving the simple system for the two coefficients:
                #   a^2 + b^2 = 1  (variances add)   &    a * this_layer_update_coeff = b
                # gives
                #   a = 1 / sqrt(1 + this_layer_update_coeff^2)  &  b = this_layer_update_coeff / sqrt(1 + this_layer_update_coeff^2)
                # rsqrt is reciprocal sqrt
                coefficient_old = torch.rsqrt(this_layer_update_coeff.square() + 1)
                coefficient_new = this_layer_update_coeff * coefficient_old
                # Residual update
                # Note that it only runs when there are latents to resnet with, so not at the first layer
                # index_add adds only to the edges for which we have something to contribute
                latents = torch.index_add(
                    coefficient_old * latents,
                    0,
                    active_edges,
                    coefficient_new * new_latents,
                )
            else:
                # Normal (non-residual) update
                # index_copy replaces, unlike index_add
                latents = torch.index_copy(latents, 0, active_edges, new_latents)

            # From the latents, compute the weights for active edges:
            weights = env_embed_mlp(latents[active_edges])
            w_index: int = 0

            if layer_index == 0:
                # embed initial edge
                env_w = weights.narrow(-1, w_index, self._initial_env_weighter.weight_numel)
                w_index += self._initial_env_weighter.weight_numel
                old_features = self._initial_env_weighter(
                    edge_attr[prev_mask], env_w
                )  # features is edge_attr
            else:
                # just take the previous features that we still need
                old_features = features[active_edges]

            # Extract weights for the environment builder
            env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
            w_index += self._env_weighter.weight_numel

            # Build the local environments
            # This local environment should only be a sum over neighbors
            # who are within the cutoff of the _current_ layer
            # Those are the active edges, which are the only ones we
            # have weights for (env_w) anyway.
            # So we mask out the edges in the sum:
            emb_latent = self._env_weighter(edge_attr[active_edges], env_w)
            old_features = old_features.reshape(emb_latent.shape[0], emb_latent.shape[1], -1)

            local_env_per_atom = torch.zeros((len(node_invariants), emb_latent.shape[1], emb_latent.shape[2]), dtype=emb_latent.dtype, device=emb_latent.device)
            update_local_env_per_atom = scatter(
                emb_latent + old_features,
                edge_center[active_edges],
                dim=0,
            )
            local_env_per_atom[:edge_center[active_edges].max()+1] += update_local_env_per_atom
            if self.env_sum_normalizations.ndim < 2:
                # it's a scalar per layer
                norm_const = self.env_sum_normalizations[layer_index]
            else:
                # it's per type
                # get shape [N_atom, 1] for broadcasting
                norm_const = self.env_sum_normalizations[
                    layer_index, data[AtomicDataDict.ATOM_TYPE_KEY]
                ].unsqueeze(-1)
            local_env_per_atom = local_env_per_atom * norm_const
            local_env_per_atom = env_linear(local_env_per_atom)

            new_features: torch.Tensor = prod(
                node_feats=local_env_per_atom, sc=None, node_attrs=node_invariants
            )

            new_features: torch.Tensor = reshape_back(new_features)

            # Copy to get per-edge
            # Large allocation, but no better way to do this:
            new_features: torch.Tensor = new_features[edge_center[active_edges]]

            # Get invariants
            # new_features has shape [z][mul][k]
            scalars_to_cat = []
            last_n_degree_l_outs: int = 0
            for n_degree_l_outs in self._n_degree_l_outs:
                if n_degree_l_outs == 1:
                    scalar = new_features[:, :, 0]
                else:
                    scalar = new_features[:, :, last_n_degree_l_outs:n_degree_l_outs].norm(p=2, dim=-1)
                scalars_to_cat.append(
                    scalar
                )
                last_n_degree_l_outs = n_degree_l_outs
            
            scalars = torch.cat(scalars_to_cat, dim=-1).reshape(
                new_features.shape[0], -1
            )

            # do the linear
            latent_inputs_to_cat = [
                latents[active_edges],
                scalars,
            ]

            update_features = feature_linear(new_features)
            update_features = update_features.reshape(len(update_features), -1)
            features = torch.index_copy(features, 0, active_edges, update_features)

            # increment counter
            layer_index += 1

        # final linear
        cutoff_coeffs = cutoff_coeffs_all[layer_index]
        prev_mask = cutoff_coeffs[active_edges] > -1e-3
        active_edges = (cutoff_coeffs > -1e-3).nonzero().squeeze(-1)

        linear_weights = self.final_latent_to_weights(latents)
        final_features = self.final_linear(features, w=linear_weights).reshape(features.shape[0], -1) * self.final_features_modulator
        
        final_latent = self.final_tp(final_features, final_features).squeeze()
        data[self.out_field] = self.readout(torch.cat([latents, final_latent], dim=-1))

        return data