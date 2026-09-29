# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""1D CP encoder modules for 2D mesh ``(dp, cp)``.

These are thin wrappers around the existing 2D CP encoder modules
(``SingleConditioning``, ``PairwiseConditioning``, ``FourierEmbedding``)
that adapt the placement assertions for the simpler 2D mesh topology.

Under 2D CP (3D mesh ``(dp, cp0, cp1)``), placements are 3-element:

- Single ``s``: ``(Shard(0), Shard(1), Replicate())``
- Pair ``z``: ``(Shard(0), Shard(1), Shard(2))``
- Scalar ``times``: ``(Shard(0), Replicate(), Replicate())``

Under 1D CP (2D mesh ``(dp, cp)``), placements are 2-element:

- Single ``s``: ``(Shard(0), Shard(1))``
- Pair ``z``: ``(Shard(0), Shard(1))`` (row-slab)
- Scalar ``times``: ``(Shard(0), Replicate())``

The computation logic is identical — only the input validation differs.
"""

from math import pi

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from boltz.distributed.model.layers.cat_and_chunk import shardwise_cat
from boltz.distributed.model.layers.elementwise_op import ElementwiseOp, elementwise_op, single_tensor_op
from boltz.distributed.model.layers.layernorm import LayerNormParamsReplicated
from boltz.distributed.model.layers.linear import LinearParamsReplicated
from boltz.distributed.model.layers.replicate_op import ReplicateOp, replicate_op
from boltz.distributed.model.layers.squeeze import shardwise_unsqueeze
from boltz.distributed.model.layers.transition import Transition
from boltz.distributed.utils import all_gather_on_cp, update_exhaustive_strides
from boltz.model.modules.encodersv2 import FourierEmbedding as SerialFourierEmbeddingV2
from boltz.model.modules.encodersv2 import PairwiseConditioning as SerialPairwiseConditioningV2
from boltz.model.modules.encodersv2 import RelativePositionEncoder as SerialRelativePositionEncoderV2
from boltz.model.modules.encodersv2 import SingleConditioning as SerialSingleConditioningV2
from boltz.model.modules.encodersv2 import one_hot

# Pair (row-slab) placements on the 1D-CP 2D mesh ``(dp, cp)``.
PAIR_PLACEMENTS_1D = (Shard(0), Shard(1))


class FourierEmbedding1D(nn.Module):
    """1D CP FourierEmbedding for 2D mesh ``(dp, cp)``.

    Same computation as :class:`~.encoders.FourierEmbedding` but asserts
    2-element placements ``(Shard(0), Replicate())`` for times.
    """

    def __init__(
        self,
        layer: SerialFourierEmbeddingV2,
        device_mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        assert isinstance(layer, SerialFourierEmbeddingV2), f"Expected SerialFourierEmbeddingV2, got {type(layer)}"
        self.device_mesh = device_mesh
        self.proj = LinearParamsReplicated(layer.proj, device_mesh)
        if self.proj.weight.requires_grad or self.proj.bias.requires_grad:
            raise ValueError("Linear layer in FourierEmbedding should not have trainable parameters")

    def forward(self, times: DTensor) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        times : DTensor
            Shape ``(B,)`` with placements ``(Shard(0), Replicate())``.

        Returns
        -------
        DTensor
            Fourier embedding, shape ``(B, D)``, placements ``(Shard(0), Replicate())``.
        """
        expected_placements = (Shard(0), Replicate())
        if times.placements != expected_placements:
            raise ValueError(f"Times tensor has incorrect placements: {times.placements} != {expected_placements}")

        if times.ndim != 1:
            raise ValueError(f"Times tensor should have shape (B,) but got {times.shape}")

        times = shardwise_unsqueeze(times, dim=1)
        rand_proj = self.proj(times)
        return single_tensor_op(2 * pi * rand_proj, ElementwiseOp.COS)


class SingleConditioning1D(nn.Module):
    """1D CP SingleConditioning for 2D mesh ``(dp, cp)``.

    Same computation as :class:`~.encoders.SingleConditioning` but asserts
    2-element placements.

    Placements:

    - ``times``: ``(Shard(0), Replicate())``
    - ``s_trunk``, ``s_inputs``: ``(Shard(0), Shard(1))``
    """

    def __init__(
        self,
        layer: SerialSingleConditioningV2,
        device_mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        assert isinstance(layer, SerialSingleConditioningV2), f"Expected SerialSingleConditioningV2, got {type(layer)}"
        self.device_mesh = device_mesh

        self.norm_single = LayerNormParamsReplicated(layer.norm_single, device_mesh)
        self.single_embed = LinearParamsReplicated(layer.single_embed, device_mesh)

        self.disable_times = getattr(layer, "disable_times", False)
        if not self.disable_times:
            self.fourier_embed = FourierEmbedding1D(layer.fourier_embed, device_mesh)
            self.norm_fourier = LayerNormParamsReplicated(layer.norm_fourier, device_mesh)
            self.fourier_to_single = LinearParamsReplicated(layer.fourier_to_single, device_mesh)

        self.transitions = nn.ModuleList()
        for serial_transition in layer.transitions:
            self.transitions.append(Transition(layer=serial_transition, device_mesh=device_mesh))

    def forward(self, times: DTensor, s_trunk: DTensor, s_inputs: DTensor) -> tuple[DTensor, DTensor | None]:
        """Forward pass.

        Parameters
        ----------
        times : DTensor
            Shape ``(B,)`` with placements ``(Shard(0), Replicate())``.
        s_trunk : DTensor
            Shape ``(B, N, D)`` with placements ``(Shard(0), Shard(1))``.
        s_inputs : DTensor
            Shape ``(B, N, D)`` with placements ``(Shard(0), Shard(1))``.

        Returns
        -------
        tuple[DTensor, DTensor | None]
            ``(s, normed_fourier)`` where ``s`` has placements ``(Shard(0), Shard(1))``
            and ``normed_fourier`` has placements ``(Shard(0), Replicate())``.
        """
        expected_times = (Shard(0), Replicate())
        expected_s = (Shard(0), Shard(1))

        if times.placements != expected_times:
            raise ValueError(f"times placements {times.placements} != {expected_times}")
        if s_trunk.placements != expected_s:
            raise ValueError(f"s_trunk placements {s_trunk.placements} != {expected_s}")
        if s_inputs.placements != expected_s:
            raise ValueError(f"s_inputs placements {s_inputs.placements} != {expected_s}")

        s = shardwise_cat([s_trunk, s_inputs], dim=-1)
        s = self.single_embed(self.norm_single(s))

        normed_fourier: DTensor | None = None
        if not self.disable_times:
            fourier_embed = self.fourier_embed(times)
            normed_fourier = self.norm_fourier(fourier_embed)
            fourier_to_single = self.fourier_to_single(normed_fourier)
            # fourier_to_single: (B, D) with (S(0), R) — broadcast-add to s: (B, N, D) with (S(0), S(1))
            s = replicate_op(s, fourier_to_single, dim_to_unsqueeze_rhs=1, op=ReplicateOp.ADD)

        for transition in self.transitions:
            s = elementwise_op(transition(s), s, ElementwiseOp.SUM)

        return s, normed_fourier


class PairwiseConditioning1D(nn.Module):
    """1D CP PairwiseConditioning for 2D mesh ``(dp, cp)``.

    Same computation as :class:`~.encoders.PairwiseConditioning` but asserts
    2-element placements ``(Shard(0), Shard(1))`` for pair inputs (row-slab).
    """

    def __init__(
        self,
        layer: SerialPairwiseConditioningV2,
        device_mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        assert isinstance(
            layer, SerialPairwiseConditioningV2
        ), f"Expected SerialPairwiseConditioningV2, got {type(layer)}"
        self.device_mesh = device_mesh

        self.dim_pairwise_init_proj = nn.Sequential(
            LayerNormParamsReplicated(layer.dim_pairwise_init_proj[0], device_mesh),
            LinearParamsReplicated(layer.dim_pairwise_init_proj[1], device_mesh),
        )

        self.transitions = nn.ModuleList()
        for serial_transition in layer.transitions:
            self.transitions.append(Transition(layer=serial_transition, device_mesh=device_mesh))

    def forward(self, z_trunk: DTensor, token_rel_pos_feats: DTensor) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        z_trunk : DTensor
            Shape ``(B, N, N, D)`` with placements ``(Shard(0), Shard(1))``.
        token_rel_pos_feats : DTensor
            Shape ``(B, N, N, D_rel)`` with placements ``(Shard(0), Shard(1))``.

        Returns
        -------
        DTensor
            Conditioned pair representation, shape ``(B, N, N, D)``,
            placements ``(Shard(0), Shard(1))``.
        """
        expected = (Shard(0), Shard(1))
        if z_trunk.placements != expected:
            raise ValueError(f"z_trunk placements {z_trunk.placements} != {expected}")
        if token_rel_pos_feats.placements != expected:
            raise ValueError(f"token_rel_pos_feats placements {token_rel_pos_feats.placements} != {expected}")

        z = shardwise_cat([z_trunk, token_rel_pos_feats], dim=-1)
        z = self.dim_pairwise_init_proj(z)

        for transition in self.transitions:
            z = elementwise_op(transition(z), z, ElementwiseOp.SUM)

        return z


class RelativePositionEncoder1D(nn.Module):
    """1D-CP RelativePositionEncoder for the 2D mesh ``(dp, cp)``.

    Computes pairwise relative-position features from the single-representation
    token features (``asym_id``, ``residue_index``, ``entity_id``,
    ``token_index``, ``sym_id``, ``cyclic_period``) and projects them through a
    replicated linear layer.  This is the 1D-CP counterpart to the 2D
    :class:`~boltz.distributed.model.modules.encoders.RelativePositionEncoder`:
    where the 2D version fetches column shards via ``redistribute_transpose``
    (a square 2D mesh), the 1D version all-gathers the column features along the
    flat cp group (``all_gather_on_cp``).  The two cannot share a forward
    because the collective pattern is dictated by the mesh topology.

    The same instance is reused by the trunk (``Boltz2_1D``) and the confidence
    module (``ConfidenceModule1D``) — there is a single implementation, so any
    correctness fix applies to both call sites.

    All feature computation (outer comparisons, clipping, one-hot) is
    non-differentiable and runs on local tensors; only the final linear
    projection (``LinearParamsReplicated``) is differentiable, and its backward
    handles the parameter-gradient all-reduce.

    Even token sharding (``N % cp == 0``) is a data-pipeline invariant — both
    ``trainingv2_1d`` and ``inferencev2_1d`` pad every ``Shard``-placed token
    feature (``cyclic_period`` included) to a multiple of ``cp_size``.  This
    module therefore *asserts* uniform shard sizes rather than repairing
    heterogeneous shards inside the forward.

    Communication budget (forward):
        - 5 all-gathers over cp (asym_id, entity_id, residue_index,
          token_index, sym_id) for the column features.
        - 1 scalar ``all_reduce(MAX)`` of a ``has_cyclic`` flag (only when
          ``cyclic_pos_enc`` is enabled).
        - 1 additional all-gather of ``cyclic_period`` only when some rank
          carries a cyclic token.
    """

    def __init__(
        self,
        layer: SerialRelativePositionEncoderV2,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> None:
        """Initialize the 1D-CP RelativePositionEncoder.

        Parameters
        ----------
        layer : SerialRelativePositionEncoderV2
            The serial Boltz-2 ``RelativePositionEncoder`` to distribute.
        device_mesh : DeviceMesh
            The 2D device mesh ``(dp, cp)``.
        cp_group : dist.ProcessGroup
            The flat cp process group used for column all-gathers.
        """
        super().__init__()
        if not isinstance(layer, SerialRelativePositionEncoderV2):
            raise TypeError(f"Expected SerialRelativePositionEncoderV2, got {type(layer)}")
        self.device_mesh = device_mesh
        self.cp_group = cp_group
        self.r_max = layer.r_max
        self.s_max = layer.s_max
        self.fix_sym_check = layer.fix_sym_check
        self.cyclic_pos_enc = layer.cyclic_pos_enc
        # Attribute name mirrors the serial module so named_parameters() lines
        # up: ``rel_pos.linear_layer.weight`` on both sides.
        self.linear_layer = LinearParamsReplicated(layer.linear_layer, device_mesh=device_mesh)

    def forward(self, feats: dict[str, DTensor]) -> DTensor:
        """Compute relative-position embeddings.

        Parameters
        ----------
        feats : dict[str, DTensor]
            Must contain ``asym_id``, ``entity_id``, ``residue_index``,
            ``token_index``, ``sym_id``, ``cyclic_period``, each shape
            ``(B, N)`` with placements ``(Shard(0), Shard(1))``.

        Returns
        -------
        DTensor
            ``(B, N, N, token_z)`` row-slab pair tensor with placements
            ``PAIR_PLACEMENTS_1D``.
        """
        # Validate input placements (mirrors the sibling 1D encoders): the
        # column features are read via to_local(), so a wrong placement would
        # silently corrupt the result instead of failing.
        expected_placements = (Shard(0), Shard(1))
        for key in ("asym_id", "entity_id", "residue_index", "token_index", "sym_id", "cyclic_period"):
            dt = feats[key]
            if not isinstance(dt, DTensor):
                raise TypeError(f"Expected DTensor for '{key}', got {type(dt)}")
            if dt.placements != expected_placements:
                raise ValueError(f"'{key}' placements {dt.placements} != expected {expected_placements}")

        cp_size = self.device_mesh.shape[1]

        B_global = feats["asym_id"].shape[0]
        N_global = feats["asym_id"].shape[1]
        if N_global % cp_size != 0:
            raise ValueError(
                f"Uneven token sharding: N_global ({N_global}) is not divisible by "
                f"cp_size ({cp_size}). Token features must be padded to a multiple of "
                f"cp_size in the data pipeline (trainingv2_1d / inferencev2_1d)."
            )

        # Extract row-local tensors.
        asym_id_local = feats["asym_id"].to_local()
        entity_id_local = feats["entity_id"].to_local()
        residue_idx_local = feats["residue_index"].to_local()
        token_idx_local = feats["token_index"].to_local()
        sym_id_local = feats["sym_id"].to_local()
        cyclic_period_local = feats["cyclic_period"].to_local()

        # All-gather the column features along cp (uniform shards by invariant).
        asym_id_full = all_gather_on_cp(asym_id_local, 1, self.cp_group, cp_size)
        entity_id_full = all_gather_on_cp(entity_id_local, 1, self.cp_group, cp_size)
        residue_idx_full = all_gather_on_cp(residue_idx_local, 1, self.cp_group, cp_size)
        token_idx_full = all_gather_on_cp(token_idx_local, 1, self.cp_group, cp_size)
        sym_id_full = all_gather_on_cp(sym_id_local, 1, self.cp_group, cp_size)

        # Outer comparisons: row (local) x col (full).
        b_same_chain = torch.eq(asym_id_local[:, :, None], asym_id_full[:, None, :])
        b_same_residue = torch.eq(residue_idx_local[:, :, None], residue_idx_full[:, None, :])
        b_same_entity = torch.eq(entity_id_local[:, :, None], entity_id_full[:, None, :])

        d_residue = residue_idx_local[:, :, None] - residue_idx_full[:, None, :]

        # Cyclic-period correction.  ``torch.any(cyclic_period_local > 0)`` can
        # disagree across ranks — a rank holding only non-cyclic (or padding)
        # tokens evaluates it False while another evaluates it True.  If ranks
        # then diverge on whether to enter ``all_gather_on_cp`` the collective
        # deadlocks: a single-device-semantics violation that only manifests
        # under context parallelism.  Broadcasting the decision via a scalar
        # ``all_reduce(MAX)`` forces every rank to enter/skip in lockstep.  The
        # whole block is guarded by ``cyclic_pos_enc`` (a model-level config
        # identical on all ranks) so the all_reduce is skipped entirely when
        # cyclic encoding is disabled.
        if self.cyclic_pos_enc:
            has_cyclic = torch.tensor(
                int(torch.any(cyclic_period_local > 0)),
                dtype=torch.long,
                device=cyclic_period_local.device,
            )
            dist.all_reduce(has_cyclic, op=dist.ReduceOp.MAX, group=self.cp_group)
            if bool(has_cyclic):
                # cyclic_period is column-indexed (period of token j), so gather
                # the full column and broadcast over the local rows.
                cyclic_period_full = all_gather_on_cp(cyclic_period_local, 1, self.cp_group, cp_size)
                period = torch.where(
                    cyclic_period_full > 0,
                    cyclic_period_full,
                    torch.zeros_like(cyclic_period_full) + 10000,
                ).unsqueeze(1)
                d_residue = (d_residue - period * torch.round(d_residue / period)).long()

        d_residue = torch.clip(d_residue + self.r_max, 0, 2 * self.r_max)
        d_residue = torch.where(b_same_chain, d_residue, torch.zeros_like(d_residue) + 2 * self.r_max + 1)
        a_rel_pos = one_hot(d_residue, 2 * self.r_max + 2)

        d_token = torch.clip(
            token_idx_local[:, :, None] - token_idx_full[:, None, :] + self.r_max,
            0,
            2 * self.r_max,
        )
        d_token = torch.where(
            b_same_chain & b_same_residue,
            d_token,
            torch.zeros_like(d_token) + 2 * self.r_max + 1,
        )
        a_rel_token = one_hot(d_token, 2 * self.r_max + 2)

        d_chain = torch.clip(
            sym_id_local[:, :, None] - sym_id_full[:, None, :] + self.s_max,
            0,
            2 * self.s_max,
        )
        d_chain = torch.where(
            (~b_same_entity) if self.fix_sym_check else b_same_chain,
            torch.zeros_like(d_chain) + 2 * self.s_max + 1,
            d_chain,
        )
        a_rel_chain = one_hot(d_chain, 2 * self.s_max + 2)

        cast_dtype = torch.promote_types(self.linear_layer.weight.dtype, torch.float32)
        features_local = torch.cat(
            [
                a_rel_pos.to(cast_dtype),
                a_rel_token.to(cast_dtype),
                b_same_entity.unsqueeze(-1).to(cast_dtype),
                a_rel_chain.to(cast_dtype),
            ],
            dim=-1,
        )

        # Wrap the local result as a row-slab pair-repr DTensor.
        feat_dim = features_local.shape[-1]
        pair_shape = torch.Size([B_global, N_global, N_global, feat_dim])
        pair_stride = update_exhaustive_strides(features_local.shape, features_local.stride(), pair_shape)

        features_dt = DTensor.from_local(
            features_local,
            device_mesh=self.device_mesh,
            placements=PAIR_PLACEMENTS_1D,
            shape=pair_shape,
            stride=pair_stride,
        )

        return self.linear_layer(features_dt)
