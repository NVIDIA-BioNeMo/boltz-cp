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

"""Distributed Pairformer wrappers for 1D CP (2D mesh ``(dp, cp)``).

PairformerLayer1D and PairformerModule1D compose the Phase 1 1D CP layers:
- TriangleMultiplicationOutgoing1D / TriangleMultiplicationIncoming1D
- TriangleAttentionStartingNode1D / TriangleAttentionEndingNode1D
- AttentionPairBias1D
- Transition1D

1D CP placements
----------------
Single repr ``s [B, N, C_s]``    -> ``(Shard(0), Shard(1))``
Pair repr   ``z [B, N, N, C_z]`` -> ``(Shard(0), Shard(1))``
Mask        ``mask [B, N]``       -> ``(Shard(0), Shard(1))``
Pair mask   ``pair_mask [B, N, N]`` -> ``(Shard(0), Shard(1))``

Communication budget per PairformerLayer1D:
    - tri_mul_out: 1 ring round (cp_size steps)
    - tri_mul_in: 1 ring round (cp_size steps)
    - tri_att_start: 1 ring round (cp_size steps, bias rotation)
    - tri_att_end: 1 ring round (cp_size steps, K/V/mask/bias rotation)
    - attention (pair bias): 1 ring round (cp_size steps)
    - transition_s, transition_z: 0 collectives

Column dropout must be broadcast across CP ranks so all ranks apply the
same dropout pattern (required for correctness since column dropout spans
the sharded row dimension).
"""

from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.attention_1d import AttentionPairBias1D
from boltz.distributed.model.layers.elementwise_op import ElementwiseOp, elementwise_op
from boltz.distributed.model.layers.layernorm import LayerNormParamsReplicated
from boltz.distributed.model.layers.transition_1d import Transition1D
from boltz.distributed.model.layers.triangular_attention_1d import (
    TriangleAttentionEndingNode1D,
    TriangleAttentionStartingNode1D,
)
from boltz.distributed.model.layers.triangular_mult_1d import (
    TriangleMultiplicationIncoming1D,
    TriangleMultiplicationOutgoing1D,
)
from boltz.distributed.model.modules.utils import TriAttnBackend, get_cpu_offload_context
from boltz.distributed.utils import update_exhaustive_strides
from boltz.model.layers.pairformer import (
    PairformerLayer as SerialPairformerLayer,
)
from boltz.model.layers.pairformer import (
    PairformerModule as SerialPairformerModule,
)
from boltz.model.layers.pairformer import (
    PairformerNoSeqLayer as SerialPairformerNoSeqLayer,
)
from boltz.model.layers.pairformer import (
    PairformerNoSeqModule as SerialPairformerNoSeqModule,
)


def _get_dropout_mask_1d(
    dropout: float,
    z_local: torch.Tensor,
    training: bool,
    columnwise: bool,
    cp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Generate dropout mask for 1D CP pair representations.

    For rowwise dropout, each rank generates its own mask independently
    (different rows on different ranks).

    For columnwise dropout, the mask must be identical across all CP ranks
    because column dropout spans the sharded row dimension.  Rank 0
    generates the mask and broadcasts it.

    Parameters
    ----------
    dropout : float
        Dropout rate.
    z_local : Tensor
        Local pair representation shard ``[B, N_local, N_full, C_z]``.
    training : bool
        Whether the model is in training mode.
    columnwise : bool
        Whether to apply columnwise dropout.
    cp_group : dist.ProcessGroup
        The CP process group for broadcasting column masks.

    Returns
    -------
    Tensor
        Dropout mask with shape broadcastable to ``z_local``.
    """
    dropout_rate = dropout * training
    if columnwise:
        # Column dropout: shape [B, 1, N_full, 1] — same for all rows
        v = z_local[:, 0:1, :, 0:1]
        d = torch.rand(v.shape, dtype=torch.float32, device=v.device) >= dropout_rate
        # Broadcast across CP ranks so all ranks use the same column mask
        if dist.get_world_size(cp_group) > 1:
            dist.broadcast(d, src=dist.get_process_group_ranks(cp_group)[0], group=cp_group)
    else:
        # Row dropout: shape [B, N_local, 1, 1] — different rows per rank
        v = z_local[:, :, 0:1, 0:1]
        d = torch.rand(v.shape, dtype=torch.float32, device=v.device) >= dropout_rate
    d = d * 1.0 / (1.0 - dropout_rate) if dropout_rate < 1.0 else d
    return d


def _apply_dropout_1d(
    x_dt: DTensor,
    dropout: float,
    training: bool,
    columnwise: bool,
    cp_group: dist.ProcessGroup,
) -> DTensor:
    """Apply dropout to a DTensor pair representation under 1D CP.

    The ``to_local()`` / ``from_local()`` pattern used here is safe for
    gradient flow because both operations are differentiable autograd
    functions (``_ToTorchTensor.apply`` and ``_FromTorchTensor.apply``
    respectively).  The dropout mask is a non-differentiable binary
    tensor, so the backward pass correctly computes
    ``grad_output * mask`` w.r.t. the input.  No custom
    ``autograd.Function`` is needed since no collectives participate in
    the differentiable path — the only collective (column-mask broadcast)
    operates on the non-differentiable mask.

    Parameters
    ----------
    x_dt : DTensor
        Pair representation with placements ``(Shard(0), Shard(1))``.
    dropout : float
        Dropout rate.
    training : bool
        Whether in training mode.
    columnwise : bool
        Whether to apply columnwise dropout.
    cp_group : dist.ProcessGroup
        CP process group for column dropout broadcast.

    Returns
    -------
    DTensor
        Dropout-masked pair representation.
    """
    if not training or dropout == 0.0:
        return x_dt

    x_local = x_dt.to_local()
    mask = _get_dropout_mask_1d(dropout, x_local, training, columnwise, cp_group)
    out_local = (x_local * mask.to(x_local.dtype)).contiguous()

    out_stride = update_exhaustive_strides(out_local.shape, out_local.stride(), x_dt.shape)
    return DTensor.from_local(
        out_local,
        x_dt.device_mesh,
        x_dt.placements,
        shape=x_dt.shape,
        stride=out_stride,
    )


class PairformerLayer1D(nn.Module):
    """Distributed PairformerLayer for 1D CP (2D mesh ``(dp, cp)``).

    Wraps a serial PairformerLayer or PairformerNoSeqLayer, replacing each
    sub-module with its 1D CP counterpart.  When wrapping a
    PairformerNoSeqLayer, the sequence track (attention + transition_s) is
    omitted.

    .. note::
        This implementation only supports V2 (``v2=True``) serial layers.
        V1 features (``apply_initial_norm``, ``use_model_cache``) are not
        supported. Passing a V1 layer raises ``TypeError``.

    Parameters
    ----------
    layer : SerialPairformerLayer or SerialPairformerNoSeqLayer
        The serial Pairformer layer to distribute.
    dist_manager : DistributedManager
        Distributed manager defining the distributed computation topology
        and groups.
    """

    def __init__(
        self,
        layer: Union[SerialPairformerLayer, SerialPairformerNoSeqLayer],
        dist_manager: DistributedManager,
    ) -> None:
        if not isinstance(layer, (SerialPairformerLayer, SerialPairformerNoSeqLayer)):
            raise TypeError(
                f"layer must be SerialPairformerLayer or SerialPairformerNoSeqLayer, " f"got {type(layer).__name__}"
            )
        super().__init__()
        self.dist_manager = dist_manager
        self.device_mesh = dist_manager.device_mesh
        self.cp_group = dist_manager.group["cp"]
        device_mesh = self.device_mesh
        cp_group = self.cp_group
        self.no_seq = isinstance(layer, SerialPairformerNoSeqLayer)

        self.token_z = layer.token_z
        self.dropout = layer.dropout
        self.post_layer_norm = layer.post_layer_norm

        # Mutable backend selection for triangle attention layers.
        # Default is REFERENCE (no fused kernels). To switch backend for the
        # entire model, use ``model.apply(SetTriAttnBackend(backend))``
        # (see boltz.distributed.model.modules.utils.SetTriAttnBackend).
        self.triattn_backend = TriAttnBackend.REFERENCE

        # Pairwise stack
        self.tri_mul_out = TriangleMultiplicationOutgoing1D(layer.tri_mul_out, device_mesh, cp_group)
        self.tri_mul_in = TriangleMultiplicationIncoming1D(layer.tri_mul_in, device_mesh, cp_group)
        self.tri_att_start = TriangleAttentionStartingNode1D(layer.tri_att_start, device_mesh, cp_group)
        self.tri_att_end = TriangleAttentionEndingNode1D(layer.tri_att_end, device_mesh, cp_group)
        self.transition_z = Transition1D(layer.transition_z, device_mesh)

        # Sequence stack (only for full PairformerLayer)
        if not self.no_seq:
            from boltz.model.layers.attentionv2 import AttentionPairBias as SerialAttentionPairBiasV2

            if not isinstance(layer.attention, SerialAttentionPairBiasV2):
                raise TypeError(
                    f"PairformerLayer1D only supports V2 attention (v2=True). "
                    f"Got attention type {type(layer.attention).__name__}"
                )
            self.num_heads = layer.num_heads
            self.pre_norm_s = LayerNormParamsReplicated(layer.pre_norm_s, device_mesh)
            self.attention = AttentionPairBias1D(layer.attention, device_mesh, cp_group)
            self.transition_s = Transition1D(layer.transition_s, device_mesh)
            if self.post_layer_norm:
                self.s_post_norm = LayerNormParamsReplicated(layer.s_post_norm, device_mesh)
            else:
                self.s_post_norm = None

    def forward(
        self,
        s: Optional[DTensor] = None,
        z: Optional[DTensor] = None,
        mask: Optional[DTensor] = None,
        pair_mask: Optional[DTensor] = None,
    ) -> Union[Tuple[DTensor, DTensor], DTensor]:
        """Forward pass.

        Pass ``s`` and ``mask`` for the full pairformer; omit them for
        pair-only mode.

        Parameters
        ----------
        s : DTensor, optional
            ``[B, N, C_s]`` single repr with placements ``(Shard(0), Shard(1))``.
        z : DTensor
            ``[B, N, N, C_z]`` pair repr with placements ``(Shard(0), Shard(1))``.
        mask : DTensor, optional
            ``[B, N]`` token mask with placements ``(Shard(0), Shard(1))``.
        pair_mask : DTensor
            ``[B, N, N]`` pair mask with placements ``(Shard(0), Shard(1))``.

        Returns
        -------
        DTensor or tuple[DTensor, DTensor]
            Updated ``z`` (pair-only) or ``(s, z)`` (full pairformer).
        """
        assert z is not None and pair_mask is not None

        # Pairwise stack
        z = elementwise_op(
            z,
            _apply_dropout_1d(
                self.tri_mul_out(z, mask=pair_mask),
                self.dropout,
                self.training,
                False,
                self.cp_group,
            ),
            ElementwiseOp.SUM,
        )
        z = elementwise_op(
            z,
            _apply_dropout_1d(
                self.tri_mul_in(z, mask=pair_mask),
                self.dropout,
                self.training,
                False,
                self.cp_group,
            ),
            ElementwiseOp.SUM,
        )
        z = elementwise_op(
            z,
            _apply_dropout_1d(
                self.tri_att_start(z, mask=pair_mask, triattn_backend=self.triattn_backend),
                self.dropout,
                self.training,
                False,
                self.cp_group,
            ),
            ElementwiseOp.SUM,
        )
        z = elementwise_op(
            z,
            _apply_dropout_1d(
                self.tri_att_end(z, mask=pair_mask, triattn_backend=self.triattn_backend),
                self.dropout,
                self.training,
                True,
                self.cp_group,  # columnwise
            ),
            ElementwiseOp.SUM,
        )
        z = elementwise_op(z, self.transition_z(z), ElementwiseOp.SUM)

        if s is None:
            return z

        # Sequence stack
        assert mask is not None
        with torch.autocast("cuda", enabled=False):
            safe_dtype = torch.promote_types(s.dtype, torch.float32)
            s = s.to(dtype=safe_dtype)
            s_normed = self.pre_norm_s(s)
            s = elementwise_op(
                s,
                self.attention(
                    s=s_normed,
                    z=z.to(dtype=safe_dtype),
                    mask=mask.to(dtype=safe_dtype),
                    k_in=s_normed,
                ),
                ElementwiseOp.SUM,
            )
            s = elementwise_op(s, self.transition_s(s), ElementwiseOp.SUM)
            s = self.s_post_norm(s) if self.s_post_norm is not None else s

        return s, z


class PairformerModule1D(nn.Module):
    """Distributed PairformerModule for 1D CP (2D mesh ``(dp, cp)``).

    Stack of PairformerLayer1D blocks with optional activation checkpointing.
    Handles both full pairformer (sequence + pairwise stacks) and pair-only
    mode, inferred from the serial module type.

    .. note::
        This implementation only supports V2 (``v2=True``) serial layers.
        V1 features (``apply_initial_norm``, ``use_model_cache``) are not
        supported. Passing a V1 layer raises ``TypeError`` at the
        ``PairformerLayer1D`` level.

    Parameters
    ----------
    module : SerialPairformerModule or SerialPairformerNoSeqModule
        The serial Pairformer module to distribute.
    dist_manager : DistributedManager
        Distributed manager defining the distributed computation topology
        and groups.
    cpu_offloading : bool
        Whether to offload activations to CPU during checkpointing.
    """

    def __init__(
        self,
        module: Union[SerialPairformerModule, SerialPairformerNoSeqModule],
        dist_manager: DistributedManager,
        cpu_offloading: bool = False,
    ) -> None:
        super().__init__()
        self.dist_manager = dist_manager
        self.device_mesh = dist_manager.device_mesh
        self.cp_group = dist_manager.group["cp"]

        no_seq = isinstance(module, SerialPairformerNoSeqModule)
        self.no_seq = no_seq

        self.token_z = module.token_z
        self.num_blocks = module.num_blocks
        self.dropout = module.dropout
        self.post_layer_norm = module.post_layer_norm
        self.activation_checkpointing = module.activation_checkpointing
        self.cpu_offloading = cpu_offloading

        if not no_seq:
            self.num_heads = module.num_heads

        self.layers = nn.ModuleList()
        for serial_layer in module.layers:
            self.layers.append(PairformerLayer1D(serial_layer, dist_manager))

    def forward(
        self,
        s: Optional[DTensor] = None,
        z: Optional[DTensor] = None,
        mask: Optional[DTensor] = None,
        pair_mask: Optional[DTensor] = None,
    ) -> Union[Tuple[DTensor, DTensor], DTensor]:
        """Forward pass.

        Pass ``s`` and ``mask`` for the full pairformer; omit them for
        pair-only mode.

        Parameters
        ----------
        s : DTensor, optional
            ``[B, N, C_s]`` single repr.
        z : DTensor
            ``[B, N, N, C_z]`` pair repr.
        mask : DTensor, optional
            ``[B, N]`` token mask.
        pair_mask : DTensor
            ``[B, N, N]`` pair mask.

        Returns
        -------
        DTensor or tuple[DTensor, DTensor]
            Updated ``z`` (pair-only) or ``(s, z)`` (full pairformer).
        """
        if self.activation_checkpointing and self.training:
            if self.cpu_offloading:
                with get_cpu_offload_context(optimized=True):
                    for layer in self.layers:
                        result = checkpoint(
                            layer,
                            s,
                            z,
                            mask,
                            pair_mask,
                            use_reentrant=False,
                        )
                        if self.no_seq:
                            z = result
                        else:
                            s, z = result
            else:
                for layer in self.layers:
                    result = checkpoint(
                        layer,
                        s,
                        z,
                        mask,
                        pair_mask,
                        use_reentrant=False,
                    )
                    if self.no_seq:
                        z = result
                    else:
                        s, z = result
        else:
            for layer in self.layers:
                result = layer(s=s, z=z, mask=mask, pair_mask=pair_mask)
                if self.no_seq:
                    z = result
                else:
                    s, z = result
        return z if self.no_seq else (s, z)


class PairformerNoSeqLayer1D(PairformerLayer1D):
    """Distributed PairformerNoSeqLayer for 1D CP (pairwise stack only)."""

    pass


class PairformerNoSeqModule1D(PairformerModule1D):
    """Distributed PairformerNoSeqModule for 1D CP (pairwise stack only)."""

    pass
