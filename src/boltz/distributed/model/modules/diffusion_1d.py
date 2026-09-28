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

"""1D CP DiffusionModule and AtomDiffusion for 2D mesh ``(dp, cp)``.

Boltz-2 only (externalized conditioning).

Tensor placements on 2D mesh ``(dp, cp)``:

- Single ``s [B, N, C_s]``: ``(Shard(0), Shard(1))``
- Pair / bias ``z [B, N, N, H]``: ``(Shard(0), Shard(1))`` -- row-slab
- Atom ``r [B*M, N_atoms, 3]``: ``(Shard(0), Shard(1))``
- Times ``times [B*M]``: ``(Shard(0), Replicate())``

The token-level DiffusionTransformer uses ``AttentionPairBias1D`` ring
attention (K/V rotation on cp, pair bias column slicing with zero
communication). The atom-level encoder/decoder uses window-batched
attention (unchanged from 2D CP).

Communication per DiffusionTransformerLayer1D:
- Ring attention: cp_size ring steps rotating K, V, mask.
- Bias: 0 communication (column slicing on row-slab pair bias).
"""

import warnings
from math import exp, sqrt

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, full, zeros
from torch.nn import ModuleList
from torch.utils.checkpoint import checkpoint

from boltz.distributed.data.feature.featurizer import pack_atom_features
from boltz.distributed.model.layers.attention_1d import AttentionPairBias1D
from boltz.distributed.model.layers.cat_and_chunk import shardwise_chunk
from boltz.distributed.model.layers.elementwise_op import (
    ElementwiseOp,
    elementwise_op,
    scalar_tensor_op,
    single_tensor_op,
)
from boltz.distributed.model.layers.layernorm import LayerNormParamsReplicated
from boltz.distributed.model.layers.linear import LinearParamsReplicated
from boltz.distributed.model.layers.repeat_interleave import shardwise_repeat_interleave
from boltz.distributed.model.layers.replicate_op import ReplicateOp, replicate_op
from boltz.distributed.model.layers.sharded_op import sharded_sum
from boltz.distributed.model.layers.shardwise_op import shardwise_sum
from boltz.distributed.model.layers.sigmoid_gate import sigmoid_gate
from boltz.distributed.model.layers.squeeze import shardwise_unsqueeze
from boltz.distributed.model.layers.utils import distributed_pack_and_pad, distributed_unpad_and_unpack
from boltz.distributed.model.modules.encoders import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
)
from boltz.distributed.model.modules.encoders_1d import SingleConditioning1D
from boltz.distributed.model.modules.transformers import (
    AdaLN,
    ConditionedTransitionBlock,
)
from boltz.distributed.model.modules.utils import (
    extract_checkpointing_config,
    get_cpu_offload_context,
)
from boltz.distributed.utils import (
    LayoutRightMap,
    all_reduce_weighted_mean,
    create_and_broadcast_tensor_into_placements,
    create_distributed_randn,
)
from boltz.model.modules.diffusionv2 import AtomDiffusion as SerialAtomDiffusionV2
from boltz.model.modules.diffusionv2 import DiffusionModule as SerialDiffusionModuleV2
from boltz.model.modules.transformersv2 import DiffusionTransformer as SerialDiffusionTransformerV2
from boltz.model.modules.transformersv2 import DiffusionTransformerLayer as SerialDiffusionTransformerLayerV2
from boltz.model.modules.utils import default, random_rotations

# 1D CP placements on 2D mesh (dp, cp)
SINGLE_PLACEMENTS_1D = (Shard(0), Shard(1))
TIMES_PLACEMENTS_1D = (Shard(0), Replicate())


@torch.no_grad()
def _center_random_augmentation_1d(
    atom_coords: DTensor,
    atom_mask: DTensor,
    s_trans: float = 1.0,
    augmentation: bool = True,
    centering: bool = True,
    return_second_coords: bool = False,
    second_coords: DTensor | None = None,
    return_roto: bool = False,
) -> tuple[DTensor, ...] | DTensor:
    """Center and randomly augment coordinates for 1D CP on 2D mesh ``(dp, cp)``.

    Like the 2D CP version but uses 2-element placements ``(Shard(0), Shard(1))``
    and a single CP group instead of cp_axis_0/cp_axis_1.

    Parameters
    ----------
    atom_coords : DTensor
        ``[B, N, 3]`` with placements ``(Shard(0), Shard(1))``.
    atom_mask : DTensor
        ``[B, N]`` with placements ``(Shard(0), Shard(1))``.
    s_trans : float
        Translation scale factor.
    augmentation : bool
        Whether to add random rotation + translation.
    centering : bool
        Whether to center coordinates to zero mean.
    return_second_coords : bool
        Whether to return transformed second coordinates.
    second_coords : DTensor or None
        Second coordinates to apply the same transformation.
    return_roto : bool
        Whether to return the rotation matrix.

    Returns
    -------
    DTensor or tuple[DTensor, ...]
        Augmented coordinates, and optionally second coords and rotation matrix.
    """
    if return_roto and not augmentation:
        raise ValueError("cannot return rotation matrix when augmentation is False")

    device_mesh = atom_coords.device_mesh
    input_placements = atom_coords.placements

    if input_placements != SINGLE_PLACEMENTS_1D:
        raise ValueError(f"Expected placements {SINGLE_PLACEMENTS_1D}, got {input_placements}")

    cp_group = device_mesh.get_group(1)  # cp axis on 2D mesh (dp, cp)

    if centering:
        atom_coords_local = atom_coords.to_local()
        atom_mask_local = atom_mask.to_local()

        atom_mean_local = all_reduce_weighted_mean(
            atom_mask_local.unsqueeze(-1),
            atom_coords_local,
            group_reduce=cp_group,
            dim=1,
        )

        # atom_mean_local is [B, 3], broadcast-subtract from coords
        shape_atom_mean_global = (atom_coords.shape[0], atom_coords.shape[-1])
        stride_atom_mean_global = LayoutRightMap(shape_atom_mean_global).strides
        atom_mean = DTensor.from_local(
            atom_mean_local,
            device_mesh=device_mesh,
            placements=(Shard(0), Replicate()),
            shape=shape_atom_mean_global,
            stride=stride_atom_mean_global,
        )

        atom_coords = replicate_op(atom_coords, atom_mean, 1, ReplicateOp.SUB)
        if second_coords is not None:
            second_coords = replicate_op(second_coords, atom_mean, 1, ReplicateOp.SUB)

    roto = None
    if augmentation:
        size_batch = atom_coords.shape[0]

        def create_rand_rot_fn(shape_local, dtype, device):
            return random_rotations(shape_local[0], dtype=dtype, device=device)

        R_local = create_and_broadcast_tensor_into_placements(
            shape=(size_batch, 3, 3),
            create_local_fn=create_rand_rot_fn,
            device_mesh=device_mesh,
            placements=(Shard(0), Replicate()),
            dtype=atom_coords.to_local().dtype,
        )

        # Apply rotation
        coords_local = atom_coords.to_local()
        coords_rotated_local = torch.einsum("bmd,bds->bms", coords_local, R_local)
        atom_coords = DTensor.from_local(
            coords_rotated_local,
            device_mesh=device_mesh,
            placements=input_placements,
            shape=atom_coords.shape,
            stride=atom_coords.stride(),
        )

        if second_coords is not None:
            sc_local = second_coords.to_local()
            sc_rotated_local = torch.einsum("bmd,bds->bms", sc_local, R_local)
            second_coords = DTensor.from_local(
                sc_rotated_local,
                device_mesh=device_mesh,
                placements=input_placements,
                shape=second_coords.shape,
                stride=second_coords.stride(),
            )

        if return_roto:
            shape_roto = (size_batch, 3, 3)
            stride_roto = LayoutRightMap(shape_roto).strides
            roto = DTensor.from_local(
                R_local,
                device_mesh,
                (Shard(0), Replicate()),
                shape=shape_roto,
                stride=stride_roto,
            )

        # Random translation
        random_trans = create_distributed_randn(
            (size_batch, 1, 3),
            device_mesh=device_mesh,
            placements=(Shard(0), Replicate()),
            dtype=atom_coords.to_local().dtype,
            scale=s_trans,
        )

        coords_local = atom_coords.to_local()
        random_trans_local = random_trans.to_local()
        atom_coords = DTensor.from_local(
            coords_local + random_trans_local,
            device_mesh=device_mesh,
            placements=input_placements,
            shape=atom_coords.shape,
            stride=atom_coords.stride(),
        )

        if second_coords is not None:
            sc_local = second_coords.to_local()
            second_coords = DTensor.from_local(
                sc_local + random_trans_local,
                device_mesh=device_mesh,
                placements=input_placements,
                shape=second_coords.shape,
                stride=second_coords.stride(),
            )

    if return_second_coords and return_roto:
        return atom_coords, second_coords, roto
    elif return_second_coords:
        return atom_coords, second_coords
    elif return_roto:
        return atom_coords, roto
    else:
        return atom_coords


class DiffusionTransformerLayer1D(nn.Module):
    """DiffusionTransformerLayer for 1D CP on 2D mesh ``(dp, cp)``.

    Uses ``AttentionPairBias1D`` for ring attention over the token dimension.
    Pair bias is in row-slab layout ``[B, N/cp, N, H]`` -- column slicing at
    each ring step, no bias communication.

    Communication budget per layer:
    - Forward:  cp_size ring steps (K, V, mask).
    - Backward: cp_size ring steps + dK/dV return.
    - Bias: 0 communication.
    """

    def __init__(
        self,
        layer: SerialDiffusionTransformerLayerV2,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        layer : SerialDiffusionTransformerLayerV2
            Serial Boltz-2 DiffusionTransformerLayer.
        device_mesh : DeviceMesh
            2D device mesh ``(dp, cp)``.
        cp_group : dist.ProcessGroup
            The CP process group for ring communication.
        """
        super().__init__()
        if not isinstance(layer, SerialDiffusionTransformerLayerV2):
            raise TypeError(
                f"Expected SerialDiffusionTransformerLayerV2, got {type(layer)}. "
                "DiffusionTransformerLayer1D only supports Boltz-2."
            )

        self.adaln = AdaLN(layer.adaln, device_mesh)

        # AttentionPairBias1D for ring attention with row-slab bias
        serial_attn = layer.pair_bias_attn
        self.pair_bias_attn = AttentionPairBias1D(
            layer=serial_attn,
            device_mesh=device_mesh,
            cp_group=cp_group,
        )

        self.output_projection_linear = LinearParamsReplicated(layer.output_projection_linear, device_mesh)
        self.output_projection = nn.Sequential(self.output_projection_linear)

        self.transition = ConditionedTransitionBlock(layer.transition, device_mesh)

        # Post layer norm (Boltz-2 only)
        self.post_lnorm = None
        if hasattr(layer, "post_lnorm") and not isinstance(layer.post_lnorm, nn.Identity):
            self.post_lnorm = LayerNormParamsReplicated(layer.post_lnorm, device_mesh)

    def forward(
        self,
        a: DTensor,
        s: DTensor,
        z: DTensor,
        mask: DTensor,
        multiplicity: int = 1,
    ) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        a : DTensor
            ``[B*M, N, 2*token_s]`` with placements ``(Shard(0), Shard(1))``.
        s : DTensor
            ``[B*M, N, 2*token_s]`` with placements ``(Shard(0), Shard(1))``.
        z : DTensor
            ``[B, N, N, H]`` pre-computed pair bias (per-layer chunk).
            Placements ``(Shard(0), Shard(1))`` -- row-slab.
        mask : DTensor
            ``[B, N]`` token mask with placements ``(Shard(0), Shard(1))``.
        multiplicity : int
            Number of diffusion samples per batch element.

        Returns
        -------
        DTensor
            Same shape and placements as ``a``.
        """
        b: DTensor = self.adaln(a, s)

        # Expand mask and z for multiplicity if needed
        mask_mult = shardwise_repeat_interleave(mask, multiplicity, 0) if multiplicity > 1 else mask
        z_mult = shardwise_repeat_interleave(z, multiplicity, 0) if multiplicity > 1 else z

        b = self.pair_bias_attn(s=b, z=z_mult, mask=mask_mult)

        b = sigmoid_gate(g=self.output_projection[0](s), x=b)

        a = elementwise_op(a, b, op=ElementwiseOp.SUM)
        c = self.transition(a, s)
        a = elementwise_op(a, c, op=ElementwiseOp.SUM)

        if self.post_lnorm is not None:
            a = self.post_lnorm(a)

        return a


class DiffusionTransformer1D(nn.Module):
    """Multi-layer DiffusionTransformer for 1D CP on 2D mesh ``(dp, cp)``.

    Boltz-2 only. Splits pre-computed bias across layers (last dim = num_heads * L).
    Uses ``AttentionPairBias1D`` ring attention per layer.
    """

    def __init__(
        self,
        layer: SerialDiffusionTransformerV2,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        layer : SerialDiffusionTransformerV2
            Serial Boltz-2 DiffusionTransformer.
        device_mesh : DeviceMesh
            2D device mesh ``(dp, cp)``.
        cp_group : dist.ProcessGroup
            The CP process group for ring communication.
        """
        super().__init__()
        if not isinstance(layer, SerialDiffusionTransformerV2):
            raise TypeError(
                f"Expected SerialDiffusionTransformerV2, got {type(layer)}. "
                "DiffusionTransformer1D only supports Boltz-2."
            )

        if not getattr(layer, "pair_bias_attn", True):
            raise NotImplementedError(
                "DiffusionTransformer1D does not support pair_bias_attn=False "
                "(dead code in the serial Boltz-2 implementation)."
            )

        # Detect activation checkpointing
        activation_checkpointing = set()
        cpu_offloading = set()
        for serial_layer in layer.layers:
            has_ckpt, has_offload = extract_checkpointing_config(serial_layer)
            activation_checkpointing.add(has_ckpt)
            cpu_offloading.add(has_offload)

        if len(activation_checkpointing) > 1:
            raise ValueError(
                "All layers must have the same activation checkpointing config: " f"{activation_checkpointing}"
            )
        if len(cpu_offloading) > 1:
            raise ValueError("All layers must have the same CPU offloading config: " f"{cpu_offloading}")

        layer_level_ckpt = activation_checkpointing.pop() if activation_checkpointing else False
        parent_level_ckpt = getattr(layer, "activation_checkpointing", False)
        self.activation_checkpointing = layer_level_ckpt or parent_level_ckpt
        self.cpu_offloading = cpu_offloading.pop() if cpu_offloading else False

        self.layers = ModuleList(
            [DiffusionTransformerLayer1D(serial_layer, device_mesh, cp_group) for serial_layer in layer.layers]
        )

    def forward(
        self,
        a: DTensor,
        s: DTensor,
        z: DTensor,
        mask: DTensor,
        multiplicity: int = 1,
    ) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        a : DTensor
            ``[B*M, N, 2*token_s]`` with placements ``(Shard(0), Shard(1))``.
        s : DTensor
            ``[B*M, N, 2*token_s]`` single conditioning.
        z : DTensor
            ``[B, N, N, H*L]`` pre-computed bias (all layers concatenated).
            Placements ``(Shard(0), Shard(1))`` -- row-slab.
        mask : DTensor
            ``[B, N]`` token mask.
        multiplicity : int
            Number of diffusion samples per batch element.

        Returns
        -------
        DTensor
            Same shape and placements as ``a``.
        """
        L = len(self.layers)
        if L > 1:
            if z.shape[-1] % L != 0:
                raise ValueError(
                    f"Bias last dimension ({z.shape[-1]}) must be divisible by " f"the number of layers ({L})."
                )
            z_chunks = shardwise_chunk(z, chunks=L, dim=-1)
        else:
            z_chunks = None

        for i, layer in enumerate(self.layers):
            z_i = z_chunks[i] if z_chunks is not None else z

            if self.activation_checkpointing and self.training:
                if self.cpu_offloading:
                    with get_cpu_offload_context(optimized=True):
                        a = checkpoint(
                            layer,
                            a,
                            s,
                            z_i,
                            mask,
                            multiplicity,
                            use_reentrant=False,
                        )
                else:
                    a = checkpoint(
                        layer,
                        a,
                        s,
                        z_i,
                        mask,
                        multiplicity,
                        use_reentrant=False,
                    )
            else:
                a = layer(a, s, z_i, mask=mask, multiplicity=multiplicity)
        return a


class DiffusionModule1D(nn.Module):
    """1D CP DiffusionModule for Boltz-2 on 2D mesh ``(dp, cp)``.

    Atom features (``feats``) and ``r_noisy`` are expected in **unpacked** layout
    (with intersperse padding from the CP DTensor data loader). The module calls
    ``pack_atom_features`` and ``distributed_pack_and_pad`` internally.

    The ``diffusion_conditioning`` dict arrives in packed layout as an inter-layer
    output from ``DiffusionConditioning``.

    The token-level ``DiffusionTransformer1D`` uses ring attention
    (``AttentionPairBias1D``). The atom-level encoder/decoder uses window-batched
    attention (mesh-size agnostic).
    """

    def __init__(
        self,
        layer: SerialDiffusionModuleV2,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        layer : SerialDiffusionModuleV2
            Serial Boltz-2 DiffusionModule.
        device_mesh : DeviceMesh
            2D device mesh ``(dp, cp)``.
        cp_group : dist.ProcessGroup
            The CP process group for ring communication.
        """
        super().__init__()
        if not isinstance(layer, SerialDiffusionModuleV2):
            raise TypeError(f"Expected SerialDiffusionModuleV2, got {type(layer)}")

        warnings.warn(
            "CPU offloading-based activation checkpointing by default is "
            "not used for Boltz-2 so we do not use it in 1D CP DiffusionModule.",
            UserWarning,
            stacklevel=2,
        )

        self.device_mesh = device_mesh
        self.sigma_data = layer.sigma_data
        self.atoms_per_window_queries = layer.atoms_per_window_queries
        self.atoms_per_window_keys = layer.atoms_per_window_keys
        self.activation_checkpointing = getattr(layer, "activation_checkpointing", False)

        # Sub-modules (attribute names mirror serial)
        self.single_conditioner = SingleConditioning1D(layer.single_conditioner, device_mesh)
        self.atom_attention_encoder = AtomAttentionEncoder(layer.atom_attention_encoder, device_mesh)
        self.s_to_a_linear = nn.Sequential(
            LayerNormParamsReplicated(layer.s_to_a_linear[0], device_mesh),
            LinearParamsReplicated(layer.s_to_a_linear[1], device_mesh),
        )
        self.token_transformer = DiffusionTransformer1D(layer.token_transformer, device_mesh, cp_group)
        self.a_norm = LayerNormParamsReplicated(layer.a_norm, device_mesh)
        self.atom_attention_decoder = AtomAttentionDecoder(layer.atom_attention_decoder, device_mesh)

    def forward(
        self,
        s_inputs: DTensor,
        s_trunk: DTensor,
        r_noisy: DTensor,
        times: DTensor,
        feats: dict[str, DTensor],
        diffusion_conditioning: dict[str, DTensor],
        multiplicity: int = 1,
    ) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        s_inputs : DTensor
            ``[B, N, token_s]`` with placements ``(Shard(0), Shard(1))``.
        s_trunk : DTensor
            ``[B, N, token_s]`` with placements ``(Shard(0), Shard(1))``.
        r_noisy : DTensor
            ``[B*M, N_atoms, 3]`` with placements ``(Shard(0), Shard(1))``.
        times : DTensor
            ``[B*M]`` with placements ``(Shard(0), Replicate())``.
        feats : dict[str, DTensor]
            Unpacked atom features.
        diffusion_conditioning : dict[str, DTensor]
            Pre-computed conditioning from ``DiffusionConditioning`` in packed layout.
        multiplicity : int
            Number of diffusion samples per batch element.

        Returns
        -------
        DTensor
            ``r_update`` with same shape/placements as ``r_noisy``.
        """
        # Placement checks
        expected_single = SINGLE_PLACEMENTS_1D
        expected_times = TIMES_PLACEMENTS_1D
        for name, dt, expected in [
            ("s_inputs", s_inputs, expected_single),
            ("s_trunk", s_trunk, expected_single),
            ("r_noisy", r_noisy, expected_single),
        ]:
            if dt.placements != expected:
                raise ValueError(f"{name} has placements {dt.placements}, expected {expected}")
        if times.placements != expected_times:
            raise ValueError(f"times has placements {times.placements}, expected {expected_times}")

        # Shape checks
        if s_inputs.shape[0] != feats["token_pad_mask"].shape[0]:
            raise ValueError(
                f"s_inputs batch {s_inputs.shape[0]} != feats['token_pad_mask'] batch "
                f"{feats['token_pad_mask'].shape[0]} (should not include multiplicity)"
            )
        if r_noisy.shape[0] != feats["atom_pad_mask"].shape[0] * multiplicity:
            raise ValueError(
                f"r_noisy batch {r_noisy.shape[0]} != atom_pad_mask batch "
                f"{feats['atom_pad_mask'].shape[0]} * multiplicity {multiplicity}"
            )

        # 1. Single conditioning
        s_trunk_mult = shardwise_repeat_interleave(s_trunk, multiplicity, 0)
        s_inputs_mult = shardwise_repeat_interleave(s_inputs, multiplicity, 0)
        if self.activation_checkpointing and self.training:
            s, normed_fourier = checkpoint(
                self.single_conditioner,
                times,
                s_trunk_mult,
                s_inputs_mult,
                use_reentrant=False,
            )
        else:
            s, normed_fourier = self.single_conditioner(times, s_trunk_mult, s_inputs_mult)

        compute_dtype = torch.promote_types(r_noisy.dtype, torch.float32)

        # 2. Pack atom features and r_noisy
        W = self.atoms_per_window_queries
        _keys_atom_features_packed = {
            "atom_pad_mask",
            "ref_pos",
            "ref_space_uid",
            "ref_charge",
            "ref_element",
            "ref_atom_name_chars",
            "atom_to_token",
        }
        feats_packed = pack_atom_features(feats, _keys_atom_features_packed, W)

        atom_mask_mul = shardwise_repeat_interleave(feats["atom_pad_mask"].bool(), multiplicity, 0)
        atom_mask_mul_expanded = shardwise_unsqueeze(atom_mask_mul, dim=-1)
        r_noisy_packed, atom_mask_r_noisy_packed = distributed_pack_and_pad(
            r_noisy,
            atom_mask_mul_expanded,
            W,
            axis=1,
        )

        # 3. Atom attention encoder (window-batched)
        a, q_skip, c_skip, p_skip = self.atom_attention_encoder(
            feats=feats_packed,
            q=diffusion_conditioning["q"].to(compute_dtype),
            c=diffusion_conditioning["c"].to(compute_dtype),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].to(compute_dtype),
            r=r_noisy_packed,
            multiplicity=multiplicity,
        )

        # 4. Token-level transformer (ring attention)
        a = elementwise_op(a, self.s_to_a_linear(s), ElementwiseOp.SUM)
        mask = feats["token_pad_mask"]
        a = self.token_transformer(
            a,
            mask=mask.to(compute_dtype),
            s=s,
            z=diffusion_conditioning["token_trans_bias"].to(compute_dtype),
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)

        # 5. Atom attention decoder (window-batched)
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            p=diffusion_conditioning["atom_dec_bias"].to(compute_dtype),
            feats=feats_packed,
            multiplicity=multiplicity,
        )

        # 6. Unpack
        r_update = distributed_unpad_and_unpack(
            r_update,
            atom_mask_r_noisy_packed,
            atom_mask_mul_expanded,
            axis=1,
            keep_input_padding=False,
        )
        return r_update


class AtomDiffusion1D(nn.Module):
    """1D CP AtomDiffusion for Boltz-2 on 2D mesh ``(dp, cp)``.

    Wraps ``DiffusionModule1D`` with diffusion scheduling (noise preconditioning,
    training forward, and sampling). Scalar diffusion math is identical to the
    serial and 2D CP versions.

    Atom features and atom-level tensors are in **unpacked** layout. The
    ``diffusion_conditioning`` dict arrives in packed layout from
    ``DiffusionConditioning``.
    """

    def __init__(
        self,
        layer: SerialAtomDiffusionV2,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        layer : SerialAtomDiffusionV2
            Serial Boltz-2 AtomDiffusion.
        device_mesh : DeviceMesh
            2D device mesh ``(dp, cp)``.
        cp_group : dist.ProcessGroup
            The CP process group for ring communication.
        """
        super().__init__()
        if not isinstance(layer, SerialAtomDiffusionV2):
            raise TypeError(f"Expected SerialAtomDiffusionV2, got {type(layer)}")

        self.device_mesh = device_mesh

        # Copy diffusion parameters
        self.sigma_min = layer.sigma_min
        self.sigma_max = layer.sigma_max
        self.sigma_data = layer.sigma_data
        self.rho = layer.rho
        self.P_mean = layer.P_mean
        self.P_std = layer.P_std
        self.num_sampling_steps = layer.num_sampling_steps
        self.gamma_0 = layer.gamma_0
        self.gamma_min = layer.gamma_min
        self.noise_scale = layer.noise_scale
        self.step_scale = layer.step_scale
        self.coordinate_augmentation = layer.coordinate_augmentation
        self.alignment_reverse_diff = layer.alignment_reverse_diff
        self.synchronize_sigmas = layer.synchronize_sigmas
        self.token_s = layer.token_s

        self.score_model = DiffusionModule1D(layer.score_model, device_mesh, cp_group)

    @property
    def device(self):
        """Get the device type of the model."""
        return self.device_mesh.device_type

    # ------------------------------------------------------------------
    # Diffusion preconditioning (DTensor scalar ops)
    # ------------------------------------------------------------------

    def _check_sigma_placement(self, sigma: DTensor) -> None:
        """Validate sigma placements: ``(Shard(0), Replicate())``."""
        expected = TIMES_PLACEMENTS_1D
        if sigma.placements != expected:
            raise ValueError(f"Sigma has placements {sigma.placements}, expected {expected}")

    def c_skip(self, sigma: DTensor) -> DTensor:
        """Skip-connection scaling: sigma_data^2 / (sigma^2 + sigma_data^2)."""
        self._check_sigma_placement(sigma)
        sigma_sq = scalar_tensor_op(2, sigma, ElementwiseOp.POW)
        denom = scalar_tensor_op(self.sigma_data**2, sigma_sq, ElementwiseOp.SUM)
        return scalar_tensor_op(self.sigma_data**2, denom, ElementwiseOp.DIV)

    def c_out(self, sigma: DTensor) -> DTensor:
        """Output scaling: sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)."""
        self._check_sigma_placement(sigma)
        numer = scalar_tensor_op(self.sigma_data, sigma, ElementwiseOp.PROD)
        sigma_sq = scalar_tensor_op(2, sigma, ElementwiseOp.POW)
        denom = scalar_tensor_op(self.sigma_data**2, sigma_sq, ElementwiseOp.SUM)
        denom = scalar_tensor_op(0.5, denom, ElementwiseOp.POW)
        return elementwise_op(numer, denom, ElementwiseOp.DIV)

    def c_in(self, sigma: DTensor) -> DTensor:
        """Input scaling: 1 / sqrt(sigma^2 + sigma_data^2)."""
        self._check_sigma_placement(sigma)
        sigma_sq = scalar_tensor_op(2, sigma, ElementwiseOp.POW)
        denom = scalar_tensor_op(self.sigma_data**2, sigma_sq, ElementwiseOp.SUM)
        denom = scalar_tensor_op(0.5, denom, ElementwiseOp.POW)
        return scalar_tensor_op(1, denom, ElementwiseOp.DIV)

    def c_noise(self, sigma: DTensor) -> DTensor:
        """Noise conditioning: log(sigma / sigma_data) * 0.25."""
        self._check_sigma_placement(sigma)
        scaled = scalar_tensor_op(1 / self.sigma_data, sigma, ElementwiseOp.PROD)
        scaled_local = scaled.to_local().clamp(min=1e-20)
        scaled = DTensor.from_local(
            scaled_local,
            device_mesh=scaled.device_mesh,
            placements=scaled.placements,
            shape=scaled.shape,
            stride=scaled.stride(),
        )
        log_sigma = single_tensor_op(scaled, ElementwiseOp.LOG)
        return scalar_tensor_op(0.25, log_sigma, ElementwiseOp.PROD)

    def loss_weight(self, sigma: DTensor) -> DTensor:
        """Diffusion loss weighting: (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2."""
        self._check_sigma_placement(sigma)
        sigma_sq = scalar_tensor_op(2, sigma, ElementwiseOp.POW)
        numer = scalar_tensor_op(self.sigma_data**2, sigma_sq, ElementwiseOp.SUM)
        denom = scalar_tensor_op(self.sigma_data**2, sigma_sq, ElementwiseOp.PROD)
        return elementwise_op(numer, denom, ElementwiseOp.DIV)

    def noise_distribution(self, batch_size: int, dtype: torch.dtype = torch.float32) -> DTensor:
        """Sample noise levels from training distribution."""
        noise = create_distributed_randn(
            (batch_size,),
            device_mesh=self.device_mesh,
            placements=TIMES_PLACEMENTS_1D,
            dtype=dtype,
        )
        noise = scalar_tensor_op(self.P_std, noise, ElementwiseOp.PROD)
        noise = single_tensor_op(noise, ElementwiseOp.EXP)
        noise = scalar_tensor_op(self.sigma_data * exp(self.P_mean), noise, ElementwiseOp.PROD)
        return noise

    # ------------------------------------------------------------------
    # Preconditioned network forward
    # ------------------------------------------------------------------

    def preconditioned_network_forward(
        self,
        noised_atom_coords: DTensor,
        sigma: float | DTensor,
        network_condition_kwargs: dict,
    ) -> DTensor:
        """Preconditioned forward: c_skip * x + c_out * score_model(c_in * x, c_noise).

        Parameters
        ----------
        noised_atom_coords : DTensor
            ``[B*M, N_atoms, 3]`` with placements ``(Shard(0), Shard(1))``.
        sigma : float or DTensor
            Noise level.
        network_condition_kwargs : dict
            Conditioning arguments for the score model.

        Returns
        -------
        DTensor
            Denoised coordinates, same shape/placements as input.
        """
        batch_size = noised_atom_coords.shape[0]

        if isinstance(sigma, float):
            sigma = full(
                (batch_size,),
                sigma,
                dtype=noised_atom_coords.dtype,
                device_mesh=self.device_mesh,
                placements=TIMES_PLACEMENTS_1D,
            )

        padded_sigma = shardwise_repeat_interleave(shardwise_unsqueeze(sigma, dim=-1), 3, -1)

        r_noisy = replicate_op(noised_atom_coords, self.c_in(padded_sigma), 1, ReplicateOp.PROD)
        times = self.c_noise(sigma)

        r_update = self.score_model(
            r_noisy=r_noisy,
            times=times,
            **network_condition_kwargs,
        )

        skip_term = replicate_op(noised_atom_coords, self.c_skip(padded_sigma), 1, ReplicateOp.PROD)
        out_term = replicate_op(r_update, self.c_out(padded_sigma), 1, ReplicateOp.PROD)
        denoised_coords = elementwise_op(skip_term, out_term, ElementwiseOp.SUM)

        return denoised_coords

    # ------------------------------------------------------------------
    # Sampling schedule
    # ------------------------------------------------------------------

    def sample_schedule(self, num_sampling_steps: int | None = None) -> torch.Tensor:
        """Generate sigma schedule for sampling. Returns plain Tensor."""
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        if num_sampling_steps < 2:
            raise ValueError(f"Need at least 2 sampling steps, got {num_sampling_steps}")
        inv_rho = 1 / self.rho
        steps = torch.arange(num_sampling_steps, device=self.device, dtype=torch.float32)
        sigmas = (
            self.sigma_max**inv_rho
            + steps / (num_sampling_steps - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data
        sigmas = torch.nn.functional.pad(sigmas, (0, 1), value=0.0)
        return sigmas

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        s_inputs: DTensor,
        s_trunk: DTensor,
        feats: dict[str, DTensor],
        diffusion_conditioning: dict[str, DTensor],
        multiplicity: int = 1,
    ) -> dict[str, DTensor]:
        """Training forward: add noise, run preconditioned network, return denoised coords.

        Parameters
        ----------
        s_inputs : DTensor
            ``[B, N, token_s]`` input single representation.
        s_trunk : DTensor
            ``[B, N, token_s]`` trunk single representation.
        feats : dict[str, DTensor]
            Unpacked atom features. Must contain 'coords' and 'atom_pad_mask'.
        diffusion_conditioning : dict[str, DTensor]
            Pre-computed conditioning from ``DiffusionConditioning``.
        multiplicity : int
            Number of diffusion samples per batch element.

        Returns
        -------
        dict[str, DTensor]
            ``denoised_atom_coords``, ``sigmas``, ``aligned_true_atom_coords``.
        """
        coords = feats["coords"]
        atom_pad_mask = feats["atom_pad_mask"]
        B = atom_pad_mask.shape[0]

        coords_dtype = coords.dtype
        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(B, dtype=coords_dtype)
            sigmas = shardwise_repeat_interleave(sigmas, multiplicity, 0)
        else:
            sigmas = self.noise_distribution(B * multiplicity, dtype=coords_dtype)

        padded_sigmas = shardwise_repeat_interleave(shardwise_unsqueeze(sigmas, dim=-1), 3, -1)

        atom_coords = coords
        atom_mask = shardwise_repeat_interleave(atom_pad_mask, multiplicity, 0)

        atom_coords = _center_random_augmentation_1d(
            atom_coords,
            atom_mask,
            augmentation=self.coordinate_augmentation,
        )

        noise = create_distributed_randn(
            atom_coords.shape,
            device_mesh=self.device_mesh,
            placements=atom_coords.placements,
            dtype=atom_coords.dtype,
        )

        noised_atom_coords = elementwise_op(
            atom_coords,
            replicate_op(noise, padded_sigmas, 1, ReplicateOp.PROD),
            ElementwiseOp.SUM,
        )

        network_condition_kwargs = {
            "s_inputs": s_inputs,
            "s_trunk": s_trunk,
            "feats": feats,
            "multiplicity": multiplicity,
            "diffusion_conditioning": diffusion_conditioning,
        }

        denoised_atom_coords = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            network_condition_kwargs=network_condition_kwargs,
        )

        return {
            "denoised_atom_coords": denoised_atom_coords,
            "sigmas": sigmas,
            "aligned_true_atom_coords": atom_coords,
        }

    # ------------------------------------------------------------------
    # Sampling (inference)
    # ------------------------------------------------------------------

    def sample(
        self,
        atom_mask: DTensor,
        num_sampling_steps: int | None = None,
        multiplicity: int = 1,
        max_parallel_samples: int | None = None,
        **network_condition_kwargs,
    ) -> dict[str, DTensor | None]:
        """Sample from the diffusion model (inference denoising loop).

        Parameters
        ----------
        atom_mask : DTensor
            ``[B, N_atoms]`` with placements ``(Shard(0), Shard(1))``.
        num_sampling_steps : int or None
            Number of sampling steps. If None, uses default.
        multiplicity : int
            Multiplicity factor.
        max_parallel_samples : int or None
            Maximum samples per chunk. If None, all at once.
        **network_condition_kwargs
            Conditioning: s_inputs, s_trunk, feats, diffusion_conditioning.

        Returns
        -------
        dict[str, DTensor | None]
            ``sample_atom_coords``, ``diff_token_repr`` (always None).
        """
        if max_parallel_samples is None:
            max_parallel_samples = multiplicity

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = shardwise_repeat_interleave(atom_mask, multiplicity, 0)
        shape = (*atom_mask.shape, 3)

        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        init_sigma = sigmas[0].item()
        atom_coords = create_distributed_randn(
            shape,
            device_mesh=self.device_mesh,
            placements=atom_mask.placements,
            scale=init_sigma,
        )
        atom_coords_denoised = None

        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            aug = self.coordinate_augmentation
            result = _center_random_augmentation_1d(
                atom_coords,
                atom_mask,
                augmentation=aug,
                return_second_coords=True,
                second_coords=atom_coords_denoised,
                return_roto=aug,
            )
            if aug:
                atom_coords, atom_coords_denoised, _ = result
            else:
                atom_coords, atom_coords_denoised = result

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()
            t_hat = sigma_tm * (1 + gamma)
            noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = create_distributed_randn(
                shape,
                device_mesh=self.device_mesh,
                placements=atom_mask.placements,
                scale=sqrt(noise_var),
            )
            atom_coords_noisy = elementwise_op(atom_coords, eps, ElementwiseOp.SUM)

            with torch.no_grad():
                placements = atom_coords_noisy.placements
                noisy_local = atom_coords_noisy.to_local()
                denoised_local = torch.zeros_like(noisy_local)
                if noisy_local.shape[0] % multiplicity != 0:
                    raise ValueError(
                        f"noisy_local.shape[0] not divisible by multiplicity: "
                        f"{noisy_local.shape[0]} % {multiplicity} = "
                        f"{noisy_local.shape[0] % multiplicity}"
                    )
                B_local = noisy_local.shape[0] // multiplicity

                sample_ids = torch.arange(multiplicity, device=self.device)
                n_chunks = (multiplicity + max_parallel_samples - 1) // max_parallel_samples
                sample_ids_chunks = sample_ids.chunk(n_chunks)

                for sample_ids_chunk in sample_ids_chunks:
                    chunk_M = sample_ids_chunk.numel()
                    noisy_chunk_local = noisy_local.unflatten(0, (B_local, multiplicity))[:, sample_ids_chunk].flatten(
                        0, 1
                    )
                    chunk_global_shape = (
                        atom_coords_noisy.shape[0] * chunk_M // multiplicity,
                        atom_coords_noisy.shape[1],
                        3,
                    )
                    noisy_chunk_dt = DTensor.from_local(
                        noisy_chunk_local,
                        device_mesh=self.device_mesh,
                        placements=placements,
                        shape=chunk_global_shape,
                        stride=LayoutRightMap(chunk_global_shape).strides,
                    )

                    denoised_chunk_dt = self.preconditioned_network_forward(
                        noisy_chunk_dt,
                        t_hat,
                        network_condition_kwargs=dict(multiplicity=chunk_M, **network_condition_kwargs),
                    )

                    denoised_local.unflatten(0, (B_local, multiplicity))[:, sample_ids_chunk] = (
                        denoised_chunk_dt.to_local().unflatten(0, (B_local, chunk_M))
                    )

                atom_coords_denoised = DTensor.from_local(
                    denoised_local,
                    device_mesh=self.device_mesh,
                    placements=placements,
                    shape=atom_coords_noisy.shape,
                    stride=atom_coords_noisy.stride(),
                )

            # Alignment reverse diffusion
            if self.alignment_reverse_diff:
                from boltz.distributed.model.loss.diffusion_1d import weighted_rigid_align_1d

                atom_coords_noisy = weighted_rigid_align_1d(
                    atom_coords_noisy,
                    atom_coords_denoised,
                    atom_mask,
                    atom_mask,
                )

            # Next step
            denoised_over_sigma = scalar_tensor_op(
                1 / t_hat,
                elementwise_op(atom_coords_noisy, atom_coords_denoised, ElementwiseOp.SUB),
                ElementwiseOp.PROD,
            )
            atom_coords = elementwise_op(
                atom_coords_noisy,
                scalar_tensor_op(
                    self.step_scale * (sigma_t - t_hat),
                    denoised_over_sigma,
                    ElementwiseOp.PROD,
                ),
                ElementwiseOp.SUM,
            )

        return {"sample_atom_coords": atom_coords, "diff_token_repr": None}

    # ------------------------------------------------------------------
    # Compute loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        feats: dict[str, DTensor],
        out_dict: dict[str, DTensor],
        add_smooth_lddt_loss: bool = True,
        nucleotide_loss_weight: float = 5.0,
        ligand_loss_weight: float = 10.0,
        multiplicity: int = 1,
        filter_by_plddt: float = 0.0,
        use_triton_kernel: bool = True,
    ) -> dict[str, DTensor]:
        """Compute diffusion loss.

        Parameters
        ----------
        feats : dict[str, DTensor]
            Features.
        out_dict : dict[str, DTensor]
            Output from forward().
        add_smooth_lddt_loss : bool
            Whether to add smooth LDDT loss.
        nucleotide_loss_weight : float
            Weight for nucleotide loss.
        ligand_loss_weight : float
            Weight for ligand loss.
        multiplicity : int
            Multiplicity factor.
        filter_by_plddt : float
            pLDDT filter threshold.
        use_triton_kernel : bool
            Whether to use Triton kernel for smooth LDDT loss.

        Returns
        -------
        dict[str, DTensor]
            Loss dict with "loss" and "loss_breakdown" keys.
        """
        from boltz.data import const
        from boltz.distributed.model.layers.atom_to_token import (
            _single_repr_placements,
            single_repr_token_to_atom,
        )
        from boltz.distributed.model.loss.diffusion_1d import weighted_rigid_align_1d

        with torch.autocast("cuda", enabled=False):
            safe_dtype = torch.promote_types(torch.float32, out_dict["denoised_atom_coords"].dtype)
            denoised_atom_coords = out_dict["denoised_atom_coords"].to(dtype=safe_dtype)
            sigmas = out_dict["sigmas"].to(dtype=safe_dtype)

            resolved_atom_mask_uni = feats["atom_resolved_mask"].to(dtype=safe_dtype)
            resolved_atom_mask = shardwise_repeat_interleave(resolved_atom_mask_uni, multiplicity, 0)

            if filter_by_plddt > 0:
                if "plddt" not in feats:
                    raise RuntimeError("Missing required plddt data in feats for plddt filtering")
                plddt_mask = scalar_tensor_op(filter_by_plddt, feats["plddt"], ElementwiseOp.LT)
                resolved_atom_mask_uni_plddt_masked = elementwise_op(
                    resolved_atom_mask_uni,
                    plddt_mask.to(dtype=safe_dtype),
                    ElementwiseOp.PROD,
                )
                resolved_atom_mask_plddt_masked = shardwise_repeat_interleave(
                    resolved_atom_mask_uni_plddt_masked,
                    multiplicity,
                    0,
                )
            else:
                resolved_atom_mask_uni_plddt_masked = resolved_atom_mask_uni
                resolved_atom_mask_plddt_masked = resolved_atom_mask

            # Redistribute mol_type and atom_to_token to single-repr placements
            # (Shard(0), Replicate()) before calling single_repr_token_to_atom,
            # which validates that placement. Without this, atom_to_token arrives
            # as (Shard(0), Shard(1)) from the featurizer's atom-sharding layout.
            # Pattern follows confidence_1d.py (commit 7e96fa15).
            device_mesh = self.device_mesh
            single_pl = _single_repr_placements(device_mesh.ndim)
            mol_type_for_a2t = feats["mol_type"].redistribute(device_mesh, placements=single_pl)
            atom_to_token_for_a2t = feats["atom_to_token"].redistribute(device_mesh, placements=single_pl)
            atom_type = single_repr_token_to_atom(
                mol_type_for_a2t.to(dtype=safe_dtype),
                atom_to_token_for_a2t.to(dtype=safe_dtype),
            )
            atom_type_mult = shardwise_repeat_interleave(atom_type, multiplicity, 0)

            is_nucleotide_mult = elementwise_op(
                scalar_tensor_op(const.chain_type_ids["DNA"], atom_type_mult, ElementwiseOp.EQUAL),
                scalar_tensor_op(const.chain_type_ids["RNA"], atom_type_mult, ElementwiseOp.EQUAL),
                ElementwiseOp.SUM,
            )
            nucleotide_loss_weights = scalar_tensor_op(
                nucleotide_loss_weight,
                is_nucleotide_mult,
                ElementwiseOp.PROD,
            )
            ligand_loss_weights = scalar_tensor_op(
                ligand_loss_weight,
                scalar_tensor_op(const.chain_type_ids["NONPOLYMER"], atom_type_mult, ElementwiseOp.EQUAL),
                ElementwiseOp.PROD,
            )
            align_weights = scalar_tensor_op(
                1.0,
                elementwise_op(nucleotide_loss_weights, ligand_loss_weights, ElementwiseOp.SUM),
                ElementwiseOp.SUM,
            )
            # `align_weights` inherits single-repr placements (Shard(0), Replicate()) from
            # `atom_type_mult`. Downstream uses (weighted_rigid_align_1d, elementwise_op vs
            # `resolved_atom_mask_plddt_masked`) require the atom-sharded placement
            # (Shard(0), Shard(1)) used by the resolved atom mask. Redistribute once.
            align_weights = align_weights.redistribute(device_mesh, placements=(Shard(0), Shard(1)))

            with torch.no_grad():
                atom_coords = out_dict["aligned_true_atom_coords"].to(dtype=safe_dtype)
                atom_coords_aligned_ground_truth = weighted_rigid_align_1d(
                    atom_coords,
                    denoised_atom_coords,
                    align_weights.to(dtype=safe_dtype),
                    mask=resolved_atom_mask.to(dtype=safe_dtype),
                )
            atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
                dtype=denoised_atom_coords.dtype,
            )

            # Weighted MSE loss
            mse_loss = elementwise_op(denoised_atom_coords, atom_coords_aligned_ground_truth, ElementwiseOp.SUB)
            mse_loss = scalar_tensor_op(2.0, mse_loss, ElementwiseOp.POW)
            mse_loss = shardwise_sum(mse_loss, dim=-1)
            mse_loss = elementwise_op(mse_loss, resolved_atom_mask_plddt_masked, ElementwiseOp.PROD)

            resolved_align_weights = elementwise_op(align_weights, resolved_atom_mask_plddt_masked, ElementwiseOp.PROD)
            denom = sharded_sum(
                scalar_tensor_op(3.0, resolved_align_weights, ElementwiseOp.PROD),
                dim=-1,
            )
            denom = scalar_tensor_op(1e-5, denom, ElementwiseOp.SUM)

            mse_loss = elementwise_op(mse_loss, resolved_align_weights, ElementwiseOp.PROD)
            mse_loss = sharded_sum(mse_loss, dim=-1)
            mse_loss = elementwise_op(mse_loss, denom, ElementwiseOp.DIV)
            loss_weights = self.loss_weight(sigmas)

            mse_loss = elementwise_op(mse_loss, loss_weights, ElementwiseOp.PROD)
            mse_loss = scalar_tensor_op(
                1.0 / mse_loss.shape[0],
                sharded_sum(mse_loss, dim=0),
                ElementwiseOp.PROD,
            )
            total_loss = mse_loss

            lddt_loss = zeros(
                total_loss.shape,
                requires_grad=False,
                device_mesh=total_loss.device_mesh,
                placements=total_loss.placements,
            )
            if add_smooth_lddt_loss:
                from boltz.distributed.model.loss.smooth_lddt_1d import smooth_lddt_loss_1d

                is_nucleotide = elementwise_op(
                    scalar_tensor_op(const.chain_type_ids["DNA"], atom_type, ElementwiseOp.EQUAL),
                    scalar_tensor_op(const.chain_type_ids["RNA"], atom_type, ElementwiseOp.EQUAL),
                    ElementwiseOp.SUM,
                )
                # `is_nucleotide` inherits single-repr (Shard(0), Replicate()) from `atom_type`,
                # but `smooth_lddt_loss_1d` requires atom-sharded (Shard(0), Shard(1)) inputs.
                is_nucleotide = is_nucleotide.redistribute(device_mesh, placements=(Shard(0), Shard(1)))

                dp_group = self.device_mesh.get_group(0)
                cp_group = self.device_mesh.get_group(1)

                lddt_loss = smooth_lddt_loss_1d(
                    denoised_atom_coords,
                    atom_coords,
                    is_nucleotide=is_nucleotide,
                    coords_mask=resolved_atom_mask_uni_plddt_masked,
                    device_mesh=self.device_mesh,
                    dp_group=dp_group,
                    cp_group=cp_group,
                    multiplicity=multiplicity,
                    use_triton=use_triton_kernel,
                )
                total_loss = elementwise_op(total_loss, lddt_loss, ElementwiseOp.SUM)

            loss_breakdown = {
                "mse_loss": mse_loss,
                "smooth_lddt_loss": lddt_loss,
            }

        return {"loss": total_loss, "loss_breakdown": loss_breakdown}
