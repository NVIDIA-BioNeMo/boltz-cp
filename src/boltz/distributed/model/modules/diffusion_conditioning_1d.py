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

"""1D CP DiffusionConditioning for 2D mesh ``(dp, cp)``.

Adapts the 2D CP :class:`~.diffusion_conditioning.DiffusionConditioning` for
the simpler 2D mesh topology. Key differences from the 2D CP version:

1. Uses :class:`~.encoders_1d.PairwiseConditioning1D` (2-element placement
   assertions) instead of the 3D-mesh ``PairwiseConditioning``.

2. The ``AtomEncoder`` z-path (pair-to-atom gather via ``distributed_outer_gather``,
   which the 3D-mesh version uses) is replaced with a two-step gather suited to the
   row-slab pair layout ``(Shard(0), Shard(1))`` ``[B_local, N/cp, N_full, C]``:
   (a) a CROSS-RANK row fetch of the query-token z-rows via :func:`distributed_gather`
   (axis=1), then (b) a LOCAL key-column gather via the generic
   :class:`_DTensorGatherAlongAxis` (columns are full per row-slab, so it is collective-
   free). The cross-rank step is required because ``distributed_pack_and_pad`` re-shards
   atoms by the valid mask, INDEPENDENTLY of the row-slab token order — so a rank's query
   atoms can map to token z-rows owned by another rank (at window/rank boundaries). A
   purely-local gather silently fetches the wrong row for those atoms (the prior bug).
   The window-batching index construction (broadcasting key ids over the query-window
   axis, :func:`_broadcast_window_key_ids`) lives in :func:`_atom_encoder_1d`, keeping the
   gather itself a generic ``src + idx`` operation.
"""

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.nn import Module

from boltz.distributed.data.feature.featurizer import pack_atom_features
from boltz.distributed.model.layers.cat_and_chunk import shardwise_cat
from boltz.distributed.model.layers.elementwise_op import (
    ElementwiseOp,
    elementwise_op,
    scalar_tensor_op,
)
from boltz.distributed.model.layers.flatten_and_unflatten import (
    shardwise_flatten,
    shardwise_flatten_sharded,
    shardwise_unflatten_sharded,
)
from boltz.distributed.model.layers.gather import distributed_gather
from boltz.distributed.model.layers.layernorm import LayerNormParamsReplicated
from boltz.distributed.model.layers.linear import LinearParamsReplicated
from boltz.distributed.model.layers.shardwise_op import ShardwiseOuterOp, shardwise_outer_op, shardwise_sum
from boltz.distributed.model.layers.squeeze import shardwise_unsqueeze
from boltz.distributed.model.layers.utils import convert_single_repr_to_window_batched_key
from boltz.distributed.model.modules.encoders_1d import PairwiseConditioning1D
from boltz.distributed.model.modules.utils import validate_window_batching_parameters
from boltz.distributed.utils import update_exhaustive_strides
from boltz.model.modules.diffusion_conditioning import DiffusionConditioning as SerialDiffusionConditioning
from boltz.model.modules.encodersv2 import AtomEncoder as SerialAtomEncoder


class _DTensorGatherAlongAxis(torch.autograd.Function):
    """Collective-free gather of a DTensor along a locally-replicated axis (scatter-add backward).

    Generic ``src + idx`` gather: equivalent to ``torch.gather(src, dim, idx)`` evaluated on
    the rank-local shards and re-wrapped as a DTensor. It carries NO problem-specific
    (e.g. window-batching) semantics — index construction such as broadcasting key ids over
    a query-window axis is the CALLER's responsibility.

    Shapes (``lead`` = the axes before ``dim``; ``trail`` = the axes after ``dim``)::

        src : (*lead, A, *trail)   gather axis ``dim`` has length A
        idx : (*lead, M)           integer indices into axis ``dim``; broadcast over ``trail``
        out : (*lead, M, *trail)   out[*l, m, *t] = src[*l, idx[*l, m], *t]

    The gather is communication-free iff both hold (validated in ``forward``):

      * ``src`` and ``idx`` share the same ``device_mesh`` and ``placements`` — so their
        local shards are aligned, and
      * every ``Shard`` placement targets a leading axis ``< dim``; the gather axis ``dim``
        and all trailing axes are REPLICATED (locally complete), so each rank resolves all
        of its indices without any peer fetch.

    This is a PURE gather: no clamping and no validity mask. Callers must supply in-range
    indices, and any validity/padding zeroing (e.g. serial's all-zero ``atom→token`` one-hot
    rows for padding atoms ⇒ ``z_to_p == 0`` at padding query/key positions) is the CALLER's
    responsibility — applied as an explicit mask on the gathered output (and, for the row
    fetch, via :func:`distributed_gather`'s ``idx_mask``). Keeping it mask-free keeps the helper
    a faithful ``torch.gather`` adjoint pair.

    Output placements inherit ``src.placements``; the output global shape is ``src``'s with
    axis ``dim`` resized to ``M = idx.shape[-1]``.

    Communication budget: 0 collectives forward, 0 collectives backward.
    Memory: the backward grad buffer is O(size(src)/cp) — the gather adjoint requires a zero
    buffer the size of ``src``. The only saved-for-backward tensor is the index ``(*lead, M)``
    = O(prod(lead)*M/cp); the per-trailing-axis broadcast is re-materialized as a view, so no
    full-A tensor is ever saved.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, src: DTensor, idx: DTensor, dim: int) -> DTensor:
        if not isinstance(src, DTensor):
            raise TypeError(f"src must be a DTensor, got {type(src)}")
        if not isinstance(idx, DTensor):
            raise TypeError(f"idx must be a DTensor, got {type(idx)}")
        if src.device_mesh != idx.device_mesh:
            raise ValueError(f"src and idx must share device_mesh, got {src.device_mesh} vs {idx.device_mesh}")

        ndim = src.ndim
        dim_norm = dim + ndim if dim < 0 else dim
        if not (0 <= dim_norm < ndim):
            raise ValueError(f"dim {dim} out of range for src.ndim={ndim}")

        # idx must cover src's leading axes [0, dim) plus one trailing gather-count axis M.
        if idx.ndim != dim_norm + 1:
            raise ValueError(
                f"idx.ndim must be dim+1={dim_norm + 1} (leading axes + gather-count axis), got {idx.ndim}"
            )
        if idx.shape[:dim_norm] != src.shape[:dim_norm]:
            raise ValueError(
                f"idx leading shape {tuple(idx.shape[:dim_norm])} must match "
                f"src leading shape {tuple(src.shape[:dim_norm])}"
            )
        if idx.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"idx must be an integer tensor, got dtype {idx.dtype}")

        # Placements: identical across src/idx, and Shard only on leading axes (< dim) so the
        # gather axis and all trailing axes are locally complete (replicated). Reject Partial.
        if src.placements != idx.placements:
            raise ValueError(f"src and idx must share placements, got {src.placements} vs {idx.placements}")
        mesh = src.device_mesh
        for i_mesh, placement in enumerate(src.placements):
            if placement.is_partial():
                raise ValueError(f"Partial placement on mesh dim {i_mesh} is not supported")
            if isinstance(placement, Shard):
                if placement.dim >= dim_norm:
                    raise ValueError(
                        f"gather axis dim={dim_norm} and trailing axes must be replicated, "
                        f"but mesh dim {i_mesh} shards tensor axis {placement.dim}"
                    )
                if src.shape[placement.dim] % mesh.size(i_mesh) != 0:
                    raise ValueError(
                        f"uneven sharding: src.shape[{placement.dim}]={src.shape[placement.dim]} "
                        f"not divisible by mesh size {mesh.size(i_mesh)}"
                    )

        src_local = src.to_local()
        idx_local = idx.to_local()
        # take_along_dim broadcasts the size-1 trailing axes of idx against src, so it gathers
        # along `dim` only — equivalent to torch.gather over an explicitly trailing-expanded idx.
        idx_view = idx_local.reshape(idx_local.shape + (1,) * (src_local.ndim - dim_norm - 1))
        gathered = torch.take_along_dim(src_local, idx_view, dim_norm).contiguous()

        if src.requires_grad:
            ctx.save_for_backward(idx_local)
            ctx.dim = dim_norm
            ctx.src_local_shape = src_local.shape
            ctx.src_global_shape = src.shape
            ctx.src_global_stride = src.stride()
            ctx.src_placements = src.placements
            ctx.device_mesh = mesh

        out_global_shape = src.shape[:dim_norm] + (idx.shape[-1],) + src.shape[dim_norm + 1 :]
        out_stride = update_exhaustive_strides(gathered.shape, gathered.stride(), out_global_shape)
        return DTensor.from_local(
            gathered,
            device_mesh=mesh,
            placements=src.placements,
            shape=out_global_shape,
            stride=out_stride,
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: DTensor):
        (idx_local,) = ctx.saved_tensors
        dim = ctx.dim
        grad_local = grad_output.to_local() if isinstance(grad_output, DTensor) else grad_output
        grad_local = grad_local.to(torch.promote_types(grad_local.dtype, torch.float32))

        # The adjoint of the take_along_dim gather is scatter-add into a zero buffer the size
        # of src. scatter_add_ does NOT broadcast the index (unlike the forward take_along_dim),
        # so re-expand the saved index over the trailing axes as a view — no extra memory.
        idx_expanded = _expand_over_trailing(idx_local, ctx.src_local_shape, dim)
        grad_src = torch.zeros(ctx.src_local_shape, dtype=grad_local.dtype, device=grad_local.device)
        grad_src.scatter_add_(dim, idx_expanded, grad_local)
        grad_src = grad_src.contiguous()

        grad_src_dt = DTensor.from_local(
            grad_src,
            device_mesh=ctx.device_mesh,
            placements=ctx.src_placements,
            shape=ctx.src_global_shape,
            stride=ctx.src_global_stride,
        )
        return grad_src_dt, None, None


def _expand_over_trailing(lead_m_local: torch.Tensor, src_local_shape, dim: int) -> torch.Tensor:
    """Broadcast a ``(*lead, M)`` index over ``src``'s trailing axes for the scatter-add backward.

    Returns a view of shape ``(*lead, M, *src_local_shape[dim+1:])`` (zero-stride on the
    trailing axes) so ``scatter_add_(dim, view, grad)`` shares the index across the trailing
    feature axes. No memory is materialised. (The forward gather uses ``take_along_dim``, which
    broadcasts the size-1 trailing axes itself; only the backward needs this explicit view
    because ``scatter_add_`` does not broadcast its index.)
    """
    trailing = tuple(src_local_shape[dim + 1 :])
    view = lead_m_local.reshape(lead_m_local.shape + (1,) * len(trailing))
    return view.expand(lead_m_local.shape + trailing)


def _dtensor_gather_along_axis(src: DTensor, idx: DTensor, dim: int) -> DTensor:
    """Collective-free DTensor gather along ``dim`` with scatter-add backward.

    Thin wrapper over :class:`_DTensorGatherAlongAxis`; see that class for the input/output
    placement and shape contract and the budget. Pure gather (no mask) — validity zeroing is
    the caller's responsibility.
    """
    return _DTensorGatherAlongAxis.apply(src, idx, dim)


def _broadcast_over_query_window(x: DTensor, query_window_size: int) -> DTensor:
    """Broadcast a window-batched ``(B, K, H)`` tensor over the query-window axis to ``(B, K, W, H)``.

    Window-batching semantics: the ``H`` key tokens of window ``k`` are shared by all ``W``
    query atoms in that window, so the key ids / validity mask for the row-fetched pair tensor
    ``(B, K, W, N_cols, atom_z)`` must carry a ``W`` axis. ``x`` (integer ids or a bool mask) is
    non-differentiable, so the broadcast is built directly from the local shard with an explicit
    global shape/stride/placements — no collective, no autograd node. Placements are unchanged:
    inserting the ``W`` axis at position 2 leaves the ``Shard(0)``/``Shard(1)`` batch/window
    shards in place.
    """
    local = x.to_local()  # (B_local, K_local, H)
    if local.ndim != 3:
        raise ValueError(f"x must be 3D (B, K, H), got shape {tuple(x.shape)}")
    b_local, k_local, h = local.shape
    local_bkwh = local[:, :, None, :].expand(b_local, k_local, query_window_size, h).contiguous()
    global_shape = x.shape[:2] + (query_window_size, x.shape[2])
    global_stride = update_exhaustive_strides(local_bkwh.shape, local_bkwh.stride(), global_shape)
    return DTensor.from_local(
        local_bkwh,
        device_mesh=x.device_mesh,
        placements=x.placements,
        shape=global_shape,
        stride=global_stride,
    )


def _atom_encoder_1d(
    c: DTensor,
    embed_atompair_ref_pos: nn.Module,
    embed_atompair_ref_dist: nn.Module,
    embed_atompair_mask: nn.Module,
    s_to_c_trans: nn.Module | None,
    z_to_p_trans: nn.Module | None,
    c_to_p_trans_q: nn.Module,
    c_to_p_trans_k: nn.Module,
    p_mlp: nn.Module,
    feats: dict[str, DTensor],
    s_trunk: DTensor | None,
    z: DTensor | None,
    structure_prediction: bool,
    W: int,
    H: int,
) -> tuple[DTensor, DTensor, DTensor]:
    """1D CP atom encoder with local pair-to-atom gather.

    Identical to :func:`~.encoders._atom_encoder` except that the
    ``distributed_outer_gather`` call (which requires z sharded on two
    separate mesh dims) is replaced with a local gather. Under 1D CP
    row-slab sharding, each rank holds the full column dimension of the
    pair tensor, so the gather is purely local.

    Parameters
    ----------
    c : DTensor
        Pre-embedded atom single representation, shape ``(B, N_atoms, atom_s)``.
        Placements: ``(Shard(0), Shard(1))``.
    z : DTensor or None
        Token pair representation, shape ``(B, N_tokens, N_tokens, token_z)``.
        Placements: ``(Shard(0), Shard(1))`` (row-slab). Required when
        ``structure_prediction=True``.

    Returns
    -------
    tuple[DTensor, DTensor, DTensor]
        ``(q, c, p)`` with placements ``(Shard(0), Shard(1))``.
    """
    if structure_prediction:
        if s_to_c_trans is None or z_to_p_trans is None:
            raise ValueError("structure_prediction=True requires s_to_c_trans and z_to_p_trans.")
        if s_trunk is None or z is None:
            raise ValueError("structure_prediction=True requires s_trunk and z.")
    else:
        if s_to_c_trans is not None or z_to_p_trans is not None:
            raise ValueError("structure_prediction=False but s_to_c_trans or z_to_p_trans was provided.")
        if s_trunk is not None or z is not None:
            raise ValueError("structure_prediction=False but s_trunk or z was provided.")

    N = c.shape[1]
    if N % W != 0:
        raise ValueError(f"N={N} must be divisible by W={W}, but N % W = {N % W}")
    K = N // W

    compute_dtype = torch.promote_types(c.dtype, torch.float32)

    atom_ref_pos = feats["ref_pos"]
    atom_mask_bool = feats["atom_pad_mask"].bool()
    atom_uid = feats["ref_space_uid"]

    # Window-batched pair computation (identical to 2D CP _atom_encoder)
    atom_ref_pos_q = shardwise_unflatten_sharded(atom_ref_pos, axis=1, sizes=(K, W))
    atom_ref_pos_k = convert_single_repr_to_window_batched_key(atom_ref_pos, W, H)

    d = shardwise_outer_op(atom_ref_pos_q, atom_ref_pos_k, axis=2, op=ShardwiseOuterOp.SUBTRACT)
    d = scalar_tensor_op(-1.0, d, ElementwiseOp.PROD)
    d_norm = shardwise_sum(elementwise_op(d, d, ElementwiseOp.PROD), dim=-1, keepdim=True)
    d_norm = scalar_tensor_op(1.0, scalar_tensor_op(1.0, d_norm, ElementwiseOp.SUM), ElementwiseOp.DIV)

    atom_mask_q = shardwise_unflatten_sharded(atom_mask_bool, axis=1, sizes=(K, W))
    atom_mask_k = convert_single_repr_to_window_batched_key(atom_mask_bool, W, H)
    atom_uid_q = shardwise_unflatten_sharded(atom_uid, axis=1, sizes=(K, W))
    atom_uid_k = convert_single_repr_to_window_batched_key(atom_uid, W, H)

    mask_and = shardwise_outer_op(atom_mask_q, atom_mask_k, axis=2, op=ShardwiseOuterOp.LOGICAL_AND)
    uid_eq = shardwise_outer_op(atom_uid_q, atom_uid_k, axis=2, op=ShardwiseOuterOp.EQUAL)
    v = elementwise_op(mask_and, uid_eq, ElementwiseOp.BITAND)
    v = shardwise_unsqueeze(v, -1).to(compute_dtype)

    p = embed_atompair_ref_pos(d) * v
    p = elementwise_op(p, embed_atompair_ref_dist(d_norm) * v, ElementwiseOp.SUM)
    p = elementwise_op(p, embed_atompair_mask(v) * v, ElementwiseOp.SUM)

    q = c

    if structure_prediction:
        atom_to_token_ids_global = feats["atom_to_token_ids_global"]
        atom_to_token_ids_global_q = shardwise_unflatten_sharded(atom_to_token_ids_global, axis=1, sizes=(K, W))

        # Token-to-atom gather for single repr (same as 2D CP)
        s_to_c = s_to_c_trans(s_trunk.to(compute_dtype))
        s_to_c = distributed_gather(
            s_to_c, atom_to_token_ids_global_q, axis=1, are_ids_contiguous=True, idx_mask=atom_mask_q
        )
        s_to_c = shardwise_flatten_sharded(s_to_c, start_dim=1, end_dim=2)
        c = elementwise_op(c, s_to_c.to(c.dtype), ElementwiseOp.SUM)

        # Pair-to-atom gather for 1D CP row-slab z. z_to_p has placements (Shard(0), Shard(1))
        # = row-slab: each rank owns row-token shard [r*N/cp, (r+1)*N/cp) and the FULL column
        # dimension. The query atoms are sharded by the valid-mask PACK (distributed_pack_and_pad),
        # which is an INDEPENDENT sharding from the row-slab token order — so a query atom owned by
        # this rank can map to a token whose z-row lives on ANOTHER rank (happens at window/rank
        # boundaries). A purely-local row index (q_ids - rank*N/cp) is therefore wrong for those
        # atoms. We instead fetch the query-token z-ROWS cross-rank with distributed_gather (the
        # proven interval-P2P 1D gather), then gather the H key COLUMNS locally (columns are full
        # per row-slab). This is exactly serial-faithful (matches z[t_q, t_k]).
        z_to_p = z_to_p_trans(z.to(compute_dtype))

        atom_to_token_ids_global_k = convert_single_repr_to_window_batched_key(atom_to_token_ids_global, W, H)

        # Step 1 (cross-rank): fetch the query-token z-rows. z_to_p axis 1 = token-row dim;
        # atom_to_token_ids_global_q (B, K, W) holds GLOBAL token ids. ``idx_mask=atom_mask_q``
        # ZEROES padding-query rows — serial gives z_to_p=0 there (the pad atom's one-hot
        # atom→token row is all-zero), matching the s_to_c gather above and the 2D-CP
        # ``idx_n_mask=atom_mask_q``. Output (B, K, W, N_cols, atom_z), sharded on K (full cols).
        z_rows = distributed_gather(
            z_to_p, atom_to_token_ids_global_q, axis=1, are_ids_contiguous=True, idx_mask=atom_mask_q
        )
        # Step 2 (local): gather the H key columns per window (axis 3 = N_cols, full per row-slab)
        # with the generic (mask-free) gather. Window-batching lives HERE: broadcast the (B, K, H)
        # key ids over the W query-window axis to (B, K, W, H). The clamp the prior code used is
        # dropped — ids are in-range (atom→token argmax).
        k_ids_bkwh = _broadcast_over_query_window(atom_to_token_ids_global_k, W)
        z_to_p_dt = _dtensor_gather_along_axis(z_rows, k_ids_bkwh, dim=3)
        # ZERO padding-key columns: serial gives z_to_p=0 at pad keys (its key one-hot row is
        # all-zero); 2D-CP uses idx_m_mask=atom_mask_k. Mirror the v-masking style with an explicit
        # multiply by the window-batched key mask. (Padding-query rows are already zeroed by the
        # step-1 idx_mask=atom_mask_q.) Net (query-pad ∪ key-pad zeroing) = serial-exact.
        k_mask_v = shardwise_unsqueeze(_broadcast_over_query_window(atom_mask_k, W), -1).to(z_to_p_dt.dtype)
        z_to_p_dt = z_to_p_dt * k_mask_v

        p = elementwise_op(p, z_to_p_dt.to(p.dtype), ElementwiseOp.SUM)

    # c_to_p contributions (identical to 2D CP)
    c_q = shardwise_unflatten_sharded(c, axis=1, sizes=(K, W))
    c_q = c_to_p_trans_q(c_q)
    c_k = convert_single_repr_to_window_batched_key(c, W, H)
    c_k = c_to_p_trans_k(c_k)
    c_qk = shardwise_outer_op(c_q, c_k, axis=2, op=ShardwiseOuterOp.ADD)

    p = elementwise_op(p, c_qk, ElementwiseOp.SUM)
    p = elementwise_op(p, p_mlp(p), ElementwiseOp.SUM)

    return q, c, p


class AtomEncoder1D(Module):
    """1D CP AtomEncoder using local pair-to-atom gather.

    Same structure as :class:`~.encoders.AtomEncoder` but calls
    :func:`_atom_encoder_1d` which replaces ``distributed_outer_gather``
    with a local gather for 1D CP row-slab pair tensors.
    """

    def __init__(self, layer: SerialAtomEncoder, device_mesh: DeviceMesh) -> None:
        super().__init__()
        if not isinstance(layer, SerialAtomEncoder):
            raise TypeError(f"Expected SerialAtomEncoder, got {type(layer)}")

        if layer.use_residue_feats_atoms:
            raise NotImplementedError("DTensor AtomEncoder1D does not support use_residue_feats_atoms=True.")
        if layer.use_atom_backbone_feat:
            raise NotImplementedError("DTensor AtomEncoder1D does not support use_atom_backbone_feat=True.")

        self.use_no_atom_char = layer.use_no_atom_char
        self.use_atom_backbone_feat = layer.use_atom_backbone_feat
        self.use_residue_feats_atoms = layer.use_residue_feats_atoms
        self.structure_prediction = layer.structure_prediction
        self.atoms_per_window_queries = layer.atoms_per_window_queries
        self.atoms_per_window_keys = layer.atoms_per_window_keys
        validate_window_batching_parameters(
            self.atoms_per_window_queries, self.atoms_per_window_keys, use_window_batching=True
        )

        self.embed_atom_features = LinearParamsReplicated(layer.embed_atom_features, device_mesh)
        self.embed_atompair_ref_pos = LinearParamsReplicated(layer.embed_atompair_ref_pos, device_mesh)
        self.embed_atompair_ref_dist = LinearParamsReplicated(layer.embed_atompair_ref_dist, device_mesh)
        self.embed_atompair_mask = LinearParamsReplicated(layer.embed_atompair_mask, device_mesh)

        if self.structure_prediction:
            self.s_to_c_trans = nn.Sequential(
                LayerNormParamsReplicated(layer.s_to_c_trans[0], device_mesh),
                LinearParamsReplicated(layer.s_to_c_trans[1], device_mesh),
            )
            self.z_to_p_trans = nn.Sequential(
                LayerNormParamsReplicated(layer.z_to_p_trans[0], device_mesh),
                LinearParamsReplicated(layer.z_to_p_trans[1], device_mesh),
            )

        self.c_to_p_trans_q = nn.Sequential(
            nn.ReLU(),
            LinearParamsReplicated(layer.c_to_p_trans_q[1], device_mesh),
        )
        self.c_to_p_trans_k = nn.Sequential(
            nn.ReLU(),
            LinearParamsReplicated(layer.c_to_p_trans_k[1], device_mesh),
        )
        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearParamsReplicated(layer.p_mlp[1], device_mesh),
            nn.ReLU(),
            LinearParamsReplicated(layer.p_mlp[3], device_mesh),
            nn.ReLU(),
            LinearParamsReplicated(layer.p_mlp[5], device_mesh),
        )

    def _mask_padding_atoms(self, c: DTensor, atom_pad_mask: DTensor) -> DTensor:
        """Zero out representations at padding atom positions."""
        from boltz.distributed.model.modules.encoders import _mask_padding_atoms

        return _mask_padding_atoms(c, atom_pad_mask)

    def forward(
        self,
        feats: dict[str, DTensor],
        s_trunk: DTensor | None = None,
        z: DTensor | None = None,
    ) -> tuple[DTensor, DTensor, DTensor]:
        """Forward pass using 1D CP local gather for pair-to-atom.

        Parameters
        ----------
        feats : dict[str, DTensor]
            Atom features with placements ``(Shard(0), Shard(1))``.
        s_trunk : DTensor or None
            Single repr ``(B, N, token_s)`` with placements ``(Shard(0), Shard(1))``.
        z : DTensor or None
            Pair repr ``(B, N, N, token_z)`` with placements ``(Shard(0), Shard(1))``.

        Returns
        -------
        tuple[DTensor, DTensor, DTensor]
            ``(q, c, p)`` with placements ``(Shard(0), Shard(1))``.
        """
        with torch.autocast("cuda", enabled=False):
            atom_feats_list = [
                feats["ref_pos"],
                shardwise_unsqueeze(feats["ref_charge"], -1),
                feats["ref_element"],
            ]
            if not self.use_no_atom_char:
                atom_feats_list.append(shardwise_flatten(feats["ref_atom_name_chars"], start_dim=2, end_dim=3))

            atom_feats = shardwise_cat(atom_feats_list, dim=-1)
            c = self.embed_atom_features(atom_feats)
            c = self._mask_padding_atoms(c, feats["atom_pad_mask"])

            q, c, p = _atom_encoder_1d(
                c=c,
                embed_atompair_ref_pos=self.embed_atompair_ref_pos,
                embed_atompair_ref_dist=self.embed_atompair_ref_dist,
                embed_atompair_mask=self.embed_atompair_mask,
                s_to_c_trans=self.s_to_c_trans if self.structure_prediction else None,
                z_to_p_trans=self.z_to_p_trans if self.structure_prediction else None,
                c_to_p_trans_q=self.c_to_p_trans_q,
                c_to_p_trans_k=self.c_to_p_trans_k,
                p_mlp=self.p_mlp,
                feats=feats,
                s_trunk=s_trunk,
                z=z,
                structure_prediction=self.structure_prediction,
                W=self.atoms_per_window_queries,
                H=self.atoms_per_window_keys,
            )
        return q, c, p


class DiffusionConditioning1D(Module):
    """1D CP DiffusionConditioning for 2D mesh ``(dp, cp)``.

    Uses :class:`PairwiseConditioning1D` (2-element placements) and
    :class:`AtomEncoder1D` (local pair-to-atom gather) instead of the
    3D-mesh variants.

    Placements (2D mesh ``(dp, cp)``):

    - ``s_trunk``: ``(Shard(0), Shard(1))``
    - ``z_trunk``, ``relative_position_encoding``: ``(Shard(0), Shard(1))`` (row-slab)
    - ``q``, ``c``: ``(Shard(0), Shard(1))``
    - atom pair bias: ``(Shard(0), Shard(1))``
    - token pair bias: ``(Shard(0), Shard(1))``
    """

    def __init__(
        self,
        layer: SerialDiffusionConditioning,
        device_mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        assert isinstance(
            layer, SerialDiffusionConditioning
        ), f"Expected SerialDiffusionConditioning, got {type(layer)}"
        self.device_mesh = device_mesh
        self.atoms_per_window_queries = layer.atom_encoder.atoms_per_window_queries

        # Use 1D-specific modules
        self.pairwise_conditioner = PairwiseConditioning1D(
            layer=layer.pairwise_conditioner,
            device_mesh=device_mesh,
        )
        self.atom_encoder = AtomEncoder1D(
            layer=layer.atom_encoder,
            device_mesh=device_mesh,
        )

        # Bias projection layers (mesh-agnostic: just LayerNorm + Linear on last dim)
        self.atom_enc_proj_z = nn.ModuleList()
        for serial_seq in layer.atom_enc_proj_z:
            self.atom_enc_proj_z.append(
                nn.Sequential(
                    LayerNormParamsReplicated(serial_seq[0], device_mesh),
                    LinearParamsReplicated(serial_seq[1], device_mesh),
                )
            )

        self.atom_dec_proj_z = nn.ModuleList()
        for serial_seq in layer.atom_dec_proj_z:
            self.atom_dec_proj_z.append(
                nn.Sequential(
                    LayerNormParamsReplicated(serial_seq[0], device_mesh),
                    LinearParamsReplicated(serial_seq[1], device_mesh),
                )
            )

        self.token_trans_proj_z = nn.ModuleList()
        for serial_seq in layer.token_trans_proj_z:
            self.token_trans_proj_z.append(
                nn.Sequential(
                    LayerNormParamsReplicated(serial_seq[0], device_mesh),
                    LinearParamsReplicated(serial_seq[1], device_mesh),
                )
            )

    def forward(
        self,
        s_trunk: DTensor,
        z_trunk: DTensor,
        relative_position_encoding: DTensor,
        feats: dict[str, DTensor],
    ) -> tuple[DTensor, DTensor, DTensor, DTensor, DTensor]:
        """Forward pass.

        Parameters
        ----------
        s_trunk : DTensor
            Shape ``(B, N, token_s)`` with placements ``(Shard(0), Shard(1))``.
        z_trunk : DTensor
            Shape ``(B, N, N, token_z)`` with placements ``(Shard(0), Shard(1))``.
        relative_position_encoding : DTensor
            Shape ``(B, N, N, token_z)`` with placements ``(Shard(0), Shard(1))``.
        feats : dict[str, DTensor]
            Unpacked atom features.

        Returns
        -------
        tuple[DTensor, DTensor, DTensor, DTensor, DTensor]
            ``(q, c, atom_enc_bias, atom_dec_bias, token_trans_bias)``.
        """
        _keys_atom_features_packed = {
            "atom_pad_mask",
            "ref_pos",
            "ref_space_uid",
            "ref_charge",
            "ref_element",
            "ref_atom_name_chars",
            "atom_to_token",
        }
        feats_packed = pack_atom_features(feats, _keys_atom_features_packed, self.atoms_per_window_queries)

        # Pairwise conditioning with 1D placements
        z = self.pairwise_conditioner(z_trunk, relative_position_encoding)

        # Atom encoder with local gather for pair-to-atom
        q, c, p = self.atom_encoder(feats=feats_packed, s_trunk=s_trunk, z=z)

        # Bias projections (mesh-agnostic)
        atom_enc_bias_list = []
        for proj in self.atom_enc_proj_z:
            atom_enc_bias_list.append(proj(p))
        atom_enc_bias = shardwise_cat(atom_enc_bias_list, dim=-1)

        atom_dec_bias_list = []
        for proj in self.atom_dec_proj_z:
            atom_dec_bias_list.append(proj(p))
        atom_dec_bias = shardwise_cat(atom_dec_bias_list, dim=-1)

        token_trans_bias_list = []
        for proj in self.token_trans_proj_z:
            token_trans_bias_list.append(proj(z))
        token_trans_bias = shardwise_cat(token_trans_bias_list, dim=-1)

        return q, c, atom_enc_bias, atom_dec_bias, token_trans_bias
