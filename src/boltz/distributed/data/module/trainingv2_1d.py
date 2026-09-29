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

"""1D Context Parallelism training data module.

Mirrors ``trainingv2.py`` (2D CP) but uses a 2D ``(dp, cp)`` device mesh
instead of a 3D ``(dp, cp_axis_0, cp_axis_1)`` mesh.  Feature distribution
uses the ``distribute_features`` path (broadcast + ``distribute_tensor``)
with 1-element placement tuples from ``placements_1d.py``.
"""

from typing import Any, Optional

import pytorch_lightning as pl
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.utils.data import DataLoader, DistributedSampler

from boltz.data.module.trainingv2 import (
    Boltz2TrainingDataModule as Boltz2TrainingDataModuleSerial,
)
from boltz.data.module.trainingv2 import DataConfigV2
from boltz.data.pad import pad_dim
from boltz.distributed.data.module._error_propagation import (
    rank_zero_fetch_with_propagation,
)
from boltz.distributed.data.module.placements_1d import (
    FEATURE_AXIS_SEMANTICS_1D,
    TRAINING_FEATURE_PLACEMENTS_1D,
)
from boltz.distributed.data.utils import (
    NON_SHARDED_FEATURES_V2,
    CollateDTensor1D,
    distribute_features,
)


class _BaseDatasetCP1D(torch.utils.data.Dataset):
    """Wrap a serial Boltz2 dataset and distribute features as DTensors for 1D CP.

    Uses a 2D ``(dp, cp)`` device mesh.  The CP sub-mesh is a 1D mesh
    ``(cp,)`` and placements are 1-element tuples from the 1D placement
    registry.
    """

    def __init__(
        self,
        serial_dataset: torch.utils.data.Dataset,
        device_mesh: DeviceMesh,
        device_mesh_cpu: DeviceMesh,
    ) -> None:
        """Initialize the 1D CP distributed dataset wrapper.

        Parameters
        ----------
        serial_dataset : torch.utils.data.Dataset
            The serial (single-rank) Boltz2 dataset to wrap.
        device_mesh : DeviceMesh
            2D ``(dp, cp)`` device mesh for distributed tensor operations.
        device_mesh_cpu : DeviceMesh
            2D ``(dp_cpu, cp_cpu)`` CPU device mesh for data-loading
            collectives that run before GPU transfer.
        """
        super().__init__()
        self.serial_dataset = serial_dataset
        self.device_mesh = device_mesh
        self.device_mesh_cpu = device_mesh_cpu
        # 1D CP sub-mesh: single "cp_cpu" dimension
        self._cp_submesh = device_mesh_cpu["cp_cpu"]
        self._cp_submesh_group = self._cp_submesh.get_group(0)
        # In a 2D mesh (dp, cp), coordinate index 1 is the cp rank.
        self.is_cp_rank_zero = device_mesh.get_coordinate()[1] == 0

        self.feature_to_dtensor_placement = TRAINING_FEATURE_PLACEMENTS_1D

    def __len__(self) -> int:
        """Return the number of samples in the underlying serial dataset."""
        return len(self.serial_dataset)

    def _prepare_features_for_distribution(
        self, features: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Run the rank-0-only pre-broadcast prep that must propagate via the helper.

        Called inside the ``fetch_fn`` passed to
        :func:`rank_zero_fetch_with_propagation`, so any exception here
        (KeyError for unknown tensor keys, pad_dim failures, etc.) is
        caught and broadcast to peer CP ranks rather than leaving them
        blocked at the first ``broadcast_object_list`` inside
        :meth:`_distribute_features`.

        Returns
        -------
        tuple[dict[str, torch.Tensor], dict[str, Any]]
            ``(tensor_features_all, non_sharded_features)`` — both keyed
            for ``self.feature_to_dtensor_placement``.  Caller passes the
            tuple as a single "prepared" payload to :meth:`_distribute_features`.
        """
        if "token_pair_pad_mask" not in features and "token_pad_mask" in features:
            mask = features["token_pad_mask"]
            features["token_pair_pad_mask"] = mask[:, None] * mask[None, :]

        unknown_tensor_keys = sorted(
            key
            for key, value in features.items()
            if isinstance(value, torch.Tensor)
            and key not in self.feature_to_dtensor_placement
            and key not in NON_SHARDED_FEATURES_V2
        )
        if unknown_tensor_keys:
            raise KeyError(
                "Found tensor feature keys without DTensor placement mapping. "
                f"Please add placements for: {unknown_tensor_keys}"
            )

        tensor_features_all = {
            key: value
            for key, value in features.items()
            if isinstance(value, torch.Tensor)
            and key in self.feature_to_dtensor_placement
            and key not in NON_SHARDED_FEATURES_V2
        }

        # For 1D CP, there is a single CP dimension.
        cp_size = self._cp_submesh.shape[0]
        for key, tensor in tensor_features_all.items():
            placements = self.feature_to_dtensor_placement[key]
            padded = tensor
            for placement in placements:
                if not isinstance(placement, Shard):
                    continue
                shard_dim = placement.dim
                if shard_dim >= padded.ndim:
                    continue
                remainder = padded.shape[shard_dim] % cp_size
                if remainder == 0:
                    continue
                pad_len = cp_size - remainder
                padded = pad_dim(padded, shard_dim, pad_len, 0)
            tensor_features_all[key] = padded

        # The per-feature loop only pads the explicitly Shard()-marked
        # dimension.  Features that contain N_tokens (or N_atoms) on
        # *additional* axes — e.g. pair features whose column axis is
        # also N_tokens but is left unsharded, or atom_to_token whose
        # last axis is N_tokens — must be padded on those axes too,
        # otherwise the model forward sees a dim mismatch
        # (trunkv2.elementwise_op).  This mirrors the 2D-CP safeguard in
        # pad_and_scatter_atom_features_dtensor for atom_to_token's
        # trailing token dim.
        #
        # Axis targets come from the placement-registry-paired
        # ``FEATURE_AXIS_SEMANTICS_1D`` map rather than runtime size
        # matching: when ``N_tokens == N_atoms`` (rare, atom-resolution
        # small tests) the size-match heuristic cannot tell apart
        # token-axes from atom-axes and may silently pad the wrong axis.
        target_n_tokens = (
            int(tensor_features_all["token_pad_mask"].shape[0]) if "token_pad_mask" in tensor_features_all else None
        )
        target_n_atoms = (
            int(tensor_features_all["atom_pad_mask"].shape[0]) if "atom_pad_mask" in tensor_features_all else None
        )
        target_by_label = {"tokens": target_n_tokens, "atoms": target_n_atoms}
        for key, tensor in tensor_features_all.items():
            placements = self.feature_to_dtensor_placement[key]
            sharded_dims = {p.dim for p in placements if isinstance(p, Shard)}
            axis_labels = FEATURE_AXIS_SEMANTICS_1D[key]
            padded = tensor
            for dim in range(padded.ndim):
                if dim in sharded_dims:
                    continue
                label = axis_labels[dim] if dim < len(axis_labels) else None
                if label is None:
                    continue
                target = target_by_label.get(label)
                if target is None:
                    continue
                size = padded.shape[dim]
                if target > size:
                    padded = pad_dim(padded, dim, target - size, 0)
            tensor_features_all[key] = padded

        non_sharded_features = {key: value for key, value in features.items() if key in NON_SHARDED_FEATURES_V2}
        return tensor_features_all, non_sharded_features

    def _distribute_features(
        self,
        prepared: Optional[tuple[dict[str, torch.Tensor], dict[str, Any]]],
    ) -> dict[str, Any]:
        """Distribute pre-prepared features as DTensors across 1D CP ranks.

        All rank-0-only prep work (token_pair_pad_mask synthesis, unknown-key
        validation, pad_dim) must happen inside
        :meth:`_prepare_features_for_distribution` so it is covered by the
        :func:`rank_zero_fetch_with_propagation` envelope and cannot deadlock
        peers if rank-0 raises.

        Parameters
        ----------
        prepared : tuple[dict[str, torch.Tensor], dict[str, Any]] or None
            On CP rank zero: the ``(tensor_features_all, non_sharded_features)``
            tuple returned by :meth:`_prepare_features_for_distribution`.
            On peer CP ranks: ``None``.

        Returns
        -------
        dict[str, Any]
            Feature dictionary where tensor values are DTensors distributed
            according to ``self.feature_to_dtensor_placement``, and
            non-sharded values are broadcast copies.
        """
        cp_group_src_rank_global = min(torch.distributed.get_process_group_ranks(self._cp_submesh_group))

        if self.is_cp_rank_zero:
            tensor_features_all, non_sharded_features = prepared
            tensor_feature_keys = sorted(tensor_features_all.keys())
            keys_payload = [tensor_feature_keys]
            torch.distributed.broadcast_object_list(
                keys_payload,
                src=cp_group_src_rank_global,
                group=self._cp_submesh_group,
            )
            tensor_feature_keys_shared = keys_payload[0]
            all_placements = {key: self.feature_to_dtensor_placement[key] for key in tensor_feature_keys_shared}
            object_payload = [non_sharded_features]
        else:
            keys_payload = [None]
            torch.distributed.broadcast_object_list(
                keys_payload,
                src=cp_group_src_rank_global,
                group=self._cp_submesh_group,
            )
            tensor_feature_keys_shared = keys_payload[0]
            tensor_features_all = None
            all_placements = {key: self.feature_to_dtensor_placement[key] for key in tensor_feature_keys_shared}
            object_payload = [None]

        # For 1D CP, all features (atom, token, MSA) go through
        # distribute_features.  The 2D-only pad_and_scatter path requires
        # a square 2D mesh and cannot be used with a 1D CP submesh.
        features_dtensor = distribute_features(
            features=tensor_features_all,
            placements=all_placements,
            group=self._cp_submesh_group,
            src_rank_global=cp_group_src_rank_global,
            device_mesh=self._cp_submesh,
        )

        torch.distributed.broadcast_object_list(
            object_payload,
            src=cp_group_src_rank_global,
            group=self._cp_submesh_group,
        )

        features_dtensor.update(object_payload[0] or {})
        return features_dtensor


class TrainingDatasetCP1D(_BaseDatasetCP1D):
    """Training dataset with 1D DTensor context parallelism for Boltz2."""

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fetch and distribute a single training sample.

        CP rank zero retrieves the sample from the serial dataset and runs
        the pre-broadcast preparation (token_pair_pad_mask synthesis,
        unknown-key validation, pad_dim) — both inside the same
        :func:`rank_zero_fetch_with_propagation` envelope so any raise
        before the first collective propagates uniformly to peers
        instead of deadlocking them at the first ``broadcast_object_list``
        inside :meth:`_distribute_features`.

        Parameters
        ----------
        idx : int
            Sample index in the serial dataset.

        Returns
        -------
        dict[str, Any]
            Distributed feature dictionary with DTensor values.
        """

        def _fetch_and_prepare() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
            features = self.serial_dataset[idx]
            return self._prepare_features_for_distribution(features)

        cp_group_src_rank_global = min(torch.distributed.get_process_group_ranks(self._cp_submesh_group))
        prepared = rank_zero_fetch_with_propagation(
            fetch_fn=_fetch_and_prepare,
            is_cp_rank_zero=self.is_cp_rank_zero,
            cp_group=self._cp_submesh_group,
            cp_group_src_rank_global=cp_group_src_rank_global,
        )
        return self._distribute_features(prepared)


class ValidationDatasetCP1D(_BaseDatasetCP1D):
    """Validation dataset with 1D DTensor context parallelism for Boltz2."""

    def __init__(
        self,
        serial_dataset: torch.utils.data.Dataset,
        device_mesh: DeviceMesh,
        device_mesh_cpu: DeviceMesh,
        val_skip_sample_threshold_tokens: Optional[int] = None,
        val_skip_sample_threshold_atoms: Optional[int] = None,
        val_skip_sample_threshold_seqs: Optional[int] = None,
    ) -> None:
        """Initialize the 1D CP distributed validation dataset.

        Parameters
        ----------
        serial_dataset : torch.utils.data.Dataset
            The serial (single-rank) Boltz2 validation dataset to wrap.
        device_mesh : DeviceMesh
            2D ``(dp, cp)`` device mesh.
        device_mesh_cpu : DeviceMesh
            2D ``(dp_cpu, cp_cpu)`` CPU device mesh.
        val_skip_sample_threshold_tokens : int, optional
            Skip samples with more tokens than this.
        val_skip_sample_threshold_atoms : int, optional
            Skip samples with more atoms than this.
        val_skip_sample_threshold_seqs : int, optional
            Skip samples with more MSA sequences than this.
        """
        super().__init__(serial_dataset=serial_dataset, device_mesh=device_mesh, device_mesh_cpu=device_mesh_cpu)
        self.val_skip_sample_threshold_tokens = val_skip_sample_threshold_tokens
        self.val_skip_sample_threshold_atoms = val_skip_sample_threshold_atoms
        self.val_skip_sample_threshold_seqs = val_skip_sample_threshold_seqs

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fetch and distribute a single validation sample.

        On CP rank zero, iterates from ``idx`` through the dataset looking
        for a sample that satisfies all threshold constraints.  Other CP
        ranks receive the distributed DTensor features via collectives.

        Parameters
        ----------
        idx : int
            Starting sample index in the serial dataset.

        Returns
        -------
        dict[str, Any]
            Distributed feature dictionary with DTensor values.

        Raises
        ------
        RuntimeError
            If every sample in the dataset is filtered out.
        """

        def _scan_for_valid_sample() -> dict[str, Any]:
            num_items = len(self.serial_dataset)
            for shift in range(num_items):
                curr_idx = (idx + shift) % num_items
                features = self.serial_dataset[curr_idx]

                if self.val_skip_sample_threshold_tokens is not None and "token_pad_mask" in features:
                    tokens = int(features["token_pad_mask"].sum().item())
                    if tokens > self.val_skip_sample_threshold_tokens:
                        continue
                if self.val_skip_sample_threshold_atoms is not None and "atom_pad_mask" in features:
                    atoms = int(features["atom_pad_mask"].sum().item())
                    if atoms > self.val_skip_sample_threshold_atoms:
                        continue
                if self.val_skip_sample_threshold_seqs is not None and "msa_mask" in features:
                    msa_mask = features["msa_mask"]
                    seqs = int((msa_mask.sum(dim=1) > 0).sum().item()) if msa_mask.ndim == 2 else int(msa_mask.shape[0])
                    if seqs > self.val_skip_sample_threshold_seqs:
                        continue
                return features
            raise RuntimeError("All validation samples were filtered out by val_skip_sample_threshold_*")

        def _scan_and_prepare() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
            features = _scan_for_valid_sample()
            return self._prepare_features_for_distribution(features)

        cp_group_src_rank_global = min(torch.distributed.get_process_group_ranks(self._cp_submesh_group))
        prepared = rank_zero_fetch_with_propagation(
            fetch_fn=_scan_and_prepare,
            is_cp_rank_zero=self.is_cp_rank_zero,
            cp_group=self._cp_submesh_group,
            cp_group_src_rank_global=cp_group_src_rank_global,
        )
        return self._distribute_features(prepared)


class Boltz2TrainingDataModule1D(pl.LightningDataModule):
    """DataModule for Boltz2 distributed training with 1D DTensor CP.

    Uses a 2D ``(dp, cp)`` device mesh instead of the 3D
    ``(dp, cp_axis_0, cp_axis_1)`` mesh used by :class:`Boltz2TrainingDataModule`.
    """

    def __init__(
        self,
        cfg: DataConfigV2,
        device_mesh: DeviceMesh,
        device_mesh_cpu: DeviceMesh,
    ) -> None:
        """Initialize the 1D CP distributed training data module.

        Parameters
        ----------
        cfg : DataConfigV2
            The data configuration.
        device_mesh : DeviceMesh
            2D ``(dp, cp)`` device mesh for distributed tensor operations.
        device_mesh_cpu : DeviceMesh
            2D ``(dp_cpu, cp_cpu)`` CPU device mesh for data-loading
            collectives.

        Raises
        ------
        NotImplementedError
            If ``cfg.num_workers != 0``.
        """
        super().__init__()
        if cfg.num_workers != 0:
            raise NotImplementedError("num_workers != 0 is not supported for CP")

        self.cfg = cfg
        self.device_mesh = device_mesh
        self.device_mesh_cpu = device_mesh_cpu
        self._serial_module = Boltz2TrainingDataModuleSerial(cfg=cfg)
        self.val_group_mapper = self._serial_module.val_group_mapper

        self._train_set = TrainingDatasetCP1D(
            serial_dataset=self._serial_module._train_set,
            device_mesh=self.device_mesh,
            device_mesh_cpu=self.device_mesh_cpu,
        )
        self._val_set = ValidationDatasetCP1D(
            serial_dataset=self._serial_module._val_set,
            device_mesh=self.device_mesh,
            device_mesh_cpu=self.device_mesh_cpu,
            val_skip_sample_threshold_tokens=cfg.val_skip_sample_threshold_tokens,
            val_skip_sample_threshold_atoms=cfg.val_skip_sample_threshold_atoms,
            val_skip_sample_threshold_seqs=cfg.val_skip_sample_threshold_seqs,
        )

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: ARG002
        """No-op — serial module and CP-wrapped datasets are initialized in ``__init__``."""
        return

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader.

        Returns
        -------
        DataLoader
            Training dataloader with a ``DistributedSampler`` partitioned
            across data-parallel replicas and a ``CollateDTensor1D`` collate.
        """
        sampler = DistributedSampler(
            self._train_set,
            num_replicas=self.device_mesh_cpu.shape[0],
            rank=self.device_mesh_cpu.get_local_rank(0),
            shuffle=False,
            drop_last=False,
        )
        custom_collate = CollateDTensor1D(self.device_mesh_cpu)
        return DataLoader(
            self._train_set,
            sampler=sampler,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=custom_collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader.

        Returns
        -------
        DataLoader
            Validation dataloader with a ``DistributedSampler`` partitioned
            across data-parallel replicas and a ``CollateDTensor1D`` collate.
        """
        sampler = DistributedSampler(
            self._val_set,
            num_replicas=self.device_mesh_cpu.shape[0],
            rank=self.device_mesh_cpu.get_local_rank(0),
            shuffle=False,
            drop_last=False,
        )
        custom_collate = CollateDTensor1D(self.device_mesh_cpu)
        return DataLoader(
            self._val_set,
            sampler=sampler,
            batch_size=self.cfg.val_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=custom_collate,
        )

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict:
        """Transfer a batch from CPU DTensors to the target device.

        DTensor values are moved by extracting the local shard, transferring
        it to ``device``, and re-wrapping with the GPU ``device_mesh``.
        Plain tensors and lists of tensors are transferred directly.

        Parameters
        ----------
        batch : dict
            The batch to transfer.
        device : torch.device
            The target device (typically a CUDA device).
        dataloader_idx : int
            The dataloader index (unused).

        Returns
        -------
        dict
            The batch with all tensor values on ``device``.
        """
        for key, value in batch.items():
            if isinstance(value, DTensor):
                batch_local = value.to_local().to(device)
                batch[key] = DTensor.from_local(
                    batch_local,
                    device_mesh=self.device_mesh,
                    placements=value.placements,
                    shape=value.shape,
                    stride=value.stride(),
                )
            elif isinstance(value, list):
                batch[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in value]
            elif isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        return batch
