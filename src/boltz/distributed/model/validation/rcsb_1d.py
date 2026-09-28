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


"""Distributed RCSB validator for Boltz-2 with 1D context parallelism.

1D-CP uses a 2D device mesh ``(dp, cp)`` and the fused
:func:`boltz.distributed.model.loss.distogram_1d.distogram_loss_1d` instead of
the 2D-CP TransposeComm-based :func:`distogram_loss`. This subclass overrides
the methods that depend on ``transpose_comm`` so the 1D path does not have to
synthesise a no-op communicator.

MRO: Distributed1DRCSBValidator -> DistributedRCSBValidator ->
DistributedValidator -> RCSBValidator -> Validator.
"""

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.distributed.tensor import DTensor

from boltz.distributed.model.loss.distogram_1d import distogram_loss_1d
from boltz.distributed.model.validation.rcsb import DistributedRCSBValidator


class Distributed1DRCSBValidator(DistributedRCSBValidator):
    """Distributed RCSB validator specialised for 1D context parallelism.

    The 2D-CP validator threads a :class:`TransposeComm` through ``process`` ->
    ``common_val_step`` -> ``compute_disto_loss`` to drive the column-rotate
    used by the 2D distogram loss. 1D-CP has no TransposeComm; instead the
    fused 1D distogram loss reads ``device_mesh`` / ``dp_group`` / ``cp_group``
    off the model. This subclass:

    * drops the ``transpose_comm`` parameter from ``process`` (the 1D model's
      ``validation_step`` does not pass it), and
    * routes ``compute_disto_loss`` through :func:`distogram_loss_1d`.

    All other metric computation is inherited unchanged from
    :class:`DistributedRCSBValidator` / :class:`DistributedValidator`.
    """

    def process(
        self,
        model: LightningModule,
        batch: dict[str, DTensor],
        out: dict[str, DTensor],
        idx_dataset: int,
    ) -> None:
        """Compute features for the 1D-CP path.

        Mirrors :meth:`DistributedRCSBValidator.process` but without the
        ``transpose_comm`` argument. Delegates to ``common_val_step`` with
        ``transpose_comm=None``; the overridden :meth:`compute_disto_loss`
        ignores it and routes through ``distogram_loss_1d``.
        """
        symmetry_correction = model.val_group_mapper[idx_dataset]["symmetry_correction"]
        expand_to_diffusion_samples = symmetry_correction

        self.common_val_step(
            model,
            batch,
            out,
            idx_dataset,
            expand_to_diffusion_samples=expand_to_diffusion_samples,
            transpose_comm=None,
        )

    def compute_disto_loss(
        self,
        model: LightningModule,
        out: dict[str, DTensor],
        batch: dict[str, DTensor],
        idx_dataset: int,
        transpose_comm: None = None,
    ) -> Tensor:
        """Compute distogram loss using the 1D-CP fused implementation.

        Parameters
        ----------
        transpose_comm : None
            Accepted for signature compatibility with the 2D-CP parent;
            unused on the 1D path.
        """
        val_disto_loss, _ = distogram_loss_1d(
            out,
            batch,
            device_mesh=model._dist_device_mesh,
            dp_group=model.dp_group,
            cp_group=model.cp_group,
            aggregate_distogram=model.aggregate_distogram,
        )
        return val_disto_loss.to_local()
