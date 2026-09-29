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

"""Distributed transition and SwiGLU wrappers for 1D CP (2D mesh ``(dp, cp)``).

Transition1D and SwiGLU1D are elementwise over the feature dimension: the FFN
operates independently at each token position, so each CP rank processes its
local shard with no cross-rank communication.  Only parameter replication is
needed (via ``LinearParamsReplicated`` and ``LayerNormParamsReplicated``).

The serial ``chunk_size`` parameter is dropped: under CP each rank already
processes only N/cp tokens.

1D CP placements
----------------
Single repr ``s [B, N, C_s]``   -> ``(Shard(0), Shard(1))``
Pair repr   ``z [B, N, N, C_z]`` -> ``(Shard(0), Shard(1))``
MSA repr    ``m [B, S, N, C_m]`` -> ``(Shard(0), Shard(2))``

Communication budget
--------------------
Forward:  0 collectives.
Backward: 0 collectives (only implicit ``Partial(avg)`` on replicated param
          gradients, handled by ``LinearParamsReplicated`` /
          ``LayerNormParamsReplicated``).
"""

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from boltz.distributed.model.layers.elementwise_op import ElementwiseOp, elementwise_op
from boltz.distributed.model.layers.layernorm import LayerNormParamsReplicated
from boltz.distributed.model.layers.linear import LinearParamsReplicated
from boltz.distributed.model.layers.sigmoid_gate import sigmoid_gate
from boltz.distributed.model.layers.swiglu import SwiGLU
from boltz.model.layers.transition import Transition as SerialTransition


class Transition1D(nn.Module):
    """Distributed two-layer MLP (SwiGLU) for 1D CP on a 2D mesh ``(dp, cp)``.

    Wraps the serial ``Transition`` module.  All linear and layernorm parameters
    are replicated across the mesh; the elementwise SwiGLU activation requires
    no communication.

    Parameters
    ----------
    layer : SerialTransition
        Serial transition module whose parameters will be replicated.
    device_mesh : DeviceMesh
        2D device mesh ``(dp, cp)``.
    """

    def __init__(self, layer: SerialTransition, device_mesh: DeviceMesh) -> None:
        super().__init__()
        self.device_mesh = device_mesh
        self.hidden = layer.hidden

        self.norm = LayerNormParamsReplicated(layer.norm, device_mesh)
        self.fc1 = LinearParamsReplicated(layer.fc1, device_mesh)
        self.fc2 = LinearParamsReplicated(layer.fc2, device_mesh)
        self.fc3 = LinearParamsReplicated(layer.fc3, device_mesh)

    def forward(self, x: DTensor) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        x : DTensor
            Input of shape ``(..., D)`` with any 1D CP placement.

        Returns
        -------
        DTensor
            Output of shape ``(..., D)``, same placements as input.
        """
        x = self.norm(x)

        fc1_out = self.fc1(x)
        fc2_out = self.fc2(x)

        # SwiGLU: silu(fc1) * fc2, where silu(x) = x * sigmoid(x)
        x = sigmoid_gate(fc1_out, fc1_out)
        x = elementwise_op(x, fc2_out, ElementwiseOp.PROD)

        x = self.fc3(x)
        return x


class SwiGLU1D(SwiGLU):
    """SwiGLU activation for 1D CP on a 2D mesh ``(dp, cp)``.

    Identical to the 2D CP ``SwiGLU`` — the activation is elementwise and
    placement-agnostic.  This subclass exists for naming consistency with
    other ``*1D`` modules.
    """

    pass
